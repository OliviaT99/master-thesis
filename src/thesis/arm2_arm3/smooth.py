"""Budget-calibrated smooth allocation with exact gradients for a LINEAR score.

Phase 2, step 1. Implements memo v0.3 Eqs. 24-26 (calibrated entropy allocation)
and Eqs. 30-33 (implicit multiplier derivative) for g_theta(x) = x' theta, where
x already contains an intercept column if one is wanted. Because g is linear,
dr_i/dtheta = x_i / c_i and every gradient is available in closed form, so no
autograd framework is needed (decision D4: linear primary class).

Empirical objective on n customers with DR pseudo-outcomes Gamma_i:

    p_i(theta)   = sigmoid((g_i/c_i - delta - lambda(theta)) / T)
    lambda       = 0 if mean(c p) <= q at lambda = 0, else the root of mean(c p) = q
    V(theta)     = mean(Gamma_i p_i)
    R(theta)     = mean((Gamma_i - g_i)^2)        (= CATE risk + constant, memo s.17)
    J(theta)     = (1 - gamma) R - gamma V

Gradient (binding budget; memo Eqs. 30-32 with dr/dtheta = x/c):

    w_i          = p_i (1 - p_i)                  (computed in log space)
    dlambda      = sum_i w_i x_i / sum_i c_i w_i
    dp_i         = (w_i / T) (x_i / c_i - dlambda)
    dV           = mean(Gamma_i dp_i)

Slack budget: dlambda = 0 (Eq. 33). ``detach=True`` sets dlambda = 0 always: the
detached-threshold ABLATION, which is not the gradient of the calibrated
objective (memo s.13.4, s.16.4). At the binding/slack transition the map is
not differentiable; the binding branch is reported there.

Numerical conventions from the Stage-3 audit (9 Nov 2026): gradient weights use
log_sigmoid(z) + log_sigmoid(-z), never p(1-p) from rounded probabilities, and a
binding calibration whose total weight underflows raises instead of silently
returning a zero gradient. ``info["n_eff"]`` reports how many customers
effectively carry gradient weight, for monitoring during training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, log_expit

from thesis.evaluation.policy import smooth_allocation

# Below this total calibration weight (relative to n) the implicit derivative is
# numerically meaningless: T is too small for the data.
MIN_RELATIVE_WEIGHT = 1e-12


@dataclass
class SmoothState:
    p: np.ndarray       # treatment probabilities
    lam: float          # calibrated multiplier (0 if slack)
    binding: bool
    w: np.ndarray       # p (1 - p), from logits
    spend: float


def smooth_state(g, c, q, T, delta) -> SmoothState:
    """Calibrated allocation for scores g; lambda from the production calibrator."""
    p, lam = smooth_allocation(g, c, q, T, delta)
    z = (np.asarray(g) / c - delta - lam) / T
    w = np.exp(log_expit(z) + log_expit(-z))
    return SmoothState(p=expit(z), lam=float(lam), binding=lam > 0, w=w, spend=float(np.mean(c * p)))


class LinearSmoothObjective:
    """J(theta) = (1-gamma) mean((Gamma - X theta)^2) - gamma mean(Gamma p(theta))."""

    def __init__(self, X, gamma_pseudo, c, q, T, delta, gamma, detach=False):
        self.X = np.asarray(X, float)
        self.G = np.asarray(gamma_pseudo, float)
        self.c = np.asarray(c, float)
        if self.X.ndim != 2 or len(self.X) != len(self.G) or len(self.G) != len(self.c):
            raise ValueError("X must be n x d; Gamma and c must have length n")
        if (self.c <= 0).any() or not (T > 0) or delta < 0 or not (0 <= gamma <= 1):
            raise ValueError("need c > 0, T > 0, delta >= 0, gamma in [0, 1]")
        self.q, self.T, self.delta, self.gamma, self.detach = q, T, delta, gamma, detach
        self.Xc = self.X / self.c[:, None]          # dr_i / dtheta

    def state(self, theta) -> SmoothState:
        return smooth_state(self.X @ theta, self.c, self.q, self.T, self.delta)

    def value_and_grad(self, theta):
        theta = np.asarray(theta, float)
        n = len(self.G)
        g = self.X @ theta
        s = smooth_state(g, self.c, self.q, self.T, self.delta)
        resid = self.G - g
        risk = np.mean(resid ** 2)
        d_risk = -2.0 * self.X.T @ resid / n
        value = np.mean(self.G * s.p)
        if s.binding and not self.detach:
            denom = np.sum(self.c * s.w)
            if denom <= MIN_RELATIVE_WEIGHT * n:
                raise FloatingPointError(
                    f"calibration weight underflow (sum c*w = {denom:.3e}); T = {self.T:.3e} is too small for these scores")
            d_lam = self.X.T @ s.w / denom
        else:
            d_lam = np.zeros(self.X.shape[1])
        # dV = mean(Gamma * w / T * (x/c - dlambda))
        a = self.G * s.w / (n * self.T)
        d_value = self.Xc.T @ a - a.sum() * d_lam
        J = (1 - self.gamma) * risk - self.gamma * value
        dJ = (1 - self.gamma) * d_risk - self.gamma * d_value
        if not np.all(np.isfinite(dJ)):
            raise FloatingPointError(f"non-finite gradient at T = {self.T:.3e}")
        # Kish effective number of customers carrying gradient weight: small values
        # mean the smooth objective is close to the hard one and gradients are noisy.
        sw = s.w.sum()
        n_eff = sw ** 2 / np.sum(s.w ** 2) if sw > 0 else 0.0
        return J, dJ, dict(risk=risk, value=value, lam=s.lam, binding=s.binding, spend=s.spend, n_eff=n_eff)
