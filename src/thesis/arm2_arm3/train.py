"""Phase 2, step 2: training and validation selection for budget PT-CATE (Arm 3).

Model class G (decision D4): linear scores g(x) = theta_0 + x' theta_1 on the
standardised features, with a FIXED ridge penalty alpha/n * ||theta_1||^2. The
penalty is part of the class definition, so it does not change with gamma:

    J_gamma(theta) = (1-gamma) mean((Gamma - g)^2) - gamma V_T(theta) + alpha/n ||theta_1||^2

* gamma = 0 is exactly sklearn Ridge(alpha) on the DR pseudo-outcomes (Arm 2).
* V_T is the calibrated smooth policy value from ``smooth.LinearSmoothObjective``.
* Temperature (decision D2): T = kappa * IQR(g0(X)/c), with g0 the gamma = 0 fit on
  training data, delta = 5 T. Training anneals T through ``anneal`` multiples of
  the target and warm-starts every stage; the final stage is the deployment T.
* Optimiser: L-BFGS-B with the exact gradient. The objective is nonconvex:
  results are local optima from the Arm 2 start (Frauen's step 1), not global ones.
* Selection: gamma and kappa maximise the validation DR value of the deployed
  smooth policy (decision D1), with the policy calibrated on the validation
  cohort's covariates. The test split is never used here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from thesis.arm2_arm3.smooth import LinearSmoothObjective
from thesis.evaluation.policy import dr_value, hard_allocation, smooth_allocation

DELTA_OVER_T = 5.0
ANNEAL = (4.0, 2.0, 1.0)


def design(X):
    """Prepend an intercept column; the intercept is never penalised."""
    X = np.asarray(X, float)
    return np.column_stack([np.ones(len(X)), X])


def ridge_fit(Xd, G, alpha):
    """Closed-form ridge with unpenalised intercept (== sklearn Ridge on centred data)."""
    n, d = Xd.shape
    P = alpha * np.eye(d)
    P[0, 0] = 0.0
    return np.linalg.solve(Xd.T @ Xd + P, Xd.T @ G)


def temperature(theta0, Xd, c, kappa):
    r = (Xd @ theta0) / c
    iqr = np.subtract(*np.percentile(r, [75, 25]))
    if not iqr > 0:
        raise ValueError("gamma=0 ratios have zero spread; cannot set T")
    return kappa * iqr


@dataclass
class FitResult:
    theta: np.ndarray
    gamma: float
    kappa: float
    T: float
    stages: list = field(default_factory=list)


def fit_budget_ptcate(Xd, G, c, q, gamma, T, alpha, theta_init, detach=False,
                      anneal=ANNEAL, maxiter=500, free=None, bounds=None):
    """Minimise J_gamma from theta_init through the annealing schedule.

    free: optional boolean mask; False entries stay at theta_init (used to embed
    the memo's one-parameter examples). bounds: optional list of (lo, hi) for the
    free coordinates (L-BFGS-B box constraints).
    """
    theta = np.asarray(theta_init, float).copy()
    free = np.ones(len(theta), bool) if free is None else np.asarray(free, bool)
    pen = np.full(len(theta), alpha / len(G)); pen[0] = 0.0
    stages = []
    for f in anneal:
        Ts = f * T
        obj = LinearSmoothObjective(Xd, G, c, q, Ts, DELTA_OVER_T * Ts, gamma, detach=detach)

        def fun(z):
            th = theta.copy(); th[free] = z
            J, g, _ = obj.value_and_grad(th)
            return J + np.sum(pen * th ** 2), (g + 2 * pen * th)[free]

        res = minimize(fun, theta[free], jac=True, method="L-BFGS-B", bounds=bounds,
                       options=dict(maxiter=maxiter, ftol=1e-15, gtol=1e-12, maxcor=20))
        theta[free] = res.x
        _, _, info = obj.value_and_grad(theta)
        stages.append(dict(T=Ts, J=float(res.fun), nit=int(res.nit), success=bool(res.success),
                           message=str(res.message), **{k: info[k] for k in ("risk", "value", "lam", "binding", "spend", "n_eff")}))
    return theta, stages


def deploy(theta, Xd, c, q, T):
    """Deployed policies on a cohort: smooth (primary, D1) and hard (secondary)."""
    g = Xd @ theta
    p, _ = smooth_allocation(g, c, q, T, DELTA_OVER_T * T)
    return dict(smooth=p, hard=hard_allocation(g, c, q))


def select_on_validation(Xtr, Gtr, ctr, Xva, Gva, cva, q, alpha, gammas, kappas, detach=False):
    """Fit every (gamma, kappa) on train; score the deployed smooth policy on validation."""
    Xd_tr, Xd_va = design(Xtr), design(Xva)
    theta0 = ridge_fit(Xd_tr, Gtr, alpha)                    # Arm 2 and the common start
    rows, fits = [], {}
    for kappa in kappas:
        T = temperature(theta0, Xd_tr, ctr, kappa)
        for gamma in gammas:
            if gamma == 0:
                theta, stages = theta0.copy(), []
            else:
                theta, stages = fit_budget_ptcate(Xd_tr, Gtr, ctr, q, gamma, T, alpha, theta0, detach=detach)
            pol = deploy(theta, Xd_va, cva, q, T)
            v_s, se_s = dr_value(pol["smooth"], Gva)
            v_h, se_h = dr_value(pol["hard"], Gva)
            last = stages[-1] if stages else {}
            rows.append(dict(kappa=kappa, gamma=gamma, T=T, val_value_smooth=v_s, val_se_smooth=se_s,
                             val_value_hard=v_h, val_se_hard=se_h,
                             train_value=last.get("value"), train_risk=last.get("risk"),
                             n_eff=last.get("n_eff"), converged=all(s["success"] for s in stages),
                             coef_change=float(np.linalg.norm(theta - theta0))))
            fits[(kappa, gamma)] = FitResult(theta=theta, gamma=gamma, kappa=kappa, T=T, stages=stages)
    table = pd.DataFrame(rows)
    best = table.loc[table["val_value_smooth"].idxmax()]
    return table, fits, theta0, best
