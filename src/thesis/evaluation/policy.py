"""Costs, budget allocation and DR policy evaluation, shared by all arms.

Conventions follow memo v0.3 section 3 and decisions D1/D2 (30 Oct 2026):

* Costs are dimensionless, c~ = c / mean_train(c), so the per-customer budget
  is b = q ("what treating a fraction q at average cost would spend").
* c_k = (1 - k) + k * h~ / mean_train(h~), with h~ = max(h, floor) and the floor
  and mean frozen from TRAINING data (memo section 3.2 positivity rule).
* hard_allocation: estimated-benefit-optimal fractional allocation (memo Eq. 9):
  rank positive scores by g/c, fill the budget, equal fraction on an exact tie
  group at the boundary, never treat g <= 0, unused budget allowed.
* smooth_allocation: memo Eq. 25 at fixed T, delta, with lambda calibrated on the
  same cohort (bisection; lambda = 0 when the budget is slack).
* Policies here are calibrated on the evaluation cohort's covariates
  (a cohort-level allocation, memo section 17 caveat); outcomes are never used.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit


class CostModel:
    """Scenario cost c_k(x) from a raw basket value; fitted on training data only."""

    def __init__(self, k: float, floor_quantile: float = 0.01):
        if not 0 <= k <= 1:
            raise ValueError("k must be in [0, 1]")
        self.k, self.floor_quantile = k, floor_quantile

    def fit(self, h_train):
        h_train = np.asarray(h_train, float)
        self.floor_ = float(np.quantile(h_train, self.floor_quantile))
        if self.floor_ <= 0:
            raise ValueError("floor must be positive")
        self.mean_ = float(np.maximum(h_train, self.floor_).mean())
        return self

    def __call__(self, h):
        h_tilde = np.maximum(np.asarray(h, float), self.floor_)
        return (1 - self.k) + self.k * h_tilde / self.mean_


def hard_allocation(g, c, q):
    g, c = np.asarray(g, float), np.asarray(c, float)
    n, budget = len(g), q * len(g)  # per-customer budget q -> cohort budget q*n
    a = np.zeros(n)
    pos = np.flatnonzero(g > 0)
    if budget <= 0 or pos.size == 0:
        return a
    r = g[pos] / c[pos]
    order = pos[np.argsort(-r, kind="stable")]
    r_sorted = g[order] / c[order]
    # group exact ties of the ratio (no tolerance; memo section 16.5)
    starts = np.flatnonzero(np.r_[True, r_sorted[1:] != r_sorted[:-1]])
    ends = np.r_[starts[1:], len(order)]
    remaining = budget
    for s, e in zip(starts, ends):
        idx = order[s:e]
        cost = c[idx].sum()
        frac = min(1.0, remaining / cost)
        a[idx] = frac
        remaining -= frac * cost
        if remaining <= 0:
            break
    return a


def smooth_allocation(g, c, q, T, delta):
    """p = sigmoid((g/c - delta - lam)/T), lam >= 0 calibrated so mean(c p) <= q.

    Root found with Brent's method (about 10-40 spend evaluations instead of 200
    bisection steps), then nudged up by single floating-point steps until the
    spend is feasible, so the budget is never exceeded by rounding.
    """
    from scipy.optimize import brentq

    g, c = np.asarray(g, float), np.asarray(c, float)
    if q <= 0:
        return np.zeros_like(g), np.inf
    r = g / c
    spend = lambda lam: np.mean(c * expit((r - delta - lam) / T))
    if spend(0.0) <= q:
        return expit((r - delta) / T), 0.0
    hi = max(1.0, r.max() - delta + 50 * T)
    while spend(hi) > q:                      # bracket (only for extreme inputs)
        hi *= 2
    lam = brentq(lambda l: spend(l) - q, 0.0, hi, xtol=1e-300, rtol=4 * np.finfo(float).eps, maxiter=500)
    for _ in range(64):
        if spend(lam) <= q:
            break
        lam = np.nextafter(lam, np.inf)
    else:
        raise ArithmeticError("could not reach a feasible multiplier")
    return expit((r - delta - lam) / T), float(lam)


def dr_value(pi, gamma):
    """Incremental value per eligible customer: mean(pi * Gamma), with its SE."""
    contrib = np.asarray(pi) * np.asarray(gamma)
    return contrib.mean(), contrib.std(ddof=1) / np.sqrt(len(contrib))


def paired_difference(pi_a, pi_b, gamma):
    """V(a) - V(b) with the paired SE of (pi_a - pi_b) * Gamma (fixed policies)."""
    d = (np.asarray(pi_a) - np.asarray(pi_b)) * np.asarray(gamma)
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))
