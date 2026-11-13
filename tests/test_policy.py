"""Allocator/evaluator tests. The hard allocator must agree with the finite-support
reference implementation (memo v0.3 experiment.py, `hard`) on expanded cohorts."""
import numpy as np
import pytest

from thesis.evaluation.policy import CostModel, dr_value, hard_allocation, paired_difference, smooth_allocation


def reference_hard(p, c, g, b):  # copied from experiment.py (finite-support reference)
    a = [0.0] * len(p); groups = {}
    for i in range(len(p)):
        if g[i] > 0:
            groups.setdefault(g[i] / c[i], []).append(i)
    remaining = b
    for r in sorted(groups, reverse=True):
        ix = groups[r]; mass = sum(p[i] * c[i] for i in ix)
        frac = min(1.0, max(0.0, remaining / mass))
        for i in ix: a[i] = frac
        remaining = max(0.0, remaining - frac * mass)
    return a


@pytest.mark.parametrize("g,b", [([1., 2., -1.], .5), ([1., 0., -1.], 1.), ([3., 2., 1.], .5),
                                  ([0., 0., 0.], .6), ([1., 2., 3.], 2.), ([1., 2., 3.], 0.), ([.2, -.2, -.6], 1/3)])
def test_matches_finite_support_reference(g, b):
    c = [1., 2., 3.]
    ref = reference_hard([1/3] * 3, c, g, b)
    m = 50  # 50 customers of each type -> cohort with masses 1/3
    G, C = np.repeat(g, m), np.repeat(c, m)
    a = hard_allocation(G, C, b)
    assert np.allclose(a.reshape(3, m).mean(1), ref, atol=1e-12)
    assert np.mean(C * a) <= b + 1e-12


def test_random_costs_budget_and_greedy_optimality():
    rng = np.random.default_rng(1)
    g, c = rng.normal(0.03, 0.05, 5000), rng.lognormal(0, 0.5, 5000)
    a = hard_allocation(g, c, 0.3)
    assert np.mean(c * a) == pytest.approx(min(0.3, np.mean(c * (g > 0))), abs=1e-9)
    assert np.all(a[g <= 0] == 0)
    r = g / c
    assert r[a == 1].min() >= r[(a == 0) & (g > 0)].max()


def test_smooth_converges_to_hard_and_is_feasible():
    rng = np.random.default_rng(2)
    g, c = rng.normal(0.03, 0.05, 4000), rng.lognormal(0, 0.5, 4000)
    h = hard_allocation(g, c, 0.3)
    p, lam = smooth_allocation(g, c, 0.3, T=1e-5, delta=5e-5)
    assert np.mean(c * p) <= 0.3 + 1e-9 and lam > 0
    assert np.mean(np.abs(p - h)) < 1e-3


def test_cost_model_mean_and_positivity():
    rng = np.random.default_rng(3)
    h = np.r_[0.0, rng.lognormal(6, 1, 9999)]
    cm = CostModel(1.0).fit(h)
    assert cm(h).mean() == pytest.approx(1.0) and cm(h).min() > 0
    assert np.allclose(CostModel(0.0).fit(h)(h), 1.0)


def test_paired_difference_se():
    gamma = np.array([1., -1., 2., 0.])
    d, se = paired_difference(np.ones(4), np.zeros(4), gamma)
    v, se_v = dr_value(np.ones(4), gamma)
    assert d == v and se == pytest.approx(se_v)


def test_hard_allocation_matches_independent_highs_lp():
    """Stage-2 method applied to the PRODUCTION allocator (audit 13 Nov 2026)."""
    from scipy.optimize import linprog
    rng = np.random.default_rng(20261113)
    for case in range(300):
        n = int(rng.choice([3, 10, 50, 300]))
        c = rng.uniform(0.2, 5, n)
        if case % 3 == 0:
            g = rng.choice([-1., 0., .25, 1., 2.], n) * c          # exact ratio ties
        elif case % 3 == 1:
            g = rng.uniform(-2, 2, n)                                # generic ratios
        else:
            g, c = rng.normal(0.03, 0.04, n), c / c.mean()           # X5-like scale
        q = rng.uniform(0, 1.2 * c.mean())
        a = hard_allocation(g, c, q)
        res = linprog(-g / n, A_ub=[c / n], b_ub=[q], bounds=(0, 1), method="highs")
        assert res.success
        assert abs(-res.fun - np.mean(g * a)) <= 1e-9 * max(1, abs(res.fun), q)
        assert np.mean(c * a) <= q + 1e-12 and np.all(a[g <= 0] == 0)
        for r in np.unique(g[g > 0] / c[g > 0]):                     # equal probability on ties
            assert np.ptp(a[(g > 0) & (g / c == r)]) == 0


@pytest.mark.parametrize("kappa", [0.02, 0.05])
def test_smooth_d2_regime_binds_and_stays_close_to_hard(kappa):
    """D2 operating regime: normalised costs, T = kappa * IQR(g/c), delta = 5T, q = 0.3."""
    rng = np.random.default_rng(7)
    h = rng.lognormal(6, 0.7, 40000)
    c = h / h.mean()
    g = rng.normal(0.031, 0.025, 40000)
    r = g / c
    T = kappa * np.subtract(*np.percentile(r, [75, 25]))
    p, lam = smooth_allocation(g, c, 0.3, T, 5 * T)
    assert lam > 0 and abs(np.mean(c * p) - 0.3) < 1e-12           # budget binds exactly
    assert np.mean(g * (hard_allocation(g, c, 0.3) - p)) < 2e-4      # < 0.2 per 1,000 estimated value
