"""Phase 2 step 1: smooth budget operator and exact gradients for a linear score."""
import math

import numpy as np
import pytest

from thesis.arm2_arm3.smooth import LinearSmoothObjective, smooth_state
from thesis.evaluation.policy import smooth_allocation


def random_problem(n=3000, d=6, seed=0, q=0.3):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, d - 1))])
    c = rng.lognormal(0, 0.5, n); c /= c.mean()
    G = 0.03 + 0.03 * X[:, 1] + rng.normal(0, 1.0, n)        # noisy DR-like pseudo-outcomes
    theta = np.r_[0.03, 0.02, rng.normal(0, 0.01, d - 2)]
    return X, G, c, q, theta


def fd_grad(obj, theta, h=1e-6):
    out = np.empty_like(theta)
    for j in range(len(theta)):
        e = np.zeros_like(theta); e[j] = h
        out[j] = (obj.value_and_grad(theta + e)[0] - obj.value_and_grad(theta - e)[0]) / (2 * h)
    return out


@pytest.mark.parametrize("gamma", [0.0, 0.5, 0.98])
@pytest.mark.parametrize("q", [0.3, 5.0])                     # binding and slack budgets
def test_gradient_matches_finite_differences(gamma, q):
    X, G, c, _, theta = random_problem(q=q)
    T = 0.05 * np.subtract(*np.percentile(X @ theta / c, [75, 25]))
    obj = LinearSmoothObjective(X, G, c, q, T, 5 * T, gamma)
    J, grad, info = obj.value_and_grad(theta)
    assert info["binding"] == (q == 0.3)
    fd = fd_grad(obj, theta)
    assert np.max(np.abs(grad - fd)) <= 1e-6 * max(1.0, np.max(np.abs(fd)))


def test_state_matches_production_allocator():
    X, G, c, q, theta = random_problem()
    g, T = X @ theta, 0.002
    s = smooth_state(g, c, q, T, 5 * T)
    p, lam = smooth_allocation(g, c, q, T, 5 * T)
    assert s.lam == lam and np.allclose(s.p, p, atol=1e-15) and abs(s.spend - q) < 1e-12
    assert np.allclose(s.w, p * (1 - p), atol=1e-15)


def test_gamma_zero_is_ols_on_pseudo_outcomes():
    X, G, c, q, _ = random_problem()
    ols = np.linalg.lstsq(X, G, rcond=None)[0]
    _, grad, _ = LinearSmoothObjective(X, G, c, q, 1e-3, 5e-3, gamma=0.0).value_and_grad(ols)
    assert np.max(np.abs(grad)) < 1e-12


def test_common_ratio_shift_has_zero_value_gradient_when_binding():
    """Homogeneous costs: intercept shifts every ratio equally; lambda absorbs it (memo s.13.4)."""
    X, G, _, q, theta = random_problem()
    c = np.ones(len(G))
    T = 0.002
    implicit = LinearSmoothObjective(X, G, c, q, T, 5 * T, gamma=1.0)
    detached = LinearSmoothObjective(X, G, c, q, T, 5 * T, gamma=1.0, detach=True)
    assert implicit.value_and_grad(theta)[2]["binding"]
    assert abs(implicit.value_and_grad(theta)[1][0]) < 1e-12        # intercept component
    assert abs(detached.value_and_grad(theta)[1][0]) > 1e-3


def boundary_example(T):
    """Memo s.16.2: three equal-mass types, c=(1,2,3), tau=(.6,.8,.2), b=.5, gamma=.8.
    g_theta = A + theta B with A=(.2,.4,-.2), B=(.2,-.4,0); X = [A, B], coefficient on A fixed at 1."""
    m = 200
    A, B = np.repeat([.2, .4, -.2], m), np.repeat([.2, -.4, 0.], m)
    X = np.column_stack([A, B])
    tau, c = np.repeat([.6, .8, .2], m), np.repeat([1., 2., 3.], m)
    kw = dict(X=X, gamma_pseudo=tau, c=c, q=.5, T=T, delta=math.sqrt(T), gamma=.8)
    return LinearSmoothObjective(**kw), LinearSmoothObjective(**kw, detach=True)


@pytest.mark.parametrize("T", [1e-3, 1e-4])
def test_boundary_example_reproduces_memo_derivatives(T):
    """Memo s.16.4: J'_implicit(0) = 4/375 - 4/(1125 T), J'_detached(0) = 4/375 + 1/(375 T)
    (+ exponentially small terms). Implicit descent leaves theta=0; detached stays."""
    implicit, detached = boundary_example(T)
    theta0 = np.array([1.0, 0.0])
    gi = implicit.value_and_grad(theta0)[1][1]
    gd = detached.value_and_grad(theta0)[1][1]
    assert gi == pytest.approx(4 / 375 - 4 / (1125 * T), rel=1e-3)
    assert gd == pytest.approx(4 / 375 + 1 / (375 * T), rel=1e-3)
    assert gi < 0 < gd


def test_underflow_guard_raises_at_exactly_affordable_prefix():
    """Stage-3 failure mode: the budget exactly buys a prefix, so at tiny T every
    calibration weight underflows. Must raise, not return a zero gradient."""
    n = 1000
    X = np.column_stack([np.ones(n), (np.arange(n) < 300).astype(float)])
    theta = np.array([0.01, 0.04])                   # g = 0.05 for 30%, 0.01 for the rest
    G, c = np.full(n, 0.03), np.ones(n)
    ok = LinearSmoothObjective(X, G, c, 0.3, 1e-3, 5e-3, gamma=0.8).value_and_grad(theta)
    assert ok[2]["binding"] and ok[2]["n_eff"] > 100
    with pytest.raises(FloatingPointError):
        LinearSmoothObjective(X, G, c, 0.3, 1e-6, 5e-6, gamma=0.8).value_and_grad(theta)


def test_input_validation():
    X, G, c, q, _ = random_problem(n=10)
    with pytest.raises(ValueError):
        LinearSmoothObjective(X, G, -c, q, 0.01, 0.05, 0.5)
    with pytest.raises(ValueError):
        LinearSmoothObjective(X, G[:5], c, q, 0.01, 0.05, 0.5)
