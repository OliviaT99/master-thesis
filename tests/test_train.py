"""Phase 2 step 2: Arm 3 trainer and validation selection."""
import numpy as np
import pytest
from sklearn.linear_model import Ridge

from thesis.arm2_arm3.train import (DELTA_OVER_T, deploy, design, fit_budget_ptcate, ridge_fit,
                                     select_on_validation, temperature)
from thesis.arm2_arm3.smooth import LinearSmoothObjective


def misspecified(n, seed):
    """Nonlinear true effect, linear class; basket-like heterogeneous costs."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    tau = 0.03 + 0.04 * np.tanh(2 * X[:, 0]) * (X[:, 1] > 0) + 0.01 * X[:, 2]
    c = np.exp(0.5 * X[:, 3]); c /= c.mean()
    G = tau + rng.normal(0, 1.0, n)               # DR-like noisy pseudo-outcomes
    return X, G, c, tau


def test_ridge_fit_matches_sklearn():
    X, G, _, _ = misspecified(5000, 0)
    ours = ridge_fit(design(X), G, alpha=300.0)
    sk = Ridge(alpha=300.0).fit(X, G)
    assert np.allclose(ours[1:], sk.coef_, atol=1e-12) and ours[0] == pytest.approx(sk.intercept_, abs=1e-12)


def test_gamma_zero_training_recovers_arm2_ridge():
    X, G, c, _ = misspecified(5000, 1)
    Xd = design(X)
    theta0 = ridge_fit(Xd, G, alpha=300.0)
    start = theta0 + np.random.default_rng(2).normal(0, 0.05, len(theta0))
    theta, stages = fit_budget_ptcate(Xd, G, c, 0.3, gamma=0.0, T=0.01, alpha=300.0, theta_init=start)
    assert np.allclose(theta, theta0, atol=1e-7) and stages[-1]["success"]


def boundary_population(m=200):
    """Memo s.16.2 embedded as a linear class: g = 1*A + theta*B, theta in [0, .9]."""
    A, B = np.repeat([.2, .4, -.2], m), np.repeat([.2, -.4, 0.], m)
    Xd = np.column_stack([A, B])              # no intercept column; alpha = 0 so nothing is penalised
    tau, c = np.repeat([.6, .8, .2], m), np.repeat([1., 2., 3.], m)
    return Xd, tau, c


@pytest.mark.parametrize("T", [1e-3, 1e-4])
def test_boundary_example_implicit_escapes_detached_stays(T):
    """Memo s.16.4: from the CATE projection theta=0 the implicit gradient moves to
    theta>0 (smooth value -> 4/15); the detached ablation stays at 0 (value 7/30)."""
    Xd, tau, c = boundary_population()
    kw = dict(Xd=Xd, G=tau, c=c, q=.5, gamma=.8, T=T, alpha=0.0, theta_init=np.array([1., 0.]),
              anneal=(1.0,), free=[False, True], bounds=[(0.0, 0.9)])
    th_imp, st_imp = fit_budget_ptcate(**kw)
    th_det, st_det = fit_budget_ptcate(**kw, detach=True)
    assert th_imp[1] > 0 and st_imp[-1]["value"] > 0.26        # 4/15 = 0.2667
    assert th_det[1] == 0.0 and st_det[-1]["value"] == pytest.approx(7 / 30, abs=1e-6)


def test_training_improves_own_objective_under_misspecification():
    X, G, c, tau = misspecified(20000, 3)
    Xd = design(X)
    theta0 = ridge_fit(Xd, G, alpha=100.0)
    T = temperature(theta0, Xd, c, 0.05)
    theta, stages = fit_budget_ptcate(Xd, G, c, 0.3, gamma=0.9, T=T, alpha=100.0, theta_init=theta0)
    obj = LinearSmoothObjective(Xd, G, c, 0.3, T, DELTA_OVER_T * T, 0.9)
    pen = lambda th: 100.0 / len(G) * np.sum(th[1:] ** 2)
    assert obj.value_and_grad(theta)[0] + pen(theta) < obj.value_and_grad(theta0)[0] + pen(theta0)
    assert all(s["binding"] for s in stages) and stages[-1]["n_eff"] > 100
    assert abs(stages[-1]["spend"] - 0.3) < 1e-12


@pytest.mark.slow
def test_select_on_validation_bookkeeping():
    Xtr, Gtr, ctr, _ = misspecified(20000, 4)
    Xva, Gva, cva, _ = misspecified(8000, 5)
    table, fits, theta0, best = select_on_validation(Xtr, Gtr, ctr, Xva, Gva, cva, q=0.3, alpha=100.0,
                                                     gammas=[0.0, 0.5, 0.9], kappas=[0.02, 0.05])
    assert len(table) == 6 and best["val_value_smooth"] == table["val_value_smooth"].max()
    assert (table.loc[table.gamma == 0, "coef_change"] == 0).all()
    assert table.loc[table.gamma > 0, "converged"].all()
    # gamma = 0 rows are Arm 2 deployed through the same smooth operator
    for kappa in (0.02, 0.05):
        f = fits[(kappa, 0.0)]
        p = deploy(theta0, design(Xva), cva, 0.3, f.T)["smooth"]
        row = table[(table.kappa == kappa) & (table.gamma == 0)].iloc[0]
        assert np.mean(p * Gva) == pytest.approx(row.val_value_smooth, abs=1e-15)
