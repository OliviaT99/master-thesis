"""End-to-end Phase 1 run on an X5-sized synthetic RCT with known heterogeneity."""
import numpy as np
import pandas as pd
import pytest

from thesis.phase1_signal_check import run


def make(n, seed, het=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12))
    base = 1 / (1 + np.exp(-(0.4 + 0.8 * X[:, 0] - 0.5 * X[:, 1])))
    tau = 0.033 + (0.06 * np.tanh(2 * X[:, 2]) * (X[:, 3] > 0) if het else 0)
    a = rng.binomial(1, 0.5, n)
    y = rng.binomial(1, np.clip(base + a * tau, 0, 1))
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(12)])
    df.insert(0, "client_id", [f"c{seed}_{i}" for i in range(n)])
    df.insert(1, "treatment_flg", a); df.insert(2, "target", y)
    df.insert(3, "cost_basket_raw", rng.lognormal(6, 0.6, n))
    return df


@pytest.mark.slow
def test_detects_heterogeneity_when_present():
    tables, s = run(make(100_000, 0), make(40_000, 1), propensity=0.5, folds=3)
    het = tables["heterogeneity"].set_index("score")
    assert het.loc["flexible_dr", "slope_t"] > 3
    assert abs(s["ate_validation_dr"] - 0.033) < 0.015
    print(tables["comparisons"].round(4).to_string()); print(s)


@pytest.mark.slow
def test_no_false_heterogeneity_when_absent():
    tables, s = run(make(100_000, 2, het=False), make(40_000, 3, het=False), propensity=0.5, folds=3)
    het = tables["heterogeneity"].set_index("score")
    assert abs(het.loc["linear_dr", "slope_t"]) < 3
    print(het.round(3).to_string()); print(s)
