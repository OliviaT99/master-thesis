"""Smoke tests for the Arm 4 adapter. Skipped where torch/lightning or the local
Frauen et al. checkout are unavailable (e.g. CI); run on the thesis Mac."""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

from thesis.arm2_arm3.arm4_frauen import fit_frauen_ptcate, frauen_source  # noqa: E402

try:
    frauen_source()
except (FileNotFoundError, RuntimeError) as e:  # pragma: no cover
    pytest.skip(f"Frauen source unavailable: {e}", allow_module_level=True)

FAST = dict(epoch_init=2, epoch_prob=2, epoch_combined=2, batch_size=256)


def data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    tau = 0.03 + 0.04 * np.tanh(2 * X[:, 0])
    return X, tau + rng.normal(0, 1.0, n)


def test_runs_and_reaches_finished_phase():
    X, G = data()
    pred = fit_frauen_ptcate(X, G, gamma=0.8, config_overrides=FAST)
    g = pred(X)
    assert g.shape == (len(X),) and np.isfinite(g).all()
    assert pred.finished_phase == "finished"


def test_gamma_zero_is_a_linear_cate_fit():
    X, G = data()
    g = fit_frauen_ptcate(X, G, gamma=0.0, config_overrides=FAST)(X)
    # a linear network: predictions are exactly affine in X
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), g, rcond=None)
    assert np.allclose(np.column_stack([np.ones(len(X)), X]) @ coef, g, atol=1e-5)


def test_seeded_runs_are_reproducible():
    X, G = data()
    a = fit_frauen_ptcate(X, G, gamma=0.5, seed=3, config_overrides=FAST)(X)
    b = fit_frauen_ptcate(X, G, gamma=0.5, seed=3, config_overrides=FAST)(X)
    assert np.allclose(a, b)
