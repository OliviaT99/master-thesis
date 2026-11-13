"""Arm 4: Frauen et al.'s original PT-CATE (zero threshold, adaptive alpha network),
followed by the same budget allocator as every other arm (decision D3).

Licensing: the upstream repository (github.com/DennisFrauen/CATEForPolicy,
commit 14a549b, 20 Oct 2025) has no licence file, so its code is NOT copied into
this repository. The class is imported from a local checkout at run time:

    $FRAUEN_SRC, or by default ../CATEForPolicy-FRAUEN next to this repo.

The imported models/neural_nets.py is checked against the SHA-256 of the audited
upstream file; a mismatch raises, so a modified copy cannot silently change Arm 4.

Configuration: Frauen's own real-data (Hillstrom) setting, experiments/configs/
sim_real/hyperparams/ptcate_dr.yaml: linear CATE network, nonlinear alpha network,
Adam weight decay 0.0247, 10/10/10 epochs, K=1, batch 64, lr 1e-4.

Deviations from their pipeline, made so that all arms share inputs (D3):
* pseudo-outcomes are OUR cross-fitted DR Gamma with the known RCT propensity
  (their config estimates the propensity and does not cross-fit);
* features are OUR standardised pre-treatment features;
* their weight decay 0.0247 is kept as published, so Arm 4's restriction is
  their L2 setting, not the shared ridge alpha of Arms 2/3 (reported in the thesis).

Requires torch and lightning (Frauen pinned torch 2.5.1 / lightning 2.4.0; this
adapter was written against torch 2.15 / lightning 2.6.6, so it must be smoke-tested).
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np

UPSTREAM_SHA256 = "e5b3ef9e960d9c5d2bedba68784e031d01a781dc0f8cd7f3840200c8f40efacd"
REPO_ROOT = Path(__file__).resolve().parents[3]

# experiments/configs/sim_real/hyperparams/ptcate_dr.yaml (Frauen et al.)
FRAUEN_REAL_DR_CONFIG = dict(
    dropout=0.0, hidden_dim_cate=8, hidden_dim_prob=8,
    lr_init=1e-4, lr_prob=1e-4, lr_combined=1e-4,
    epoch_init=10, epoch_prob=10, epoch_combined=10, n_cycles=1, batch_size=64,
    l2_reg=0.0247, complexity_cate="linear", complexity_prob="nonlinear",
    prob_lower=5, prob_upper=None, weighting=True, alpha_bce=True,
    alpha_softplus=True, cate_tanh=True,
)


def frauen_source() -> Path:
    path = Path(os.environ.get("FRAUEN_SRC", REPO_ROOT.parent / "CATEForPolicy-FRAUEN"))
    nn_file = path / "models" / "neural_nets.py"
    if not nn_file.exists():
        raise FileNotFoundError(f"Frauen et al. code not found at {nn_file}; set FRAUEN_SRC")
    digest = hashlib.sha256(nn_file.read_bytes()).hexdigest()
    if digest != UPSTREAM_SHA256:
        raise RuntimeError(f"{nn_file} differs from audited upstream (sha256 {digest})")
    return nn_file


def load_ptcate_class():
    spec = importlib.util.spec_from_file_location("frauen_neural_nets", frauen_source())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PTCATENet


def fit_frauen_ptcate(X, gamma_pseudo, gamma, seed=0, config_overrides=None):
    """Train Frauen's PTCATENet on (X, Gamma); return a predict(X) -> g function.

    gamma = 0 stops after the initial CATE phase (their own convention), which is
    their linear DR-learner with Adam weight decay.
    """
    import lightning.pytorch as pl
    import torch
    from torch.utils.data import DataLoader, Dataset

    X = np.asarray(X, np.float32)
    G = np.asarray(gamma_pseudo, np.float32).reshape(-1, 1)

    class _Data(Dataset):
        def __len__(self):
            return len(X)

        def __getitem__(self, i):
            return {"data": {"x": torch.from_numpy(X[i]), "y_pseudo": torch.from_numpy(G[i])}}

    cfg = dict(FRAUEN_REAL_DR_CONFIG, **(config_overrides or {}))
    cfg.update(input_dim=X.shape[1], gamma=float(gamma), logging=False)
    pl.seed_everything(seed, workers=True, verbose=False)
    model = load_ptcate_class()(cfg)
    # Same epoch count as their training.py
    epochs = cfg["epoch_init"] + cfg["n_cycles"] * (cfg["epoch_prob"] + cfg["epoch_combined"])
    trainer = pl.Trainer(max_epochs=epochs, accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False, deterministic=True,
                         num_sanity_val_steps=0)
    loader = DataLoader(_Data(), batch_size=cfg["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    trainer.fit(model, train_dataloaders=loader)
    model.eval()

    def predict(Xnew):
        with torch.no_grad():
            return model.network_cate(torch.as_tensor(np.asarray(Xnew, np.float32))).numpy().ravel().astype(float)

    predict.finished_phase = model.get_current_phase()
    return predict
