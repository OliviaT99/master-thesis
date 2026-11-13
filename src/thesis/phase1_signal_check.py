"""Phase 1: X5 signal and power check (review plan, 30 Oct 2026).

Question: does X5 contain enough detectable effect heterogeneity for policy
targeting to matter, and can differences between policies of the size we expect
be distinguished from noise on a held-out split?

Uses TRAIN to fit everything and VALIDATION to evaluate. The test split is not
read. Results are exploratory (several budgets/costs are inspected), so they
inform the Gate A/B decision but are not thesis results.

Outputs (results/tables/phase1_*):
  * heterogeneity.csv  - BLP test per score (Chernozhukov et al., generic ML
                         inference on heterogeneous effects): slope beta2 on the
                         demeaned score; beta2 > 0 significant => detectable
                         heterogeneity; beta2 ~ 1 => well calibrated.
  * gates.csv          - mean DR pseudo-outcome per score quintile (GATES).
  * policy_values.csv  - DR value per customer for each policy, budget q, cost k.
  * comparisons.csv    - paired differences with SEs (the Gate A/B evidence).
  * summary.json

Run from the repo root:  PYTHONPATH=src python -m thesis.phase1_signal_check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

from thesis.evaluation.policy import CostModel, dr_value, hard_allocation, paired_difference
from thesis.nuisance import NuisanceFit

NON_FEATURES = ["client_id", "treatment_flg", "target", "cost_basket_raw"]
BUDGETS = [0.1, 0.3, 0.5]
COST_K = [0.0, 1.0]
COMPARISONS = [  # (a, b): question
    ("flexible_dr", "linear_dr", "headroom: does flexibility beat the linear class?"),
    ("linear_dr", "cost_only", "does the linear CATE beat cost-only targeting?"),
    ("linear_dr", "random", "does the linear CATE beat random treatment?"),
    ("arm1_predictive", "linear_dr", "predictive response vs causal score"),
    ("t_learner", "linear_dr", "unrestricted T-learner vs linear DR"),
]


def blp_test(gamma, score):
    """OLS gamma ~ 1 + (score - mean) with HC1 standard errors."""
    s = score - score.mean()
    X = np.column_stack([np.ones_like(s), s])
    beta, *_ = np.linalg.lstsq(X, gamma, rcond=None)
    resid = gamma - X @ beta
    n, p = X.shape
    bread = np.linalg.inv(X.T @ X)
    meat = (X * resid[:, None] ** 2).T @ X
    cov = bread @ meat @ bread * n / (n - p)
    se = np.sqrt(np.diag(cov))
    return dict(ate=beta[0], ate_se=se[0], slope=beta[1], slope_se=se[1], slope_t=beta[1] / se[1],
                score_sd=float(score.std()))


def gates(gamma, score, groups=5):
    q = pd.qcut(pd.Series(score).rank(method="first"), groups, labels=False)
    df = pd.DataFrame({"g": q, "gamma": gamma})
    out = df.groupby("g")["gamma"].agg(["mean", "std", "size"])
    out["se"] = out["std"] / np.sqrt(out["size"])
    return out.drop(columns="std").reset_index().rename(columns={"g": "quintile"})


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 1 X5 signal and power check")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="results/tables")
    ap.add_argument("--propensity", type=float, default=0.4998)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-gain", type=float, default=0.003,
                    help="smallest per-customer gain worth reporting; fix before running")
    args = ap.parse_args(argv)

    tr = pd.read_parquet(Path(args.processed) / "train.parquet")
    va = pd.read_parquet(Path(args.processed) / "validation.parquet")
    n_test = len(pd.read_parquet(Path(args.processed) / "test.parquet", columns=["client_id"]))
    tables, summary = run(tr, va, propensity=args.propensity, folds=args.folds, n_test=n_test, min_gain=args.min_gain)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"phase1_{name}.csv", index=False)
    (out / "phase1_summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 160)
    print(tables["heterogeneity"].round(4).to_string(index=False), "\n")
    print(tables["comparisons"].round(5).to_string(index=False), "\n")
    print(json.dumps(summary, indent=2))
    return summary


def run(tr: pd.DataFrame, va: pd.DataFrame, propensity: float, folds: int = 5,
        n_test: int = 60012, min_gain: float = 0.003):
    """All Phase 1 computations on in-memory train/validation frames.

    n_test: size of the untouched test split (used only to scale SEs).
    min_gain: smallest per-customer value difference worth reporting, fixed
    BEFORE looking at results (default 3 purchases per 1,000 customers, about
    10% of the X5 ATE). Gate A requires the test-set MDE to be below it.
    """
    feats = [c for c in tr.columns if c not in NON_FEATURES]
    Xtr, Xva = tr[feats].to_numpy(float), va[feats].to_numpy(float)

    nf = NuisanceFit(propensity=propensity, n_folds=folds)
    fit_tr = nf.fit_crossfit(Xtr, tr["target"].to_numpy(), tr["treatment_flg"].to_numpy())
    fit_va = nf.evaluate(Xva, va["target"].to_numpy(), va["treatment_flg"].to_numpy())
    g_tr, g_va = fit_tr["gamma"], fit_va["gamma"]

    ridge = RidgeCV(alphas=np.logspace(-2, 7, 37)).fit(Xtr, g_tr)
    flex = HistGradientBoostingRegressor(
        learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=1000, l2_regularization=1.0,
        max_iter=500, early_stopping=True, validation_fraction=0.2, n_iter_no_change=30, random_state=0,
    ).fit(Xtr, g_tr)
    scores = {
        "arm1_predictive": fit_va["mu1"],
        "t_learner": fit_va["mu1"] - fit_va["mu0"],
        "linear_dr": ridge.predict(Xva),
        "flexible_dr": flex.predict(Xva),
    }

    het = pd.DataFrame([dict(score=k, **blp_test(g_va, v)) for k, v in scores.items()])
    gt = pd.concat([gates(g_va, v).assign(score=k) for k, v in scores.items()])

    rows, comps = [], []
    for k in COST_K:
        cm = CostModel(k).fit(tr["cost_basket_raw"])
        c = cm(va["cost_basket_raw"])
        for q in BUDGETS:
            pols = {"random": np.full(len(c), min(1.0, q / c.mean())),
                    "cost_only": hard_allocation(np.ones(len(c)), c, q)}
            pols.update({name: hard_allocation(s, c, q) for name, s in scores.items()})
            for name, pi in pols.items():
                v, se = dr_value(pi, g_va)
                rows.append(dict(k=k, q=q, policy=name, value=v, se=se,
                                 per_1000=1000 * v, treated_frac=float(np.mean(pi)),
                                 spend=float(np.mean(c * pi))))
            for a, b, question in COMPARISONS:
                d, se = paired_difference(pols[a], pols[b], g_va)
                comps.append(dict(k=k, q=q, a=a, b=b, diff=d, se=se, z=d / se if se > 0 else np.nan,
                                  per_1000=1000 * d,
                                  mde80_test_per_1000=1000 * 2.8 * se * np.sqrt(len(va) / n_test),
                                  share_differing=float(np.mean(np.abs(pols[a] - pols[b]) > 1e-12)),
                                  question=question))
    pv, cp = pd.DataFrame(rows), pd.DataFrame(comps)

    head = cp[(cp.a == "flexible_dr") & (cp.b == "linear_dr")]
    lin_het = het.set_index("score").loc["linear_dr"]
    # Arm-vs-arm differences (e.g. PT-CATE vs CATE) change the policy for a share
    # of customers comparable to flexible-vs-linear, so its SE is the proxy.
    mde = float(head["mde80_test_per_1000"].median())
    detectable = bool(lin_het.slope_t > 2)
    powered = mde <= 1000 * min_gain
    summary = dict(
        n_train=len(tr), n_validation=len(va), n_features=len(feats), propensity=propensity,
        ate_train_dr=float(g_tr.mean()), ate_validation_dr=float(g_va.mean()),
        ridge_alpha=float(ridge.alpha_), ridge_alpha_at_grid_edge=bool(ridge.alpha_ in (1e-2, 1e7)),
        flex_iterations=int(flex.n_iter_),
        linear_blp_slope_t=float(lin_het.slope_t), heterogeneity_detectable=detectable,
        max_headroom_z=float(head.z.max()),
        test_mde80_per_1000_median=mde, min_gain_per_1000=1000 * min_gain, test_powered=powered,
        gate="A" if detectable and powered else "B",
        gate_rule=("A if linear-score BLP slope t > 2 AND median test-set MDE (80% power, 5% two-sided) "
                   "for an arm-level paired difference <= min_gain; otherwise B. Exploratory, unadjusted."),
    )
    tables = dict(heterogeneity=het, gates=gt, policy_values=pv, comparisons=cp)
    return tables, summary


if __name__ == "__main__":
    main()
