"""Pre-treatment feature construction, split and preprocessing for X5 RetailHero.

Replaces the logic in ``notebooks/preprocessing.ipynb`` and fixes the issues
documented in ``docs/preprocessing_audit.md`` (30 Oct 2026):

1. ``issue_redeem_delay`` used ``first_redeem_date`` values recorded AFTER the
   SMS communication (11.2% of clients redeem between 2019-03-19 and 2019-11).
   Only redemption information available at the reference date is used now.
2. Purchase-interval features contained NaNs that were never imputed
   (linear models / torch cannot consume NaN).
3. ``gender`` was ordinal-encoded (imposes F < M < U); now one-hot.
4. No standardisation; now StandardScaler fitted on train only.
5. ``observation_period_months`` duplicated ``observation_period_days`` (r = 1).
6. Recency and card tenure at the reference date were missing; now added.
7. ``client_id``, the raw basket value (needed for the cost function) and the
   split assignment are now kept, so splits are reproducible and costs can be built.

Usage (from the repo root, inside the project .venv)::

    python -m thesis.data.preprocess            # with src/ on PYTHONPATH
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# purchases.csv is "purchase history prior to communication" (organiser page and
# sklift docs); its last transaction is 2019-03-18 23:40:03. Anything dated on or
# after this reference date is post-treatment and must not enter X.
REFERENCE_DATE = pd.Timestamp("2019-03-19")
SEED = 42

# Count / monetary aggregates: right-skewed, no missing values -> log1p.
LOG_VARS = [
    "loyalty_points_spent", "loyalty_points_earned",
    "spending_last_30d", "spending_last_60d", "spending_last_90d", "total_spending",
    "n_transactions", "transactions_last_30d", "transactions_last_60d",
    "transactions_last_90d", "n_distinct_products", "n_stores",
    "median_basket_value", "avg_basket_value",
    "median_items_per_basket", "avg_items_per_basket",
]
# Inter-purchase intervals: undefined for clients with <2 (<3 for std)
# transactions. Imputed with the train median; explicit flags carry the
# structure (one flag each instead of five identical imputer indicators).
INTERVAL_VARS = [
    "min_days_between_purchases", "median_days_between_purchases",
    "mean_days_between_purchases", "max_days_between_purchases",
]
INTERVAL_STD_VARS = ["std_days_between_purchases"]
PLAIN_VARS = [
    "observation_period_days", "prop_transactions_points_spent",
    "prop_transactions_points_earned", "recency_days", "card_tenure_days",
    "redeemed_by_ref", "has_interval", "has_interval_std",
]
# Undefined values imputed; age keeps a missing indicator (invalid ages are
# data errors); the delay's missingness is already carried by redeemed_by_ref.
AGE_VARS = ["age"]
DELAY_VARS = ["issue_redeem_delay"]
CATEGORICAL_VARS = ["gender"]

ID_COLS = ["client_id", "treatment_flg", "target"]
# Kept unchanged beside the features: raw basket value is the cost input
# (memo section 3.2); it must NOT be log-transformed inside the cost.
PASSTHROUGH_COLS = ["avg_basket_value"]


def add_pre_treatment_features(df: pd.DataFrame, ref: pd.Timestamp = REFERENCE_DATE) -> pd.DataFrame:
    """Return a copy with all date-derived features computed as of ``ref``."""
    out = df.copy()
    issue = pd.to_datetime(out["first_issue_date"])
    redeem = pd.to_datetime(out["first_redeem_date"])
    last = pd.to_datetime(out["last_purchase_date"])
    if (issue >= ref).any():
        raise ValueError("first_issue_date after reference date: post-treatment rows")
    if (last >= ref).any():
        raise ValueError("purchase after reference date: purchases are not pre-treatment")

    # Age: loyalty cards require 18+; >100 treated as data error (as before).
    out["age"] = out["age"].where(out["age"].between(18, 100))

    # Redemption as known at the reference date. A redemption recorded after
    # the SMS is unknown at decision time and may be caused by the treatment.
    redeemed = redeem.notna() & (redeem < ref)
    out["redeemed_by_ref"] = redeemed.astype(float)
    delay = (redeem - issue).dt.total_seconds() / 86400
    same_event = (delay < 0) & (issue.dt.normalize() == redeem.dt.normalize()) & (delay.abs() <= 1 / 24)
    delay = delay.mask(same_event, 0.0)
    delay = delay.where(redeemed & (delay >= 0))  # negative = data error -> NaN
    out["issue_redeem_delay"] = delay

    out["card_tenure_days"] = (ref - issue).dt.total_seconds() / 86400
    out["recency_days"] = (ref - last).dt.total_seconds() / 86400
    out["has_interval"] = (out["n_transactions"] >= 2).astype(float)
    out["has_interval_std"] = (out["n_transactions"] >= 3).astype(float)
    return out


def split_ids(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """50/20/30 split stratified on treatment; identical to the notebook's split."""
    train, temp = train_test_split(df, test_size=0.50, stratify=df["treatment_flg"], random_state=seed)
    validation, test = train_test_split(temp, test_size=0.60, stratify=temp["treatment_flg"], random_state=seed)
    split = pd.Series(index=df.index, dtype="object")
    split.loc[train.index], split.loc[validation.index], split.loc[test.index] = "train", "validation", "test"
    return split


def make_preprocessor() -> Pipeline:
    log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    median = dict(strategy="median")
    columns = ColumnTransformer(
        [
            ("log", Pipeline([("impute", SimpleImputer(**median)), ("log", log1p)]), LOG_VARS),
            ("interval", Pipeline([("impute", SimpleImputer(**median)), ("log", log1p)]),
             INTERVAL_VARS + INTERVAL_STD_VARS),
            ("plain", "passthrough", PLAIN_VARS),
            ("age", SimpleImputer(add_indicator=True, **median), AGE_VARS),
            ("delay", SimpleImputer(**median), DELAY_VARS),
            ("gender", OneHotEncoder(drop=["U"], handle_unknown="ignore", sparse_output=False), CATEGORICAL_VARS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("columns", columns), ("scale", StandardScaler())]).set_output(transform="pandas")


def build(modelling: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], Pipeline, pd.Series]:
    """Feature table -> processed splits. Preprocessor is fitted on train only."""
    if not modelling["client_id"].is_unique:
        raise ValueError("client_id must be unique")
    feats = add_pre_treatment_features(modelling)
    split = split_ids(feats)
    pre = make_preprocessor().fit(feats.loc[split == "train"])
    out = {}
    for name in ["train", "validation", "test"]:
        part = feats.loc[split == name]
        X = pre.transform(part)
        if X.isna().any().any():
            raise ValueError(f"NaN left in processed {name} features: {X.columns[X.isna().any()].tolist()}")
        meta = part[ID_COLS + PASSTHROUGH_COLS].rename(columns={"avg_basket_value": "cost_basket_raw"})
        out[name] = pd.concat([meta.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    return out, pre, split


def main() -> None:
    import joblib

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--interim", default="data/interim/modelling_dataset.parquet")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--models", default="results/models")
    args = ap.parse_args()
    modelling = pd.read_parquet(args.interim)
    splits, pre, split = build(modelling)
    out, models = Path(args.out), Path(args.models)
    out.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_parquet(out / f"{name}.parquet", index=False)
    pd.DataFrame({"client_id": modelling["client_id"], "split": split}).to_csv(out / "split_assignment.csv", index=False)
    joblib.dump(pre, models / "preprocessor.joblib")
    features = [c for c in splits["train"].columns if c not in ID_COLS + ["cost_basket_raw"]]
    summary = {
        "reference_date": str(REFERENCE_DATE), "seed": SEED,
        "rows": {k: len(v) for k, v in splits.items()},
        "treated_share": {k: round(float(v["treatment_flg"].mean()), 4) for k, v in splits.items()},
        "n_features": len(features), "features": features,
    }
    (out / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
