"""Tests for thesis.data.preprocess on a synthetic frame with X5's schema."""
import numpy as np
import pandas as pd
import pytest

from thesis.data.preprocess import REFERENCE_DATE, add_pre_treatment_features, build, split_ids, LOG_VARS


def synthetic(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    issue = pd.Timestamp("2017-04-04") + pd.to_timedelta(rng.uniform(0, 700, n), unit="D")
    redeem = issue + pd.to_timedelta(rng.uniform(0, 800, n), unit="D")  # some after REFERENCE_DATE
    redeem = pd.Series(redeem).where(rng.uniform(size=n) > 0.09)          # ~9% never redeemed
    n_tx = rng.integers(1, 60, n)
    df = pd.DataFrame({
        "client_id": [f"c{i:06d}" for i in range(n)],
        "treatment_flg": rng.integers(0, 2, n), "target": rng.integers(0, 2, n),
        "first_issue_date": issue, "first_redeem_date": redeem,
        "last_purchase_date": pd.Timestamp("2019-03-18") - pd.to_timedelta(rng.uniform(0, 100, n), unit="D"),
        "age": rng.integers(5, 120, n), "gender": rng.choice(["F", "M", "U"], n),
        "n_transactions": n_tx, "observation_period_days": rng.uniform(0, 117, n),
        "observation_period_months": 0.0,
        "prop_transactions_points_spent": rng.uniform(size=n), "prop_transactions_points_earned": rng.uniform(size=n),
    })
    for v in LOG_VARS:
        if v != "n_transactions":
            df[v] = rng.lognormal(3, 1, n)
    for v in ["min", "median", "mean", "max", "std"]:
        df[f"{v}_days_between_purchases"] = np.where(n_tx >= (3 if v == "std" else 2), rng.uniform(0, 30, n), np.nan)
    # a same-hour negative delay and a large negative delay (data error)
    df.loc[0, "first_redeem_date"] = df.loc[0, "first_issue_date"] - pd.Timedelta(minutes=5)
    df.loc[1, "first_redeem_date"] = df.loc[1, "first_issue_date"] - pd.Timedelta(days=42)
    df.loc[2, "first_redeem_date"] = pd.Timestamp("2019-06-01")
    df.loc[2, "first_issue_date"] = pd.Timestamp("2018-06-01")
    return df


def test_post_treatment_redemption_is_masked():
    f = add_pre_treatment_features(synthetic())
    late = pd.to_datetime(synthetic()["first_redeem_date"]) >= REFERENCE_DATE
    assert late.any()
    assert (f.loc[late, "redeemed_by_ref"] == 0).all()
    assert f.loc[late, "issue_redeem_delay"].isna().all()
    assert f.loc[2, "redeemed_by_ref"] == 0
    assert f.loc[0, "issue_redeem_delay"] == 0.0           # same-hour -> simultaneous
    assert np.isnan(f.loc[1, "issue_redeem_delay"])         # -42 days -> data error
    assert f["age"].dropna().between(18, 100).all()


def test_rejects_post_reference_purchases():
    df = synthetic()
    df.loc[5, "last_purchase_date"] = REFERENCE_DATE
    with pytest.raises(ValueError):
        add_pre_treatment_features(df)


def test_build_outputs_are_clean_and_train_standardised():
    splits, pre, split = build(synthetic())
    assert {k: len(v) for k, v in splits.items()} == {"train": 1000, "validation": 400, "test": 600}
    train = splits["train"]
    feats = train.drop(columns=["client_id", "treatment_flg", "target", "cost_basket_raw"])
    assert not feats.isna().any().any()
    assert np.allclose(feats.mean(), 0, atol=1e-8) and np.allclose(feats.std(ddof=0), 1, atol=1e-8)
    assert "observation_period_months" not in feats and "gender_F" in feats and "gender_U" not in feats
    assert (train["cost_basket_raw"] > 0).all()
    ids = pd.concat([s["client_id"] for s in splits.values()])
    assert ids.is_unique and len(ids) == 2000


def test_split_matches_notebook_procedure():
    from sklearn.model_selection import train_test_split
    df = synthetic()
    tr, tmp = train_test_split(df, test_size=0.5, stratify=df["treatment_flg"], random_state=42)
    va, te = train_test_split(tmp, test_size=0.6, stratify=tmp["treatment_flg"], random_state=42)
    s = split_ids(df)
    assert set(s[s == "train"].index) == set(tr.index) and set(s[s == "test"].index) == set(te.index)
