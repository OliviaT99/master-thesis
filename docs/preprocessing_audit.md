# Preprocessing audit — 30 Oct 2026

Scope: `notebooks/feature_engineering.ipynb`, `notebooks/EDA.ipynb` and
`notebooks/preprocessing.ipynb` at commit `1b9181d` (local working tree is identical
except for `EDA.ipynb`). The fixes are implemented in `src/thesis/data/preprocess.py`
and tested in `tests/test_preprocess.py`.

Source for timing: the organiser page and the scikit-uplift docs both describe
`purchases` as "clients' purchase history prior to communication". The last
transaction is 2019-03-18 23:40:03, so the reference (decision) date is 2019-03-19.

## Findings, by severity

| # | Severity | Finding | Evidence | Fix |
|---|---|---|---|---|
| 1 | Critical | `issue_redeem_delay` uses `first_redeem_date` values recorded **after** the SMS, which makes it a post-treatment covariate | `clients.csv`: 44,953 of 400,162 clients (11.2%) redeem between 2019-03-19 and 2019-11-20, peaking in March–April 2019; another 35,469 (8.9%) never redeem. Redemption can be caused by the SMS and is tied to the outcome (a purchase). | `redeemed_by_ref` flag plus delay only when redemption happened before 2019-03-19 |
| 2 | Critical | The 5 inter-purchase interval features keep NaN (no imputer in `log_structural_pipeline`) | 3,826 / 9,298 NaN in the modelling table. Linear models and torch cannot consume NaN. | Median imputation (train), plus flags `has_interval` (≥2 transactions) and `has_interval_std` (≥3) |
| 3 | High | `gender` is ordinal-encoded (F=0, M=1, U=2) | This imposes a meaningless order on a linear score | One-hot encoding, baseline U |
| 4 | High | No standardisation | Needed for ridge/linear g and NN; Frauen's config uses `standardize: True` | `StandardScaler` fitted on train |
| 5 | High | Processed files drop `client_id`, the raw basket value and the split assignment | Costs cannot be built and splits cannot be audited | Kept: `client_id`, `cost_basket_raw` (raw `avg_basket_value`, not a feature), `split_assignment.csv` |
| 6 | Medium | Recency and card tenure at the decision date are missing (`last_purchase_date` / `first_issue_date` are dropped as datetimes) | Recency is the "R" of RFM and a standard uplift predictor | `recency_days`, `card_tenure_days` as of 2019-03-19 |
| 7 | Medium | `observation_period_months` = days / 30.44 (r = 1.000); several pairs have r > 0.95 | EDA correlation table | Months dropped. Other pairs kept for now (ridge copes); prune only if linear-g coefficients need to be interpreted |
| 8 | Medium | `data/raw/uplift_train.csv` is missing locally, but `feature_engineering.ipynb` reads it | Directory listing | Re-run `src/thesis/data/download_x5.py`; record the sklift version and file hashes |
| 9 | Low | The pipeline lives only in notebooks | — | Module plus tests; the notebook can import `thesis.data.preprocess` |

## Verified as correct (keep)

- Zero-value purchases: all 620 rows sit in 567 zero-only transactions, and no client loses all their purchases. Excluding them is sound.
- The transaction key is (`client_id`, `transaction_id`); 28 recycled IDs were correctly identified.
- Per-transaction values are constant within a transaction (asserted), so the basket aggregates are correct.
- The 30/60/90-day windows end at 2019-03-19, and every purchase precedes it.
- Age outside 18–100 is set to NaN, then imputed with an indicator.
- The split is stratified on treatment and is reproducible (the new module reproduces it exactly; see `test_split_matches_notebook_procedure`). Imputers are fitted on train only.
- RCT balance: all absolute SMDs < 0.02, and gender shares balance by arm. The treated share is 0.4998 in every split.

## Still to check on your machine (needs pyarrow, i.e. your .venv)

Is post-communication redemption itself affected by treatment? If yes, finding 1 also biased CATE estimates, not just their interpretation:

```python
import pandas as pd
m = pd.read_parquet("data/interim/modelling_dataset.parquet")
late = pd.to_datetime(m.first_redeem_date) >= "2019-03-19"
print(m.assign(late=late).groupby("treatment_flg")[["late"]].mean())
print(m.assign(late=late).groupby(["late", "treatment_flg"]).target.mean().unstack())
```

### Result (run 9 Nov 2026 on the full modelling table)

| | Control | Treated |
|---|---|---|
| Share redeeming on/after 2019-03-19 | 11.10% | 11.49% |
| Purchase rate, redeemed after the SMS | 57.47% | 62.96% |
| Purchase rate, everyone else | 60.68% | 63.74% |

The SMS raised post-campaign redemption by 0.39 pp (z ≈ 2.7, two-proportion test by hand from the
shares with about 100k customers per arm). So the old `issue_redeem_delay` was a post-treatment
variable affected by treatment. Among late redeemers the treated–control purchase difference is
5.5 pp vs 3.1 pp for everyone else, a gap a CATE model would have mistaken for heterogeneity.
Finding 1 therefore biased effect estimates, not only their interpretation. Phase 1 (9 Nov) already
uses the fixed features.
