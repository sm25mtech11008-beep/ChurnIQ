# ChurnIQ — Phase A EDA Findings

*All numbers computed on the **full 1M rows** (validated against the stratified 10K sample — rankings identical). Snapshot reference date: 2026-01-11 (dataset max signup timestamp).*

## Headline numbers

| metric | value |
|---|---|
| Rows × columns | 1,000,000 × 32 |
| Churn rate | **9.92%** |
| `scale_pos_weight` for XGBoost | ≈ 9.1 |
| PR-AUC random baseline | 0.099 |
| In-memory size (typed, deep) | 82 MB |
| Clean parquet | 37 MB |

## Top-4 separating numeric features (standardized mean difference)

| feature | mean (stay) | mean (churn) | std. mean diff | single-feature AUC |
|---|---|---|---|---|
| `customer_satisfaction` | 6.23 | 5.57 | 0.283 | 0.576 |
| `num_complaints` | 0.68 | 0.90 | 0.268 | 0.564 |
| `num_service_calls` | 1.72 | 2.11 | 0.259 | 0.563 |
| `late_payments` | 0.39 | 0.49 | 0.160 | 0.535 |

Runners-up: `has_tech_support` (0.152), `has_online_security` (0.114), `num_services` (0.112). These seven form the spine of feature engineering and the sanity check for SHAP in Phase B ("did the model learn what EDA showed?").

## Highest-risk segments (churn lift vs 9.92% base)

| segment | churn rate | lift | n |
|---|---|---|---|
| `contract = month_to_month` | **26.5%** | **2.67** | 19,992 (2.0% of base) |
| `contract = one_year` | 12.7% | 1.28 | 550,468 |
| `contract = two_year` | 5.7% | 0.57 | 429,540 |

**Contract type is the only categorical that matters.** Every other categorical level — gender, education, marital status, payment method, paperless billing — sits at lift 0.98–1.01, i.e. pure noise. Note the month-to-month pool is small here (2% of customers, unlike real telco where it is often ~half), so the highest-risk segment is high-lift but low-volume; the one-year cohort (55% of base at lift 1.28) may matter more in absolute revenue terms.

## Leakage verdicts (audit run on full 1M)

**Verdict: no leakage — `LEAKAGE_COLS = []`.** All four suspects retained.

| suspect | verdict | evidence |
|---|---|---|
| `customer_satisfaction` | keep | AUC 0.576; drop 6.2→5.6 is a mild pre-churn signal, not the 2–3/9 crater a cancellation-call survey would show |
| `num_complaints` | keep | AUC 0.564; +0.23 mean shift consistent with a running count, not a contaminated one |
| `num_service_calls` | keep | AUC 0.563; same reasoning |
| `days_since_last_interaction` | keep (zero signal) | AUC 0.500; class distributions identical to the decimal — no mechanical "churned customers go silent" effect |

Supporting evidence: missingness is uninformative — churn rate among missing rows ≈ 9.9% (= base) for all five NaN columns (`annual_income`, `customer_satisfaction`, `num_complaints`, `avg_monthly_gb`, `credit_score`), so NaNs are not caused by churn.

**Assumption on record:** this is synthetic data with no data dictionary; verdicts rest on statistical signatures. With real production data, measurement timing would be verified against the logging pipeline before shipping any of these features.

## Surprises / open questions

1. **Demographics carry nothing.** `age`, `annual_income`, `credit_score`, `senior_citizen`, `dependents`, `signup_date_days_ago` all have single-feature AUC ≤ 0.501. Churn here is driven entirely by contract commitment + service experience.
2. **`days_since_last_interaction` is pure noise** despite being the most suspicious-sounding column — keep-or-drop is indifferent for accuracy; kept for now.
3. **This is a weak-signal dataset** (best single feature 0.576). Realistic Phase B expectation: combined model ROC-AUC ~0.60–0.65, PR-AUC judged against the 0.099 baseline — not tutorial-land 0.85+.

## Validated on full 1M?

Yes. Sections 5–7 re-run on all 1,000,000 rows; segment lift ranking and top-4 numeric ranking match the 10K sample exactly.
