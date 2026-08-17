# ChurnIQ Model Card

**Model:** LightGBM (isotonic-calibrated) inside a leakage-free sklearn pipeline.
**Trained:** 2026-08-17T19:06:54+00:00 on 1,000,000 rows (churn rate 9.92%).

## Selection

5-fold stratified CV, ROC-AUC:

| model | CV AUC |
|---|---|
| Logistic baseline | 0.6851 ± 0.0017 |
| LightGBM (default) | 0.6844 ± 0.0016 |

Optuna (20 TPE trials, 3-fold on 200k subsample) best CV AUC: 0.6845183732973438.

## Holdout performance (20%, untouched during training/tuning/calibration)

| metric | value | context |
|---|---|---|
| ROC-AUC | 0.6846 | 0.5 = random |
| PR-AUC | 0.2005 | base rate 0.099 = random |
| Brier | 0.0852 | lower is better; churn-rate-only predictor ≈ 0.0894 |

Phase A predicted a weak-signal dataset (best single-feature AUC 0.576) — results should be read
against that ceiling, not against tutorial datasets.

## Business threshold

Chosen to maximize expected retention-campaign savings on the holdout
(offer cost $50, save rate 30%,
CLV = 12 × monthly charge):

- **threshold = 0.17** → target 20,837 customers,
  catch 5,055 true churners,
  expected savings **$499,664** on the holdout slice.

## Sanity check vs EDA

Top gain features: `contract_two_year`, `customer_satisfaction`, `num_complaints`, `num_service_calls`, `contract_month_to_month`, `contract_one_year`, `late_payments`, `has_tech_support`.
Phase A's top separators (customer_satisfaction, num_complaints, num_service_calls,
late_payments, contract) should dominate this list — if they don't, investigate before trusting the model.

Top-decile lift: predicted-risk decile 10 churns at 24.52%
(2.47× base rate).

## Known limitations

- Synthetic data; no true temporal split (no event timestamps beyond signup date).
- Business assumptions (offer cost / save rate / CLV) are placeholders to demonstrate
  the method — sensitivity analysis belongs in front of any real deployment.
