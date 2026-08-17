# Resume Bullets — ChurnIQ

Pick 2–3 per application; every number below is reproducible from this repo.

**Full project (data science / MLE roles):**

- Built an end-to-end churn-prediction system on 1M telco customers: leakage-audited
  EDA, calibrated LightGBM pipeline (holdout ROC-AUC 0.685, PR-AUC 0.201 vs 0.099
  base rate), FastAPI serving with per-customer SHAP explanations, Streamlit
  dashboard, and CI-tested Python package.

- Replaced the default 0.5 classification threshold with a business
  expected-savings optimization (offer cost vs per-customer CLV), targeting 10.4%
  of the base to capture an estimated $500k in retention value per 200k customers.

**Rigor-focused bullets (analytics / interview talking points):**

- Ran a full-population leakage audit (1M rows) on four suspect features using
  single-feature AUC and missingness-vs-outcome tests; documented keep/drop
  verdicts and shipped a leakage-free sklearn pipeline (all preprocessing fitted
  inside CV folds).

- Benchmarked model families across six training sizes (2k → 800k rows) on a fixed
  200k holdout; showed the dataset's signal is largely linear (logistic CV AUC
  0.6851 ≈ tuned LightGBM 0.6845) and recorded a model-selection decision based on
  explainability (native TreeSHAP) rather than raw accuracy.

- Calibrated predicted probabilities (isotonic, Brier 0.0852 vs 0.0894 baseline)
  before dollar-denominated thresholding, so campaign ROI estimates are computed
  from honest probabilities rather than reweighted scores.

**Engineering bullets (MLE / platform roles):**

- Shipped a tested serving stack: FastAPI `/predict`, `/explain` (LightGBM
  `pred_contrib` TreeSHAP aggregated to raw features), `/health`; 13 pytest cases
  that train a tiny model on synthetic data so CI runs without the private dataset;
  Docker + compose; GitHub Actions.

- Built an LLM retention-analyst assistant (Claude tool-use API) with two typed
  data tools — per-customer score lookup and population aggregates — over the 1M
  batch-scored customer base.

**Interview-ready "why" answers** (rehearse these): why fit-inside-folds, why no
scale_pos_weight, why threshold ≠ 0.5, why PR-AUC vs base rate, why the leakage
verdicts were evidence-based, why TabPFN was excluded (CPU-only 1M-row serving vs
≤10k GPU regime).
