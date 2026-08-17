"""Central configuration: paths, column roles, and business assumptions.

Column roles were decided in Phase A (see reports/eda_findings.md):
- ID/DATE columns are dropped — they identify rows, they don't describe customers.
  `signup_date` is replaced by the engineered `signup_date_days_ago` (anchored to
  the dataset max date 2026-01-11, not "today", so results are reproducible).
- LEAKAGE_COLS is empty *by audit, not by assumption*: every suspect
  (customer_satisfaction, num_complaints, num_service_calls,
  days_since_last_interaction) was checked on the full 1M rows and showed
  pre-churn-signal statistical signatures, not post-outcome contamination.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_CSV = ROOT / "data" / "Raw" / "customer_churn_1M.csv"
CLEAN_PARQUET = ROOT / "data" / "Processed" / "clean_full.parquet"
SAMPLE_PARQUET = ROOT / "data" / "Processed" / "sample_10k.parquet"
SCORES_PARQUET = ROOT / "data" / "Processed" / "scores.parquet"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = "churn"

ID_COLS = ["customer_id"]
DATE_COLS = ["signup_date"]
LEAKAGE_COLS: list[str] = []  # audited on full 1M — no leakage found

CATEGORICAL_COLS = [
    "gender",
    "education",
    "marital_status",
    "contract",
    "payment_method",
    "paperless_billing",
]

NUMERIC_COLS = [
    "age",
    "annual_income",
    "dependents",
    "tenure",
    "senior_citizen",
    "monthlycharges",
    "totalcharges",
    "num_services",
    "has_phone_service",
    "has_internet_service",
    "has_online_security",
    "has_online_backup",
    "has_device_protection",
    "has_tech_support",
    "has_streaming_tv",
    "has_streaming_movies",
    "customer_satisfaction",
    "num_complaints",
    "num_service_calls",
    "late_payments",
    "avg_monthly_gb",
    "days_since_last_interaction",
    "credit_score",
    "signup_date_days_ago",
]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# --- Business assumptions for the retention-offer threshold -----------------
# The classification threshold is NOT 0.5: it is chosen to maximize expected
# savings of a retention campaign. Assumptions (documented so they can be
# challenged — in production these come from the retention team):
#   * A targeted customer receives an offer costing OFFER_COST (sent to
#     everyone we target, churner or not).
#   * If a *true churner* is targeted, the offer saves them with probability
#     SAVE_RATE; a saved customer is worth ~CLV_MONTHS of their monthly charge.
OFFER_COST = 50.0     # $ per retention offer sent
SAVE_RATE = 0.30      # P(offer prevents churn | customer would have churned)
CLV_MONTHS = 12       # months of monthly charge retained if saved

RANDOM_STATE = 42
