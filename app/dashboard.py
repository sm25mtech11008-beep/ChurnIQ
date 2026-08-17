"""Phase E: executive retention dashboard (Streamlit).

    streamlit run app/dashboard.py

Reads the batch scores (data/Processed/scores.parquet) and model metrics
(reports/metrics.json). Two views: an executive retention view (who is at
risk, what a campaign is worth) and a model-performance view (is the model
good enough to act on).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from churniq import config

st.set_page_config(page_title="ChurnIQ", page_icon="📉", layout="wide")


@st.cache_data
def load_scores() -> pd.DataFrame:
    return pd.read_parquet(config.SCORES_PARQUET)


@st.cache_data
def load_metrics() -> dict:
    return json.loads((config.REPORTS_DIR / "metrics.json").read_text())


if not config.SCORES_PARQUET.exists():
    st.error("No scores found — run `python -m churniq.batch_score` first.")
    st.stop()

scores = load_scores()
metrics = load_metrics()
threshold = metrics["business_threshold"]["threshold"]

st.title("ChurnIQ — Retention Intelligence")
tab_exec, tab_model = st.tabs(["Executive view", "Model performance"])

with tab_exec:
    n = len(scores)
    flagged = scores["target_for_retention"].sum()
    high = (scores["risk_tier"] == "high").sum()
    est_savings = metrics["business_threshold"]["expected_savings"] * (n / 200_000)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers scored", f"{n:,}")
    c2.metric("Flagged for retention", f"{flagged:,}", f"{flagged / n:.1%} of base")
    c3.metric("High-risk tier", f"{high:,}")
    c4.metric("Est. campaign savings (extrapolated)", f"${est_savings:,.0f}")

    st.caption(
        f"Threshold {threshold:.2f} chosen to maximize expected savings "
        f"(offer ${metrics['business_assumptions']['offer_cost']:.0f}, "
        f"save rate {metrics['business_assumptions']['save_rate']:.0%}, "
        f"CLV = {metrics['business_assumptions']['clv_months']} x monthly charge)."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Risk-tier distribution")
        st.bar_chart(scores["risk_tier"].value_counts())
    with right:
        st.subheader("Churn risk by contract type")
        by_contract = scores.groupby("contract", observed=True).agg(
            customers=("churn_probability", "size"),
            avg_risk=("churn_probability", "mean"),
            flagged=("target_for_retention", "sum"),
        )
        st.dataframe(by_contract.style.format({"avg_risk": "{:.3f}"}))

    st.subheader("Highest-risk customers")
    st.dataframe(
        scores.nlargest(100, "churn_probability")[
            ["customer_id", "churn_probability", "risk_tier", "contract", "monthlycharges"]
        ].reset_index(drop=True)
    )

with tab_model:
    cal = metrics["holdout_metrics_calibrated"]
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC (holdout)", f"{cal['roc_auc']:.4f}", "0.5 = random")
    c2.metric("PR-AUC (holdout)", f"{cal['pr_auc']:.4f}", f"base rate {metrics['churn_rate']:.3f}")
    c3.metric("Brier score", f"{cal['brier']:.4f}", "lower = better calibrated")

    st.caption(
        "Weak-signal dataset by design (best single-feature AUC 0.576 in EDA) — "
        "judge PR-AUC against the 0.099 base rate, not against 1.0."
    )

    for fig, caption in [
        ("calibration_curve.png", "Calibration: predicted probabilities vs observed churn rates"),
        ("savings_curve.png", "Expected-savings curve used to pick the business threshold"),
        ("feature_importance.png", "LightGBM feature importance — sanity check vs EDA top separators"),
    ]:
        path = config.FIGURES_DIR / fig
        if path.exists():
            st.image(str(path), caption=caption)

    lift_path = config.REPORTS_DIR / "decile_lift.csv"
    if lift_path.exists():
        st.subheader("Decile lift (holdout)")
        st.dataframe(pd.read_csv(lift_path).style.format({"churn_rate": "{:.3f}", "lift": "{:.2f}"}))
