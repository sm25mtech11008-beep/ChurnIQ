"""Phase D serving: FastAPI app exposing /predict, /explain, /health.

    uvicorn churniq.api:app --port 8000

The model bundle is loaded once (lazily) from CHURNIQ_MODEL or the default
models/churn_model.joblib. /predict returns the calibrated probability and the
business-threshold decision; /explain returns the top signed SHAP factors.
"""

import os
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config
from .batch_score import risk_tier
from .explain import explain_row

app = FastAPI(title="ChurnIQ API", version="1.0.0")


@lru_cache(maxsize=1)
def get_bundle():
    path = os.environ.get("CHURNIQ_MODEL", str(config.MODEL_PATH))
    if not os.path.exists(path):
        raise HTTPException(status_code=503, detail=f"model not found at {path} — run churniq.train first")
    return joblib.load(path)


class Customer(BaseModel):
    """Raw customer features. Numerics may be null — the pipeline's median
    imputer (fitted on training folds only) handles missing values."""

    age: float | None = None
    annual_income: float | None = None
    dependents: float | None = None
    tenure: float | None = None
    senior_citizen: float | None = None
    monthlycharges: float | None = None
    totalcharges: float | None = None
    num_services: float | None = None
    has_phone_service: float | None = None
    has_internet_service: float | None = None
    has_online_security: float | None = None
    has_online_backup: float | None = None
    has_device_protection: float | None = None
    has_tech_support: float | None = None
    has_streaming_tv: float | None = None
    has_streaming_movies: float | None = None
    customer_satisfaction: float | None = None
    num_complaints: float | None = None
    num_service_calls: float | None = None
    late_payments: float | None = None
    avg_monthly_gb: float | None = None
    days_since_last_interaction: float | None = None
    credit_score: float | None = None
    signup_date_days_ago: float | None = None
    gender: str = "Female"
    education: str = "college"
    marital_status: str = "single"
    contract: str = Field("one_year", description="month_to_month | one_year | two_year")
    payment_method: str = "credit_card"
    paperless_billing: str = "Yes"

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.model_dump()])[config.FEATURE_COLS]


@app.get("/health")
def health():
    bundle = get_bundle()
    meta = bundle["metadata"]
    return {
        "status": "ok",
        "model_trained_at": meta["trained_at"],
        "holdout_roc_auc": meta["holdout_metrics_calibrated"]["roc_auc"],
        "business_threshold": bundle["threshold"],
    }


@app.post("/predict")
def predict(customer: Customer):
    bundle = get_bundle()
    p = float(bundle["model"].predict_proba(customer.to_frame())[:, 1][0])
    return {
        "churn_probability": round(p, 4),
        "risk_tier": str(risk_tier(pd.Series([p]).to_numpy(), bundle["threshold"])[0]),
        "target_for_retention": bool(p >= bundle["threshold"]),
        "threshold": bundle["threshold"],
    }


@app.post("/explain")
def explain(customer: Customer, top_k: int = 5):
    bundle = get_bundle()
    row = customer.to_frame()
    p = float(bundle["model"].predict_proba(row)[:, 1][0])
    return {
        "churn_probability": round(p, 4),
        "top_factors": explain_row(bundle["raw"], row, top_k=top_k),
        "note": "contributions are log-odds SHAP values of the raw model; positive pushes toward churn",
    }
