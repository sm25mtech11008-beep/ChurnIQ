"""Shared fixtures: a synthetic dataframe with the production schema and a
tiny trained model bundle. Tests never touch the real 1M-row dataset, so the
suite runs anywhere — including CI, where data/ is not checked in."""

import numpy as np
import pandas as pd
import pytest

from churniq import config


@pytest.fixture(scope="session")
def synth_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame(
        {
            "customer_id": [f"CUST{i:010d}" for i in range(n)],
            "signup_date": pd.Timestamp("2023-01-01"),
            "age": rng.integers(18, 90, n).astype("int8"),
            "gender": rng.choice(["Male", "Female"], n),
            "annual_income": rng.normal(60_000, 15_000, n).astype("float32"),
            "education": rng.choice(["high_school", "college", "master"], n),
            "marital_status": rng.choice(["single", "married", "widowed"], n),
            "dependents": rng.integers(0, 4, n).astype("int8"),
            "tenure": rng.integers(0, 72, n).astype("int8"),
            "contract": rng.choice(["month_to_month", "one_year", "two_year"], n),
            "payment_method": rng.choice(["credit_card", "bank_transfer", "electronic_check"], n),
            "paperless_billing": rng.choice(["Yes", "No"], n),
            "senior_citizen": rng.integers(0, 2, n).astype("int8"),
            "monthlycharges": rng.uniform(20, 120, n).astype("float32"),
            "totalcharges": rng.uniform(20, 8000, n).astype("float32"),
            "num_services": rng.integers(0, 8, n).astype("int8"),
            **{
                c: rng.integers(0, 2, n).astype("int8")
                for c in [
                    "has_phone_service", "has_internet_service", "has_online_security",
                    "has_online_backup", "has_device_protection", "has_tech_support",
                    "has_streaming_tv", "has_streaming_movies",
                ]
            },
            "customer_satisfaction": rng.integers(1, 10, n).astype("float32"),
            "num_complaints": rng.integers(0, 5, n).astype("float32"),
            "num_service_calls": rng.integers(0, 8, n).astype("int8"),
            "late_payments": rng.integers(0, 4, n).astype("int8"),
            "avg_monthly_gb": rng.uniform(1, 200, n).astype("float32"),
            "days_since_last_interaction": rng.integers(0, 365, n).astype("int16"),
            "credit_score": rng.uniform(300, 850, n).astype("float32"),
            "signup_date_days_ago": rng.uniform(30, 1500, n).astype("float32"),
        }
    )
    # Planted signal: low satisfaction -> churn, so the tiny model learns something.
    logit = -2.2 + 0.35 * (5 - df["customer_satisfaction"]) + 0.4 * df["num_complaints"]
    df["churn"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype("int8")
    # NaNs in the columns that have them in production.
    for col in ["annual_income", "customer_satisfaction", "num_complaints", "avg_monthly_gb", "credit_score"]:
        df.loc[rng.choice(n, size=n // 20, replace=False), col] = np.nan
    return df


@pytest.fixture(scope="session")
def tiny_bundle_path(synth_df, tmp_path_factory):
    """Train a tiny model on the synthetic data and save a real bundle."""
    import joblib
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from sklearn.model_selection import train_test_split

    from churniq.data import split_xy
    from churniq.pipeline import build_lgbm

    X, y = split_xy(synth_df)
    X_fit, X_cal, y_fit, y_cal = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    raw = build_lgbm(n_estimators=25, num_leaves=15, min_child_samples=20)
    raw.fit(X_fit, y_fit)
    calibrated = CalibratedClassifierCV(FrozenEstimator(raw), method="isotonic")
    calibrated.fit(X_cal, y_cal)

    path = tmp_path_factory.mktemp("model") / "tiny_model.joblib"
    joblib.dump(
        {
            "model": calibrated,
            "raw": raw,
            "threshold": 0.2,
            "feature_cols": config.FEATURE_COLS,
            "metadata": {
                "trained_at": "test",
                "holdout_metrics_calibrated": {"roc_auc": 0.0, "pr_auc": 0.0, "brier": 0.0},
            },
        },
        path,
    )
    return path
