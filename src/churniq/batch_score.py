"""Phase D batch scoring: score all customers and write scores + risk tiers.

    python -m churniq.batch_score

Writes data/Processed/scores.parquet with customer_id, churn probability,
risk tier, and above/below the business threshold — the table a campaign
team (or the dashboard / Postgres load) consumes.
"""

import time

import joblib
import numpy as np
import pandas as pd

from . import config
from .data import load_clean, split_xy


def risk_tier(p: np.ndarray, threshold: float) -> np.ndarray:
    """Three human tiers anchored on the business threshold, not arbitrary cuts."""
    return np.select(
        [p >= threshold, p >= threshold / 2], ["high", "medium"], default="low"
    )


def main():
    t0 = time.time()
    bundle = joblib.load(config.MODEL_PATH)
    df = load_clean()
    X, y = split_xy(df)

    print(f"Scoring {len(df):,} customers ...")
    proba = bundle["model"].predict_proba(X)[:, 1]
    threshold = bundle["threshold"]

    scores = pd.DataFrame(
        {
            "customer_id": df["customer_id"],
            "churn_probability": proba.astype("float32"),
            "risk_tier": risk_tier(proba, threshold),
            "target_for_retention": proba >= threshold,
            "monthlycharges": df["monthlycharges"],
            "contract": df["contract"],
            "churn_actual": y,  # available on this snapshot; absent in production
        }
    )
    scores.to_parquet(config.SCORES_PARQUET, index=False)

    n_target = int(scores["target_for_retention"].sum())
    print(f"  wrote {config.SCORES_PARQUET}")
    print(f"  threshold {threshold:.2f} -> {n_target:,} customers flagged "
          f"({n_target / len(df):.1%}) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
