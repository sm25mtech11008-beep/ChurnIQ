"""Metrics and the business-threshold savings curve."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import config


def classification_metrics(y_true, y_proba) -> dict:
    """The three metrics that matter for imbalanced churn scoring:

    - ROC-AUC: ranking quality overall (baseline 0.5).
    - PR-AUC: ranking quality on the rare positive class — judge it against
      the churn base rate (~0.099 here), not against 1.0.
    - Brier: probability *calibration* — mean squared error of the predicted
      probability. Critical here because the retention threshold converts
      probabilities into dollars.
    """
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
    }


def savings_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    monthly_charges: np.ndarray,
    offer_cost: float = config.OFFER_COST,
    save_rate: float = config.SAVE_RATE,
    clv_months: int = config.CLV_MONTHS,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Expected campaign savings as a function of the targeting threshold.

    For each threshold t we target everyone with P(churn) >= t:
        cost    = offer_cost * (number targeted)          — paid for everyone
        benefit = save_rate * CLV_i summed over *true churners* targeted
                  (CLV_i = clv_months * customer's own monthly charge)
        savings = benefit - cost

    Maximizing this instead of accuracy is the whole point: a symmetric 0.5
    threshold is meaningless when a missed churner costs ~16x an offer.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.02, 0.90, 0.01), 2)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    clv = clv_months * np.asarray(monthly_charges, dtype="float64")

    rows = []
    for t in thresholds:
        targeted = y_proba >= t
        n_targeted = int(targeted.sum())
        tp = int((targeted & (y_true == 1)).sum())
        benefit = save_rate * clv[targeted & (y_true == 1)].sum()
        cost = offer_cost * n_targeted
        rows.append(
            {
                "threshold": float(t),
                "n_targeted": n_targeted,
                "true_churners_caught": tp,
                "expected_savings": float(benefit - cost),
            }
        )
    return pd.DataFrame(rows)


def best_threshold(curve: pd.DataFrame) -> dict:
    """Threshold with maximum expected savings."""
    row = curve.loc[curve["expected_savings"].idxmax()]
    return {
        "threshold": float(row["threshold"]),
        "expected_savings": float(row["expected_savings"]),
        "n_targeted": int(row["n_targeted"]),
        "true_churners_caught": int(row["true_churners_caught"]),
    }


def decile_lift(y_true, y_proba) -> pd.DataFrame:
    """Lift table by predicted-risk decile — the standard 'is the ranking
    useful' report for a campaign team."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_proba)})
    df["decile"] = pd.qcut(df["p"].rank(method="first"), 10, labels=False) + 1
    base = df["y"].mean()
    out = (
        df.groupby("decile", observed=True)
        .agg(n=("y", "size"), churn_rate=("y", "mean"), avg_score=("p", "mean"))
        .reset_index()
        .sort_values("decile", ascending=False)
    )
    out["lift"] = out["churn_rate"] / base
    return out
