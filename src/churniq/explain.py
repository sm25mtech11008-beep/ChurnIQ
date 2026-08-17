"""Per-customer explanations via LightGBM's native SHAP values.

LightGBM implements TreeSHAP internally (`pred_contrib=True`), so we get exact
SHAP values for the raw (uncalibrated) model without the shap package. The
one-hot contributions are summed back to their original column so the answer
reads as "contract: +0.42" instead of "cat__contract_month_to_month: +0.42".
Contributions are in log-odds space of the raw model; sign and ranking carry
over to the calibrated probability.
"""

import numpy as np
import pandas as pd


def _original_column(transformed_name: str, original_cols: list[str]) -> str:
    """Map 'num__age' / 'cat__contract_month_to_month' -> 'age' / 'contract'."""
    name = transformed_name.split("__", 1)[-1]
    for col in sorted(original_cols, key=len, reverse=True):
        if name == col or name.startswith(col + "_"):
            return col
    return name


def explain_row(raw_pipeline, row: pd.DataFrame, top_k: int = 5) -> list[dict]:
    """Top-k signed feature contributions for a single customer row."""
    prep = raw_pipeline.named_steps["prep"]
    booster = raw_pipeline.named_steps["clf"].booster_
    X_t = prep.transform(row)
    contrib = booster.predict(X_t, pred_contrib=True)[0]  # last entry = bias

    feature_names = list(prep.get_feature_names_out())
    original_cols = list(row.columns)
    agg: dict[str, float] = {}
    for fname, value in zip(feature_names, contrib[:-1]):
        col = _original_column(fname, original_cols)
        agg[col] = agg.get(col, 0.0) + float(value)

    top = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    return [
        {
            "feature": col,
            "value": _json_safe(row.iloc[0][col]),
            "contribution": round(c, 4),
            "pushes": "toward churn" if c > 0 else "away from churn",
        }
        for col, c in top
    ]


def _json_safe(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    return v if not pd.isna(v) else None
