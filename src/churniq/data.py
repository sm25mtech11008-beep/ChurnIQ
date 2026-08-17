"""Data loading and feature/target splitting."""

import pandas as pd

from . import config


def load_clean(path=None) -> pd.DataFrame:
    """Load the cleaned, typed dataset produced by the Phase A notebook."""
    return pd.read_parquet(path or config.CLEAN_PARQUET)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split into features X and target y, dropping ID/date/leakage columns.

    Selecting FEATURE_COLS explicitly (rather than dropping known-bad columns)
    means a new column appearing upstream cannot silently leak into the model.
    """
    X = df[config.FEATURE_COLS].copy()
    y = df[config.TARGET].astype("int8")
    return X, y
