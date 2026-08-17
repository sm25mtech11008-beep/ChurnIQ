"""Leakage-free sklearn pipelines.

Why pipelines: every fitted step (imputer medians, scaler means, one-hot
vocabulary) is learned *inside* the training fold only. Computing these on the
full dataset before splitting would let statistics of the test rows bleed into
training — the classic subtle leakage that inflates CV scores.
"""

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Impute + encode. Scaling only matters for the linear model; trees are
    scale-invariant, so the LightGBM pipeline sets scale=False."""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), config.NUMERIC_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_COLS,
            ),
        ],
        verbose_feature_names_out=True,
    )


def build_logistic() -> Pipeline:
    """Baseline: logistic regression with class_weight='balanced'.

    A linear baseline sets the bar — if boosting can't beat it, the extra
    complexity isn't earning its keep on this dataset.
    """
    return Pipeline(
        [
            ("prep", build_preprocessor(scale=True)),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ]
    )


def build_lgbm(**params) -> Pipeline:
    """LightGBM pipeline.

    Deliberately *no* scale_pos_weight / class_weight: reweighting improves
    recall at 0.5 but distorts predicted probabilities, and this project picks
    its threshold from a business savings curve — which needs honest,
    calibrated probabilities more than it needs a balanced-looking 0.5 cut.
    """
    defaults = dict(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=100,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    defaults.update(params)
    return Pipeline(
        [
            ("prep", build_preprocessor(scale=False)),
            ("clf", LGBMClassifier(**defaults)),
        ]
    )
