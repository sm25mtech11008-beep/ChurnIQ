"""Pipeline and data-contract tests."""

import numpy as np

from churniq import config
from churniq.data import split_xy
from churniq.pipeline import build_lgbm, build_logistic, build_preprocessor


def test_split_xy_drops_non_features(synth_df):
    X, y = split_xy(synth_df)
    assert list(X.columns) == config.FEATURE_COLS
    for banned in config.ID_COLS + config.DATE_COLS + config.LEAKAGE_COLS + [config.TARGET]:
        assert banned not in X.columns
    assert set(np.unique(y)) <= {0, 1}


def test_preprocessor_handles_nans(synth_df):
    X, _ = split_xy(synth_df)
    assert X[config.NUMERIC_COLS].isna().any().any(), "fixture should contain NaNs"
    Xt = build_preprocessor().fit_transform(X)
    assert not np.isnan(Xt).any(), "imputer must remove all NaNs"


def test_pipelines_produce_valid_probabilities(synth_df):
    X, y = split_xy(synth_df)
    for model in [build_logistic(), build_lgbm(n_estimators=25)]:
        p = model.fit(X, y).predict_proba(X)[:, 1]
        assert p.shape == (len(X),)
        assert (p >= 0).all() and (p <= 1).all()


def test_model_beats_random_on_planted_signal(synth_df):
    from sklearn.metrics import roc_auc_score

    X, y = split_xy(synth_df)
    p = build_lgbm(n_estimators=50).fit(X, y).predict_proba(X)[:, 1]
    assert roc_auc_score(y, p) > 0.6
