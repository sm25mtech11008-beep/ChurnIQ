"""Business-threshold and metrics tests."""

import numpy as np

from churniq.evaluate import best_threshold, classification_metrics, decile_lift, savings_curve


def test_classification_metrics_keys():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9])
    m = classification_metrics(y, p)
    assert set(m) == {"roc_auc", "pr_auc", "brier"}
    assert m["roc_auc"] == 1.0


def test_savings_curve_prefers_predictive_scores():
    rng = np.random.default_rng(1)
    n = 5000
    y = (rng.uniform(size=n) < 0.1).astype(int)
    charges = np.full(n, 70.0)
    good_p = np.clip(0.08 + 0.5 * y + rng.normal(0, 0.05, n), 0, 1)
    curve = savings_curve(y, good_p, charges)
    best = best_threshold(curve)
    assert best["expected_savings"] > 0
    assert 0 < best["n_targeted"] < n


def test_savings_never_positive_for_useless_scores():
    """With random scores, targeting loses money at every threshold that
    targets anyone — offer costs are paid, churners aren't preferentially caught."""
    rng = np.random.default_rng(2)
    n = 20000
    y = (rng.uniform(size=n) < 0.05).astype(int)
    p = rng.uniform(size=n)
    # expected value per random target: 0.05 * 0.3 * 12*30 - 50 < 0
    curve = savings_curve(y, p, np.full(n, 30.0))
    assert curve.loc[curve["n_targeted"] > 100, "expected_savings"].max() < 0


def test_decile_lift_is_monotone_for_perfect_scores():
    y = np.array([0] * 900 + [1] * 100)
    p = np.linspace(0, 1, 1000)
    table = decile_lift(y, p)
    assert table.iloc[0]["lift"] == 10.0  # top decile holds all churners
