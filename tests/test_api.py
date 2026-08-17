"""API tests against a tiny model bundle trained on synthetic data."""

import pytest
from fastapi.testclient import TestClient

EXAMPLE = {
    "age": 40, "tenure": 3, "contract": "month_to_month",
    "monthlycharges": 90.0, "customer_satisfaction": 2, "num_complaints": 4,
    "num_service_calls": 6, "late_payments": 2,
}


@pytest.fixture()
def client(tiny_bundle_path, monkeypatch):
    monkeypatch.setenv("CHURNIQ_MODEL", str(tiny_bundle_path))
    from churniq import api

    api.get_bundle.cache_clear()
    with TestClient(api.app) as c:
        yield c
    api.get_bundle.cache_clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_probability_and_decision(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_tier"] in {"low", "medium", "high"}
    assert isinstance(body["target_for_retention"], bool)


def test_predict_handles_missing_numerics(client):
    r = client.post("/predict", json={"contract": "two_year"})  # everything else defaulted/null
    assert r.status_code == 200


def test_explain_returns_top_factors(client):
    r = client.post("/explain", json=EXAMPLE)
    assert r.status_code == 200
    factors = r.json()["top_factors"]
    assert len(factors) == 5
    assert {"feature", "value", "contribution", "pushes"} <= set(factors[0])
    # factors must be ranked by absolute contribution
    mags = [abs(f["contribution"]) for f in factors]
    assert mags == sorted(mags, reverse=True)


def test_explain_surfaces_planted_signal(client):
    """The fixture plants satisfaction/complaints as the true drivers — the
    explainer should surface at least one of them for an extreme customer."""
    r = client.post("/explain", json=EXAMPLE)
    top_features = {f["feature"] for f in r.json()["top_factors"]}
    assert top_features & {"customer_satisfaction", "num_complaints"}
