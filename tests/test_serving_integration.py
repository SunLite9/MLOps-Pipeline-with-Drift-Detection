"""Integration test for the serving API. Fully hermetic: the `mlflow_test_env`
fixture (tests/conftest.py) points MLflow at a throwaway local file store and
registers a synthetic Production model, so this runs the same in CI as it
does locally — no live server or the real (uncommitted) dataset needed.
"""

import sqlite3

import pytest
from fastapi.testclient import TestClient

SAMPLE_REQUEST = {
    "income": 0.3,
    "name_email_similarity": 0.98,
    "prev_address_months_count": -1,
    "current_address_months_count": 25,
    "customer_age": 40,
    "days_since_request": 0.006,
    "intended_balcon_amount": 102.45,
    "payment_type": "AA",
    "zip_count_4w": 1059,
    "velocity_6h": 13096.0,
    "velocity_24h": 7850.9,
    "velocity_4w": 6742.0,
    "bank_branch_count_8w": 5,
    "date_of_birth_distinct_emails_4w": 5,
    "employment_status": "CB",
    "credit_risk_score": 163,
    "email_is_free": 1,
    "housing_status": "BC",
    "phone_home_valid": 0,
    "phone_mobile_valid": 1,
    "bank_months_count": 9,
    "has_other_cards": 0,
    "proposed_credit_limit": 1500.0,
    "foreign_request": 0,
    "source": "INTERNET",
    "session_length_in_minutes": 16.2,
    "device_os": "linux",
    "keep_alive_session": 1,
    "device_distinct_emails_8w": 1,
    "device_fraud_count": 0,
    "month": 0,
}


@pytest.fixture(scope="module")
def client(mlflow_test_env):
    from src.serving.api import app

    with TestClient(app) as c:
        yield c


def test_health_reports_a_loaded_model_version(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_version"] is not None


def test_predict_returns_well_formed_response(client):
    resp = client.post("/predict", json=SAMPLE_REQUEST)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["fraud_prediction"] in (0, 1)
    assert body["model_version"] is not None


def test_predict_rejects_incomplete_input(client):
    resp = client.post("/predict", json={"income": 0.5})
    assert resp.status_code == 422


def test_predictions_are_logged(client, tmp_path=None):
    import src.serving.api as api_module

    resp = client.post("/predict", json=SAMPLE_REQUEST)
    assert resp.status_code == 200

    conn = sqlite3.connect(api_module.PREDICTION_LOG_DB)
    row = conn.execute(
        "SELECT model_version, fraud_probability, fraud_prediction FROM predictions ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()

    body = resp.json()
    assert row is not None
    assert row[0] == body["model_version"]
    assert row[1] == pytest.approx(body["fraud_probability"])
    assert row[2] == body["fraud_prediction"]


def test_reload_model_endpoint_reports_current_version(client):
    resp = client.post("/reload-model")
    assert resp.status_code == 200
    assert resp.json()["model_version"] is not None
