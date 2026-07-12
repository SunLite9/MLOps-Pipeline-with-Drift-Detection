"""FastAPI serving app: loads the current Production-stage fraud model from
the MLflow registry and serves predictions.

The Production model is loaded once at startup and cached in memory. A
background thread polls the registry every RELOAD_CHECK_INTERVAL_SECONDS and
swaps in a newer Production version if one appears (e.g. after the training
DAG promotes a challenger) — no redeploy needed. The same reload logic is
exposed as POST /reload-model for an on-demand check.

Every prediction is logged (input features, prediction, timestamp, model
version) to a local SQLite table, which is what the drift detector reads.
"""

import json
import os
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.artifacts
import mlflow.xgboost
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

from src.training.data import CATEGORICAL_COLS

MODEL_NAME = "fraud-model"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
PREDICTION_LOG_DB = Path(
    os.environ.get(
        "PREDICTION_LOG_DB",
        Path(__file__).resolve().parents[2] / "data" / "predictions.db",
    )
)
RELOAD_CHECK_INTERVAL_SECONDS = int(os.environ.get("RELOAD_CHECK_INTERVAL_SECONDS", "30"))


class PredictRequest(BaseModel):
    income: float
    name_email_similarity: float
    prev_address_months_count: int
    current_address_months_count: int
    customer_age: int
    days_since_request: float
    intended_balcon_amount: float
    payment_type: str
    zip_count_4w: int
    velocity_6h: float
    velocity_24h: float
    velocity_4w: float
    bank_branch_count_8w: int
    date_of_birth_distinct_emails_4w: int
    employment_status: str
    credit_risk_score: int
    email_is_free: int
    housing_status: str
    phone_home_valid: int
    phone_mobile_valid: int
    bank_months_count: int
    has_other_cards: int
    proposed_credit_limit: float
    foreign_request: int
    source: str
    session_length_in_minutes: float
    device_os: str
    keep_alive_session: int
    device_distinct_emails_8w: int
    device_fraud_count: int
    month: int


class PredictResponse(BaseModel):
    fraud_probability: float
    fraud_prediction: int
    model_version: str


class ModelState:
    """Holds the currently-loaded Production model. Access is guarded by a
    lock since the background poller and request handlers both touch it."""

    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.version: Optional[str] = None
        self.feature_columns: Optional[list] = None

    def snapshot(self):
        with self.lock:
            return self.model, self.version, self.feature_columns

    def update(self, model, version, feature_columns):
        with self.lock:
            self.model = model
            self.version = version
            self.feature_columns = feature_columns


state = ModelState()


def _mlflow_client() -> MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def _fetch_production_version(client: MlflowClient):
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    return versions[0] if versions else None


def _fetch_feature_columns(client: MlflowClient, run_id: str) -> Optional[list]:
    try:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="feature_columns.json")
        with open(path) as f:
            return json.load(f)["feature_columns"]
    except Exception:
        return None


def reload_if_new_version() -> bool:
    """Checks the registry for the current Production version; if it differs
    from what's loaded, loads and swaps it in. Returns True if a reload
    happened."""
    client = _mlflow_client()
    version = _fetch_production_version(client)
    if version is None:
        return False

    _, current_version, _ = state.snapshot()
    if version.version == current_version:
        return False

    model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/{version.version}")
    feature_columns = _fetch_feature_columns(client, version.run_id)
    state.update(model, version.version, feature_columns)
    return True


def _init_log_db():
    PREDICTION_LOG_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(PREDICTION_LOG_DB)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_version TEXT NOT NULL,
            input_json TEXT NOT NULL,
            fraud_probability REAL NOT NULL,
            fraud_prediction INTEGER NOT NULL,
            latency_ms REAL
        )
        """
    )
    # Additive migration for DBs created before latency_ms existed.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
    if "latency_ms" not in existing_cols:
        conn.execute("ALTER TABLE predictions ADD COLUMN latency_ms REAL")
    conn.commit()
    conn.close()


def _log_prediction(model_version: str, input_dict: dict, proba: float, pred: int, latency_ms: float):
    conn = sqlite3.connect(PREDICTION_LOG_DB)
    conn.execute(
        "INSERT INTO predictions (timestamp, model_version, input_json, fraud_probability, fraud_prediction, latency_ms) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), model_version, json.dumps(input_dict), proba, pred, latency_ms),
    )
    conn.commit()
    conn.close()


def _preprocess(request: PredictRequest, feature_columns: Optional[list]) -> pd.DataFrame:
    row = pd.DataFrame([request.model_dump()])
    row = pd.get_dummies(row, columns=CATEGORICAL_COLS)
    if feature_columns:
        # Aligns to the exact columns the model was trained on: adds any
        # missing one-hot columns as 0, drops unexpected ones, fixes order.
        row = row.reindex(columns=feature_columns, fill_value=0)
    return row


def _background_poll_loop():
    while True:
        time.sleep(RELOAD_CHECK_INTERVAL_SECONDS)
        try:
            reload_if_new_version()
        except Exception:
            pass  # a transient registry/network hiccup shouldn't kill the poller


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_log_db()
    reload_if_new_version()
    poller = threading.Thread(target=_background_poll_loop, daemon=True)
    poller.start()
    yield


app = FastAPI(title="Fraud Detection Serving API", lifespan=lifespan)


@app.get("/health")
def health():
    _, version, _ = state.snapshot()
    return {
        "status": "ok" if version else "no_production_model",
        "model_version": version,
    }


@app.post("/reload-model")
def reload_model():
    reloaded = reload_if_new_version()
    _, version, _ = state.snapshot()
    if version is None:
        raise HTTPException(status_code=503, detail="No Production model available in the registry")
    return {"reloaded": reloaded, "model_version": version}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start = time.perf_counter()

    model, version, feature_columns = state.snapshot()
    if model is None:
        raise HTTPException(status_code=503, detail="No Production model available in the registry")

    X = _preprocess(request, feature_columns)
    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(proba >= 0.5)

    latency_ms = (time.perf_counter() - start) * 1000
    _log_prediction(version, request.model_dump(), proba, pred, latency_ms)

    return PredictResponse(fraud_probability=proba, fraud_prediction=pred, model_version=version)
