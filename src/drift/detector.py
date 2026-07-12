"""Population Stability Index (PSI) drift detection.

PSI compares a live window of recent inference inputs (read from the
prediction log written by src/serving/api.py) against the distribution the
currently-`Production` model was trained on (an artifact logged alongside
the model by `train_model()`).

Only numeric raw features are checked. One-hot-encoded categorical dummy
columns are 0/1 indicators of a small fixed category set and don't lend
themselves to the same quantile-binning approach; a categorical
frequency/chi-square check would be the natural follow-up but is out of
scope here.

Threshold: PSI > 0.2 is treated as significant drift — the commonly-cited
industry rule of thumb (PSI < 0.1: no significant shift, 0.1-0.2: moderate
shift worth watching, > 0.2: significant shift requiring action). A single
feature crossing 0.2 is enough to declare drift: requiring multiple features
to drift simultaneously would mean missing a real, isolated shift on one
important feature (e.g. a sudden change in transaction velocity alone is a
meaningful fraud signal shift, even if every other feature looks normal).
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.artifacts
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.training.data import CATEGORICAL_COLS

MODEL_NAME = "fraud-model"
PSI_THRESHOLD = 0.2
N_BINS = 10
MIN_LIVE_SAMPLES = 30
DEFAULT_WINDOW_SIZE = 200

PREDICTION_LOG_DB = Path(
    os.environ.get(
        "PREDICTION_LOG_DB",
        Path(__file__).resolve().parents[2] / "data" / "predictions.db",
    )
)


def _tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def numeric_columns(columns) -> list:
    """Raw numeric feature names: everything except the one-hot dummy
    columns derived from CATEGORICAL_COLS."""
    return [c for c in columns if not any(c.startswith(f"{cat}_") for cat in CATEGORICAL_COLS)]


def compute_training_distribution(X_train: pd.DataFrame, n_bins: int = N_BINS) -> dict:
    """Per numeric feature: quantile bin edges from the training data, plus
    the reference ("expected") proportion of training rows in each bin.
    Logged as an MLflow artifact alongside the model."""
    distribution = {}
    for col in numeric_columns(X_train.columns):
        values = X_train[col].astype(float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            continue  # not enough distinct values to bin meaningfully
        counts, _ = np.histogram(values, bins=edges)
        proportions = (counts / counts.sum()).tolist()
        distribution[col] = {"bin_edges": edges.tolist(), "expected_proportions": proportions}
    return distribution


def _psi(expected_proportions, actual_proportions, epsilon: float = 1e-4) -> float:
    expected = np.asarray(expected_proportions) + epsilon
    actual = np.asarray(actual_proportions) + epsilon
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def compute_drift(live_df: pd.DataFrame, training_distribution: dict) -> dict:
    """Per-feature PSI for every feature present in both the live data and
    the stored training distribution. Live values outside the training
    range are counted into the nearest boundary bin, rather than dropped —
    values falling *outside* the training range entirely is itself a strong
    drift signal that should count, not be silently ignored."""
    psi_scores = {}
    for col, dist in training_distribution.items():
        if col not in live_df.columns:
            continue
        edges = np.asarray(dist["bin_edges"])
        values = pd.to_numeric(live_df[col], errors="coerce").dropna().astype(float)
        if values.empty:
            continue

        counts, _ = np.histogram(values, bins=edges)
        counts = counts.astype(float)
        counts[0] += (values < edges[0]).sum()
        counts[-1] += (values > edges[-1]).sum()

        total = counts.sum()
        if total == 0:
            continue

        actual_proportions = (counts / total).tolist()
        psi_scores[col] = _psi(dist["expected_proportions"], actual_proportions)
    return psi_scores


def is_drifted(psi_scores: dict, threshold: float = PSI_THRESHOLD):
    drifted_features = {feature: psi for feature, psi in psi_scores.items() if psi > threshold}
    return len(drifted_features) > 0, drifted_features


def _load_recent_predictions(window_size: int = DEFAULT_WINDOW_SIZE) -> pd.DataFrame:
    if not PREDICTION_LOG_DB.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(PREDICTION_LOG_DB)
    rows = conn.execute(
        "SELECT input_json FROM predictions ORDER BY id DESC LIMIT ?", (window_size,)
    ).fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([json.loads(r[0]) for r in rows])


def _load_production_training_distribution() -> tuple[Optional[dict], Optional[str]]:
    mlflow.set_tracking_uri(_tracking_uri())
    client = MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not versions:
        return None, None
    version = versions[0]
    try:
        path = mlflow.artifacts.download_artifacts(
            run_id=version.run_id, artifact_path="training_distribution.json"
        )
        with open(path) as f:
            distribution = json.load(f)
    except Exception:
        return None, version.version
    return distribution, version.version


def check_for_drift(window_size: int = DEFAULT_WINDOW_SIZE, threshold: float = PSI_THRESHOLD) -> dict:
    """The single entry point drift_check_dag calls: reads the recent
    prediction log, compares it against the currently-Production model's
    training distribution, and reports whether retraining should trigger."""
    live_df = _load_recent_predictions(window_size)
    training_distribution, production_version = _load_production_training_distribution()

    if training_distribution is None:
        return {
            "drifted": False,
            "reason": "no training distribution available for the current Production model",
            "n_samples": len(live_df),
            "production_version": production_version,
            "psi_scores": {},
            "drifted_features": {},
        }

    if len(live_df) < MIN_LIVE_SAMPLES:
        return {
            "drifted": False,
            "reason": f"insufficient live samples ({len(live_df)} < {MIN_LIVE_SAMPLES})",
            "n_samples": len(live_df),
            "production_version": production_version,
            "psi_scores": {},
            "drifted_features": {},
        }

    psi_scores = compute_drift(live_df, training_distribution)
    drifted, drifted_features = is_drifted(psi_scores, threshold)

    return {
        "drifted": drifted,
        "reason": f"{len(drifted_features)} feature(s) exceeded PSI threshold {threshold}" if drifted else "no feature exceeded the PSI threshold",
        "psi_scores": psi_scores,
        "drifted_features": drifted_features,
        "n_samples": len(live_df),
        "production_version": production_version,
        "threshold": threshold,
    }
