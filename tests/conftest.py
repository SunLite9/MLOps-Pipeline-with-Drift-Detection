"""Shared test fixtures. Tests never depend on a running MLflow server or
the (gitignored, ~200MB) raw dataset — `mlflow_test_env` points MLflow at a
throwaway local file store and registers a tiny model trained on synthetic
data matching the real schema, so the suite is fully hermetic and runs the
same in CI as it does locally.
"""

import os

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

CATEGORY_VALUES = {
    "payment_type": ["AA", "AB", "AC", "AD", "AE"],
    "employment_status": ["CA", "CB", "CC", "CD", "CE", "CF", "CG"],
    "housing_status": ["BA", "BB", "BC", "BD", "BE", "BF", "BG"],
    "source": ["INTERNET", "TELEAPP"],
    "device_os": ["linux", "windows", "macintosh", "other", "x11"],
}


def _synthetic_raw_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """A synthetic dataset with the same columns/types as the real BAF Base
    dataset, so preprocessing and the model schema match production exactly
    — just with random values instead of the real (uncommitted) CSV."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "income": rng.uniform(0.1, 0.9, n),
            "name_email_similarity": rng.uniform(0, 1, n),
            "prev_address_months_count": rng.integers(-1, 300, n),
            "current_address_months_count": rng.integers(0, 400, n),
            "customer_age": rng.choice([20, 30, 40, 50, 60, 70, 80, 90], n),
            "days_since_request": rng.exponential(1, n),
            "intended_balcon_amount": rng.normal(0, 100, n),
            "payment_type": rng.choice(CATEGORY_VALUES["payment_type"], n),
            "zip_count_4w": rng.integers(1, 6000, n),
            "velocity_6h": rng.uniform(0, 20000, n),
            "velocity_24h": rng.uniform(0, 10000, n),
            "velocity_4w": rng.uniform(0, 8000, n),
            "bank_branch_count_8w": rng.integers(0, 500, n),
            "date_of_birth_distinct_emails_4w": rng.integers(0, 30, n),
            "employment_status": rng.choice(CATEGORY_VALUES["employment_status"], n),
            "credit_risk_score": rng.integers(-100, 400, n),
            "email_is_free": rng.integers(0, 2, n),
            "housing_status": rng.choice(CATEGORY_VALUES["housing_status"], n),
            "phone_home_valid": rng.integers(0, 2, n),
            "phone_mobile_valid": rng.integers(0, 2, n),
            "bank_months_count": rng.integers(-1, 32, n),
            "has_other_cards": rng.integers(0, 2, n),
            "proposed_credit_limit": rng.choice([200.0, 500.0, 1000.0, 1500.0, 2000.0], n),
            "foreign_request": rng.integers(0, 2, n),
            "source": rng.choice(CATEGORY_VALUES["source"], n),
            "session_length_in_minutes": rng.exponential(10, n),
            "device_os": rng.choice(CATEGORY_VALUES["device_os"], n),
            "keep_alive_session": rng.integers(0, 2, n),
            "device_distinct_emails_8w": rng.integers(0, 3, n),
            "device_fraud_count": np.zeros(n, dtype=int),
            "month": rng.integers(0, 8, n),
            "fraud_bool": rng.integers(0, 2, n),
        }
    )


@pytest.fixture(scope="session")
def mlflow_test_env(tmp_path_factory):
    """Points MLFLOW_TRACKING_URI at a throwaway local file store, trains a
    tiny model on synthetic data (with the real artifacts serving expects:
    feature_columns.json, training_distribution.json), and promotes it to
    Production — all without a running server or the real dataset."""
    import mlflow
    import mlflow.xgboost
    from mlflow.tracking import MlflowClient

    from src.drift.detector import compute_training_distribution
    from src.training.data import preprocess

    tracking_dir = tmp_path_factory.mktemp("mlflow")
    tracking_uri = f"sqlite:///{tracking_dir}/mlflow.db"
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

    predictions_db = tmp_path_factory.mktemp("data") / "predictions.db"
    os.environ["PREDICTION_LOG_DB"] = str(predictions_db)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fraud-detection-test")

    raw = _synthetic_raw_df(n=500, seed=0)
    encoded, y = preprocess(raw)

    model = xgb.XGBClassifier(n_estimators=10, max_depth=2, random_state=0)
    model.fit(encoded, y)

    with mlflow.start_run() as run:
        mlflow.xgboost.log_model(
            model, name="model", registered_model_name="fraud-model", input_example=encoded.iloc[:5]
        )
        mlflow.log_dict({"feature_columns": list(encoded.columns)}, "feature_columns.json")
        mlflow.log_dict(compute_training_distribution(encoded), "training_distribution.json")
        run_id = run.info.run_id

    client = MlflowClient()
    version = client.search_model_versions(f"run_id='{run_id}'")[0]
    client.transition_model_version_stage(name="fraud-model", version=version.version, stage="Production")

    yield {"tracking_uri": tracking_uri, "run_id": run_id, "raw_df": raw}
