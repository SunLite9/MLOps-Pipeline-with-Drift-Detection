"""Discrete, Airflow-callable training pipeline steps.

Each function takes/returns only JSON-serializable values (paths, ids,
floats) so it can be wrapped directly in an Airflow PythonOperator and pass
state between tasks via XCom, rather than passing large in-memory objects
between tasks that may run in different worker containers.

Chain: ingest_data -> validate_data -> train_model -> evaluate_model -> register_if_better
"""

import os
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, roc_auc_score

from src.drift.detector import compute_training_distribution
from src.training.data import (
    MONTH_COL,
    SAMPLE_SIZE,
    TARGET_COL,
    TEST_MONTHS,
    TRAIN_MONTHS,
    VAL_MONTHS,
    downsample,
    load_raw_data,
    preprocess,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
EXPERIMENT_NAME = "fraud-detection"
MODEL_NAME = "fraud-model"
PRIMARY_METRIC = "val_pr_auc"

# validate_data() sanity bounds
MIN_TRAIN_ROWS = 150_000
MAX_TRAIN_ROWS = 350_000
REQUIRED_COLS = [TARGET_COL, MONTH_COL, "income", "customer_age", "credit_risk_score"]
MAX_SANE_FRAUD_RATE = 0.10


def _tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def ingest_data(sample_size: int = SAMPLE_SIZE, random_state: int = 42) -> dict:
    """Load raw BAF data, stratified-downsample, one-hot encode, temporal
    split, write each split to parquet. Returns {split_name: path}."""
    raw = load_raw_data()
    sampled = downsample(raw, n=sample_size, random_state=random_state)

    encoded, _ = preprocess(sampled)
    encoded[TARGET_COL] = sampled[TARGET_COL].values
    encoded[MONTH_COL] = sampled[MONTH_COL].values

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, months in [("train", TRAIN_MONTHS), ("val", VAL_MONTHS), ("test", TEST_MONTHS)]:
        subset = encoded[encoded[MONTH_COL].isin(months)].reset_index(drop=True)
        path = DATA_DIR / f"{name}.parquet"
        subset.to_parquet(path, index=False)
        paths[name] = str(path)
    return paths


def validate_data(paths: dict) -> dict:
    """Basic sanity checks on each ingested split. Raises (fails loudly) on
    any violation so a bad ingest never reaches training. Passes `paths`
    through unchanged on success so the DAG can keep chaining via XCom."""
    for name, path in paths.items():
        df = pd.read_parquet(path)

        missing_required = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing_required:
            raise ValueError(f"[{name}] missing required columns: {missing_required}")

        nulls = df[REQUIRED_COLS].isnull().sum()
        if nulls.any():
            raise ValueError(f"[{name}] unexpected nulls in required columns:\n{nulls[nulls > 0]}")

        if name == "train" and not (MIN_TRAIN_ROWS <= len(df) <= MAX_TRAIN_ROWS):
            raise ValueError(
                f"[train] row count {len(df)} outside expected range "
                f"[{MIN_TRAIN_ROWS}, {MAX_TRAIN_ROWS}]"
            )

        fraud_rate = df[TARGET_COL].mean()
        if not (0.0 < fraud_rate < MAX_SANE_FRAUD_RATE):
            raise ValueError(f"[{name}] fraud rate {fraud_rate:.4%} outside sane bounds (0%, {MAX_SANE_FRAUD_RATE:.0%})")

    return paths


def train_model(
    paths: dict,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.1,
) -> str:
    """Fit the challenger model on the train split, log params + train
    metrics + the model artifact to MLflow, and register it (unstaged).
    Returns the MLflow run_id."""
    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    train_df = pd.read_parquet(paths["train"])
    y_train = train_df.pop(TARGET_COL)
    X_train = train_df

    with mlflow.start_run() as run:
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
            "random_state": 42,
        }
        mlflow.log_params(params)
        mlflow.log_param("train_rows", len(X_train))

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        train_proba = model.predict_proba(X_train)[:, 1]
        mlflow.log_metrics(
            {
                "train_roc_auc": roc_auc_score(y_train, train_proba),
                "train_pr_auc": average_precision_score(y_train, train_proba),
            }
        )

        mlflow.xgboost.log_model(
            model,
            name="model",
            registered_model_name=MODEL_NAME,
            input_example=X_train.iloc[:5],
        )
        # The serving API needs the exact trained column set (post one-hot-encoding)
        # to align raw incoming requests to what the model expects.
        mlflow.log_dict({"feature_columns": list(X_train.columns)}, "feature_columns.json")
        # The drift detector compares live inference traffic against this.
        mlflow.log_dict(compute_training_distribution(X_train), "training_distribution.json")

        return run.info.run_id


def evaluate_model(run_id: str, paths: dict) -> float:
    """Load the challenger model back from the run, score it on the
    validation split, log the validation metrics to that same run, and
    return the primary promotion metric (val_pr_auc).

    PR-AUC (not ROC-AUC) is used as the promotion metric because fraud is
    ~1% prevalence — ROC-AUC is optimistic under heavy class imbalance,
    while PR-AUC reflects precision/recall tradeoffs that matter here.
    """
    mlflow.set_tracking_uri(_tracking_uri())

    model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")

    val_df = pd.read_parquet(paths["val"])
    y_val = val_df.pop(TARGET_COL)
    X_val = val_df

    proba = model.predict_proba(X_val)[:, 1]
    metrics = {
        "val_roc_auc": roc_auc_score(y_val, proba),
        "val_pr_auc": average_precision_score(y_val, proba),
    }

    client = MlflowClient()
    for k, v in metrics.items():
        client.log_metric(run_id, k, v)

    return metrics[PRIMARY_METRIC]
