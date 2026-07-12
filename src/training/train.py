"""Train a fraud classifier and log the run to MLflow.

Usage:
    python -m src.training.train --n-estimators 200 --max-depth 4 --learning-rate 0.1

Each invocation is one MLflow run under the "fraud-detection" experiment,
logging hyperparameters, train/val metrics, and the model artifact (using
the `mlflow.xgboost` flavor).
"""

import argparse

import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from src.training.data import load_splits

EXPERIMENT_NAME = "fraud-detection"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"


def evaluate(model: xgb.XGBClassifier, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
    }


def train(n_estimators: int, max_depth: int, learning_rate: float, sample_size: int) -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    splits = load_splits(sample_size=sample_size)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]

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
        mlflow.log_param("sample_size", sample_size)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("val_rows", len(X_val))

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        train_metrics = evaluate(model, X_train, y_train)
        val_metrics = evaluate(model, X_val, y_val)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        mlflow.xgboost.log_model(
            model,
            name="model",
            registered_model_name="fraud-model",
            input_example=X_train.iloc[:5],
        )
        mlflow.log_dict({"feature_columns": list(X_train.columns)}, "feature_columns.json")

        print(f"run_id={run.info.run_id}")
        print(f"train: {train_metrics}")
        print(f"val:   {val_metrics}")
        return run.info.run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--sample-size", type=int, default=250_000)
    args = parser.parse_args()

    train(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
