# Fraud Detection Pipeline

A production-style ML lifecycle around a fraud classifier: tracked training, a promotion-gated
model registry, containerized serving, drift-triggered retraining, and CI/CD. The model itself is
intentionally simple — the system around it is the point.

## Dataset

[Bank Account Fraud (BAF) Dataset Suite](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)
(NeurIPS 2022), "Base" variant: a synthetic bank account opening dataset with ~1M rows, 30 features,
and a binary target (`fraud_bool`), ~1.1% fraud prevalence.

The raw CSV is not checked into version control (see `.gitignore`); place it at:

```
Bank Account Fraud Dataset Suite (NeurIPS 2022)/Base.csv
```

For local iteration speed, training uses a **250,000-row stratified downsample** (by fraud label,
preserving the ~1.1% prevalence) rather than the full 1M rows. Splits are **temporal**, using the
dataset's `month` column (0–7): months 0–5 for training, month 6 for validation, month 7 held out
as test — this mirrors how the data is meant to be used (train on the past, evaluate on data that
arrives later) and sets up a natural distribution shift to exploit later for drift detection.

## Project layout

```
src/training/   data loading, preprocessing, training, promotion logic
src/serving/    FastAPI inference service
src/drift/      drift detection + retraining trigger
dags/           Airflow DAGs
docker/         Dockerfiles / compose services
dashboard/      monitoring dashboard
tests/          test suite
```

## Setup

```bash
python -m venv .venv
./.venv/Scripts/activate       # Windows
pip install -r requirements.txt
```

## Running the MLflow tracking server

```bash
mkdir -p mlflow_store/artifacts
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow_store/mlflow.db \
  --default-artifact-root ./mlflow_store/artifacts
```

The UI is then available at http://127.0.0.1:5000.

> Windows note: MLflow's run-summary log line contains an emoji that the default `cp1252` console
> encoding can't print. If you see a `UnicodeEncodeError` after a run otherwise completes
> successfully, set `PYTHONIOENCODING=utf-8` before running training.

## Training

```bash
python -m src.training.train --n-estimators 200 --max-depth 4 --learning-rate 0.1
```

Each run trains an `XGBClassifier` (class-imbalance corrected via `scale_pos_weight`), logs
hyperparameters and train/validation metrics (ROC-AUC, PR-AUC) to MLflow, and registers the
resulting model under `fraud-model` in the MLflow Model Registry.

### Verified: two tracked runs, compared

| run | n_estimators | max_depth | learning_rate | val_roc_auc | val_pr_auc |
|---|---|---|---|---|---|
| `f053196b377e43fead27ed05a1059f33` | 200 | 4 | 0.1 | 0.8732 | 0.1467 |
| `5e98037a1a104e72bcc24841f20d542c` | 100 | 3 | 0.2 | 0.8758 | 0.1551 |

Both runs are visible in the MLflow UI under the `fraud-detection` experiment with their logged
params/metrics, and both produced a registered model version (`fraud-model` v1 and v2,
initially unstaged) confirming tracking, comparison, and registration all work end to end.
