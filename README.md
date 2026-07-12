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
mlflow server --host 127.0.0.1 --port 5000 --workers 1 --serve-artifacts \
  --artifacts-destination ./mlflow_store/artifacts \
  --backend-store-uri sqlite:///mlflow_store/mlflow.db \
  --allowed-hosts "localhost:5000,127.0.0.1:5000,host.docker.internal:5000"
```

The UI is then available at http://127.0.0.1:5000.

Notes on the flags:
- `--workers 1`: MLflow's multi-process uvicorn workers hit a socket-binding bug on Windows
  (`OSError: [WinError 10022]`) when sharing a single listening socket across processes; a single
  worker avoids it.
- `--serve-artifacts` + `--artifacts-destination` (instead of `--default-artifact-root`): proxies
  artifact uploads/downloads through the tracking server's REST API rather than requiring direct
  filesystem access to the artifact store. Without this, any client that isn't the machine running
  the server (e.g. the Airflow containers below) can't write model artifacts, since a local-path
  artifact root is accessed directly by the client, not through the server.
- `--allowed-hosts`: MLflow validates the `Host` header on incoming requests to prevent DNS
  rebinding attacks; the Airflow containers reach the server via `host.docker.internal`, which
  needs to be explicitly allow-listed (with its port) or requests are rejected with 403.

> Windows note: MLflow's run-summary log line contains an emoji that the default `cp1252` console
> encoding can't print. If you see a `UnicodeEncodeError` after a run otherwise completes
> successfully, set `PYTHONIOENCODING=utf-8` before running training.

## Training (manual, single run)

```bash
python -m src.training.train --n-estimators 200 --max-depth 4 --learning-rate 0.1
```

Each run trains an `XGBClassifier` (class-imbalance corrected via `scale_pos_weight`), logs
hyperparameters and train/validation metrics (ROC-AUC, PR-AUC) to MLflow, and registers the
resulting model under `fraud-model` in the MLflow Model Registry, unstaged.

### Verified: two tracked runs, compared

| run | n_estimators | max_depth | learning_rate | val_roc_auc | val_pr_auc |
|---|---|---|---|---|---|
| `f73d0fa2115e4de6bec042870a7fc7d5` | 250 | 4 | 0.08 | 0.8694 | 0.1402 |
| `7e17a64183074346b201e1636d670a83` | 120 | 3 | 0.15 | 0.8757 | 0.1497 |

Both runs are visible in the MLflow UI under the `fraud-detection` experiment with their logged
params/metrics, and both produced a registered model version, confirming tracking, comparison, and
registration all work end to end.

## Orchestrated training: Airflow + the promotion gate

`src/training/pipeline.py` breaks training into discrete steps — `ingest_data()`,
`validate_data()`, `train_model()`, `evaluate_model()` — chained by `dags/training_dag.py` into a
DAG: `ingest -> validate -> train -> evaluate -> register_if_better`. Only small values (file
paths, run ids, metric floats) pass between tasks via XCom; the actual data splits are written to
`data/processed/*.parquet` on a filesystem shared across tasks.

**Promotion rule** (`src/training/promotion.py`): a challenger is promoted to the registry's
`Production` stage only if its validation PR-AUC beats the current `Production` model's PR-AUC by
at least **2% relative improvement** (a small positive margin, not "any improvement," to avoid
promotion churn from run-to-run noise). PR-AUC (not ROC-AUC) is the gating metric because fraud is
~1% prevalence, where ROC-AUC is overly optimistic under heavy class imbalance. If no `Production`
model exists yet, the first challenger is promoted unconditionally (bootstrap case).

### Running Airflow locally

```bash
cd docker
docker compose build
docker compose up -d postgres
docker compose up airflow-init          # one-time: migrates the metadata DB, creates the admin user
docker compose up -d airflow-webserver airflow-scheduler
```

The UI is then available at http://localhost:8080 (user: `admin`, password: `admin`). The stack
uses `LocalExecutor` (Postgres + webserver + scheduler, no separate Celery workers) — simpler than
Airflow's default Celery quick-start and sufficient for a single-machine setup, and it means every
task in a DAG run shares the scheduler container's filesystem, so no extra volume-sharing setup is
needed between tasks. The Airflow image is built from `docker/airflow/Dockerfile` (extends
`apache/airflow:2.9.3-python3.11`, adds this repo's `requirements.txt`), and the containers reach
the MLflow server running on the host via `host.docker.internal`. `docker/docker-compose.yaml` is
the single compose file for the whole local stack (Postgres, Airflow, and the serving API below all
share one Docker network).

Trigger the DAG manually:

```bash
docker compose exec airflow-scheduler airflow dags trigger training_dag

# or, to override hyperparameters (used below to force a weak challenger):
docker compose exec airflow-scheduler airflow dags trigger training_dag \
  --conf '{"n_estimators": 1, "max_depth": 1, "learning_rate": 0.01}'
```

### Verified: promotion gate correctly accepts and rejects, via the actual DAG

Three DAG runs, triggered through Airflow (not called directly), against a fresh registry:

| run (`training_dag` trigger) | n_estimators | max_depth | learning_rate | val_pr_auc | registry outcome |
|---|---|---|---|---|---|
| 1 (defaults) | 200 | 4 | 0.1 | 0.1430 | **Promoted** to Production (v1) — bootstrap, no prior champion |
| 2 (`--conf` forced weak) | 1 | 1 | 0.01 | 0.0307 | **Rejected** (v2 stays unstaged) — below champion's 0.1430, gate correctly holds v1 as Production |
| 3 (`--conf` stronger) | 1000 | 3 | 0.02 | 0.1509 | **Promoted** to Production (v3) — beats v1's 0.1430 by >2% (threshold 0.1459); v1 archived |

All five DAG tasks (`ingest`, `validate`, `train`, `evaluate`, `register`) reported `success` for
every run, including the runs where the gate rejected the challenger — a rejection is a correct,
successful outcome of the `register` task, not a DAG failure. Final registry state after all three
runs: v1 `Archived`, v2 unstaged, v3 `Production`.

> The registry was later reset once (see the serving section below) after adding a
> `feature_columns.json` artifact needed for serving — the table above reflects the original
> verification run of the promotion gate; the mechanism is unchanged.

## Serving: FastAPI reading from the model registry, Dockerized

`src/serving/api.py` loads the current `Production`-stage model from the registry once at startup,
serves predictions over `POST /predict`, and reports the loaded model version via `GET /health`.
Every prediction (input features, prediction, timestamp, model version) is logged to a local
SQLite table (`data/predictions.db`) — this log is what the drift detector reads in the next phase.

A background thread re-checks the registry every 30s (`RELOAD_CHECK_INTERVAL_SECONDS`) and swaps in
a newer `Production` version if the training DAG has promoted one, with no redeploy — `POST
/reload-model` triggers the same check on demand.

Since the request body is raw (pre-one-hot-encoding) feature values, `train_model()` also logs a
`feature_columns.json` artifact alongside the model recording the exact post-encoding column set;
serving downloads it once at load time and reindexes each incoming row to match (missing dummy
columns filled with 0), so the request schema stays a natural, human-editable shape instead of
requiring pre-encoded input from the client.

### Running the full stack

```bash
cd docker
docker compose up -d postgres
docker compose up airflow-init
docker compose up -d airflow-webserver airflow-scheduler serving
```

`serving` builds from `docker/serving/Dockerfile` (plain `python:3.11-slim` + `requirements.txt`)
and joins the same Docker network as Airflow, reaching MLflow via `host.docker.internal` the same
way the Airflow containers do. `data/` is mounted read-write so `predictions.db` persists on the
host and is visible to tooling outside the container (e.g. the drift detector).

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "income": 0.3, "name_email_similarity": 0.98, "prev_address_months_count": -1,
  "current_address_months_count": 25, "customer_age": 40, "days_since_request": 0.006,
  "intended_balcon_amount": 102.45, "payment_type": "AA", "zip_count_4w": 1059,
  "velocity_6h": 13096.0, "velocity_24h": 7850.9, "velocity_4w": 6742.0,
  "bank_branch_count_8w": 5, "date_of_birth_distinct_emails_4w": 5, "employment_status": "CB",
  "credit_risk_score": 163, "email_is_free": 1, "housing_status": "BC", "phone_home_valid": 0,
  "phone_mobile_valid": 1, "bank_months_count": 9, "has_other_cards": 0,
  "proposed_credit_limit": 1500.0, "foreign_request": 0, "source": "INTERNET",
  "session_length_in_minutes": 16.2, "device_os": "linux", "keep_alive_session": 1,
  "device_distinct_emails_8w": 1, "device_fraud_count": 0, "month": 0
}'
```

### Verified

**Integration test** (`tests/test_serving_integration.py`, run via `pytest`): 5/5 passed —
`/health` reports a loaded model version, `/predict` returns a well-formed response, malformed
requests are rejected with `422`, predictions are correctly written to the SQLite log, and
`/reload-model` reports the current version.

**Promotion takes effect, live, no redeploy** — with the `serving` container already running:

| step | model_version reported by `/health` |
|---|---|
| Container running, serving `Production` v2 | `2` |
| A new challenger trained and promoted to `Production` (v9, val_pr_auc 0.1573 vs. champion's 0.1509 — beats the 2% margin) via `register_if_better` | — |
| `/health` re-checked (no restart, no redeploy) | `9` |

The background poller picked up the new `Production` version automatically within its 30s check
interval every time this was tested, without ever needing the manual `/reload-model` call.

## The closed loop: drift detection with automatic retraining

This is the centerpiece of the project: live inference traffic is continuously compared against
the training distribution, and when it drifts far enough, the system retrains and redeploys itself
with no human in the loop.

```
        live /predict traffic
                │
                ▼
   [serving API] ──logs──▶ data/predictions.db
                                    │
                                    ▼
                     [drift_check_dag] (hourly)
                     reads recent predictions,
                     computes PSI vs. the Production
                     model's training distribution
                                    │
                          drift PSI > 0.2 on any feature?
                             │                  │
                            no                 yes
                             │                  │
                             ▼                  ▼
                          (stop)     TriggerDagRunOperator
                                             │
                                             ▼
                                     [training_dag]
                                 ingest → validate → train
                                 → evaluate → register_if_better
                                             │
                                   beats champion by ≥2%?
                                             │
                                            yes ──▶ promote to Production,
                                                     archive old champion
                                             │
                                             ▼
                          [serving API] background poller (≤30s)
                             picks up new Production version
                             automatically — no redeploy
```

### Drift detection method (`src/drift/detector.py`)

- **Metric**: Population Stability Index (PSI), computed per numeric raw feature (the one-hot
  categorical dummy columns are excluded — PSI's quantile-binning approach doesn't suit a small
  fixed category set; a categorical frequency check would be the natural follow-up).
- **Reference distribution**: at training time, `train_model()` computes 10 quantile bins per
  numeric feature from the training split and logs the bin edges + expected proportions as a
  `training_distribution.json` MLflow artifact alongside the model.
- **Live window**: the most recent 200 predictions from `data/predictions.db` (falls back to "not
  enough data" below 30 samples, rather than computing a noisy PSI on a handful of points).
- **Comparison**: live values are bucketed into the *same* bin edges as training; live values
  outside the training range entirely are counted into the nearest boundary bin rather than
  dropped, since falling outside the training range is itself a meaningful drift signal.
- **Threshold**: PSI > 0.2 on **any single feature** triggers drift. This is the standard
  industry rule of thumb (< 0.1 no shift, 0.1–0.2 moderate, > 0.2 significant), and any-feature
  (rather than requiring multiple features together) is deliberate — an isolated shift in one
  important feature (e.g. transaction velocity suddenly spiking) is a real, actionable signal on
  its own.

### `drift_check_dag.py`

Runs hourly (also manually triggerable): `check_drift` computes PSI via the detector,
`drift_detected` is an `@task.short_circuit` gate that skips the rest of the DAG when there's no
drift, and `TriggerDagRunOperator` fires `training_dag` when there is — Airflow's native
DAG-triggers-DAG mechanism.

### Simulating drift for testing (`src/drift/simulate_drift.py`)

```bash
python -m src.drift.simulate_drift --n 150                  # normal traffic (matches training distribution)
python -m src.drift.simulate_drift --n 150 --drifted         # traffic shifted away from it
```

Samples real rows from the raw dataset and posts them to `/predict`. In `--drifted` mode, five
behavioral features (`velocity_6h`, `velocity_24h`, `velocity_4w`, `session_length_in_minutes`,
`customer_age`) are multiplicatively shifted before sending, simulating a real change in traffic
patterns (e.g. a new fraud pattern or a legitimate change in user behavior).

### Verified: the full closed loop, end to end, through the real DAGs

**Step 1 — no drift under normal traffic.** Sent 150 normal requests via the simulator, triggered
`drift_check_dag`:

| task | result |
|---|---|
| `check_drift` | success — max PSI 0.155 (across all features), below the 0.2 threshold |
| `drift_detected` (short-circuit gate) | evaluated `False` |
| `trigger_retraining` | **skipped** — correctly did not retrain on normal traffic |

**Step 2 — drift detected, retraining auto-triggered.** Sent 150 drifted requests (mixed into the
same rolling window), triggered `drift_check_dag` again:

| task | result |
|---|---|
| `check_drift` | success — 5 features exceeded PSI 0.2: `customer_age` (1.75), `velocity_6h` (1.56), `velocity_24h` (2.39), `velocity_4w` (2.49), `session_length_in_minutes` (2.12) |
| `drift_detected` | evaluated `True` |
| `trigger_retraining` | success — automatically triggered `training_dag`, no manual step |

The triggered `training_dag` run completed all five tasks (`ingest → validate → train → evaluate →
register`) successfully. The new challenger (trained on a freshly-drawn data sample — `ingest_data`
uses no fixed random seed inside the DAG specifically so repeated retrains see new data rather than
deterministically tying with the existing champion) scored val_pr_auc **0.1549** vs. the prior
champion's **0.1430**, clearing the 2% promotion margin: **promoted** to `Production`, prior
champion archived.

**Step 3 — the loop closes: serving picks up the new model with zero manual steps.** The `serving`
container's background poller picked up the new `Production` version automatically; `GET /health`
and a live `POST /predict` both reported `model_version: 5` (the newly auto-trained model) without
any restart, redeploy, or manual `/reload-model` call — the only human actions in this entire test
were sending the two simulated traffic batches and triggering `drift_check_dag`.
