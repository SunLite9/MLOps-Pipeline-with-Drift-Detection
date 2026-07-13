# Fraud Detection MLOps Pipeline

A closed-loop MLOps system that trains a fraud classifier, serves it as an API, watches live traffic for drift, and automatically retrains and redeploys itself when the data shifts — with a promotion gate that only lets genuinely better models into production.

> For the full build history, every design tradeoff, and every measured result: see [DESIGN.md](DESIGN.md).

## Problem and motivation

Most ML portfolio projects stop at "I trained a model." In production, models decay — the world drifts away from the training data, and a model deployed once and never revisited silently gets worse. This project demonstrates the part that actually matters for running ML in production: tracking experiments, gating promotions so a worse model can never silently replace a better one, serving a model as a real service, detecting when live traffic has drifted from training data, and closing the loop by retraining and redeploying automatically — no human polling a dashboard required.

## Key features

- **Tracked training** — every run's params, metrics, and model artifact logged to MLflow.
- **Champion/challenger promotion gate** — a new model only replaces the current one in production if it beats it by a real margin, not by chance.
- **Dockerized serving API** — FastAPI service that loads the current production model and picks up new promotions automatically, with zero redeploy.
- **Automatic drift detection** — Population Stability Index (PSI) computed hourly against the training distribution.
- **Closed-loop auto-retraining** — drift crossing a threshold automatically triggers a full retraining run via Airflow DAG-triggers-DAG.
- **Monitoring dashboard** — prediction volume, latency, drift score trend, and the retrain/promotion timeline.
- **CI/CD** — tests and a Docker build run on every push, fully hermetic (no live services or real dataset needed).

## Architecture

```
 Bank Account Fraud dataset (raw CSV, not committed)
                 │
                 ▼
   ┌─────────────────────────┐        ┌────────────────────────────┐
   │  training_dag (Airflow) │        │ MLflow tracking + registry │
   │  ingest → validate      │───────▶│  experiments, model        │
   │  → train → evaluate     │        │  versions, Production      │
   │  → register_if_better   │◀───────│  stage, promotion history  │
   └─────────────────────────┘        └────────────────────────────┘
                 ▲                                   │
                 │ TriggerDagRunOperator              │ loads Production model
                 │ (on drift)                          ▼
   ┌─────────────────────────┐        ┌────────────────────────────┐
   │  drift_check_dag        │        │  serving API (FastAPI,     │
   │  (hourly)                │        │  Dockerized)               │
   │  reads recent            │◀───────│  POST /predict             │
   │  predictions, computes   │  logs  │  GET  /health               │
   │  PSI vs. training dist.  │        │  background auto-reload    │
   └─────────────────────────┘        └────────────────────────────┘
                 ▲                                   ▲
                 │                                   │
                 └──────────── data/predictions.db ───┘
                                   │
                                   ▼
                     dashboard/app.py (Streamlit)
                volume · latency · drift score · retrain/promotion timeline

              GitHub Actions: tests + serving image build on every push
```

**Flow in words:** Airflow trains a model → MLflow tracks it and gates promotion → the serving API loads whatever model is current "Production" → every prediction is logged → an hourly job compares live traffic to the training distribution → if it's drifted enough, retraining fires automatically → if the new model is genuinely better, it gets promoted → serving picks it up within 30 seconds, no redeploy.

## Tech stack

| Layer | Choice |
|---|---|
| Model | XGBoost (`XGBClassifier`) |
| Experiment tracking / registry | MLflow |
| Orchestration | Apache Airflow (`LocalExecutor`) |
| Serving | FastAPI + Pydantic |
| Drift detection | Population Stability Index (PSI), custom implementation |
| Dashboard | Streamlit + Altair |
| Data | pandas, PyArrow (Parquet), SQLite (prediction log) |
| Containers | Docker + Docker Compose |
| CI | GitHub Actions |
| Tests | pytest, httpx |

## Installation and setup

**Prerequisites:** Python 3.11, Docker Desktop, and the raw dataset placed at `Bank Account Fraud Dataset Suite (NeurIPS 2022)/Base.csv` ([BAF dataset on Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) — not included in this repo).

```bash
# 1. Python environment
python -m venv .venv
./.venv/Scripts/activate          # Windows
pip install -r requirements.txt

# 2. MLflow tracking server (host process — see DESIGN.md §11.9 for why)
mkdir -p mlflow_store/artifacts
mlflow server --host 127.0.0.1 --port 5000 --workers 1 --serve-artifacts \
  --artifacts-destination ./mlflow_store/artifacts \
  --backend-store-uri sqlite:///mlflow_store/mlflow.db \
  --allowed-hosts "localhost:5000,127.0.0.1:5000,host.docker.internal:5000"

# 3. Postgres + Airflow + serving API (one Docker network)
cd docker
docker compose build
docker compose up -d postgres
docker compose up airflow-init        # one-time: DB migration + admin user
docker compose up -d airflow-webserver airflow-scheduler serving

# 4. Monitoring dashboard
cd ..
streamlit run dashboard/app.py
```

| Service | URL |
|---|---|
| MLflow UI | http://127.0.0.1:5000 |
| Airflow UI | http://localhost:8080 (`admin` / `admin`) |
| Serving API | http://localhost:8000 |
| Dashboard | http://localhost:8501 |

## Usage examples

**Predict:**
```bash
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
# → {"fraud_probability": 0.0377, "fraud_prediction": 0, "model_version": "6"}

curl http://localhost:8000/health
# → {"status": "ok", "model_version": "6"}
```

**Trigger training manually** (optionally overriding hyperparameters):
```bash
docker compose exec airflow-scheduler airflow dags trigger training_dag
docker compose exec airflow-scheduler airflow dags trigger training_dag \
  --conf '{"n_estimators": 500, "max_depth": 4, "learning_rate": 0.05}'
```

**Simulate traffic to test the drift detector:**
```bash
python -m src.drift.simulate_drift --n 300              # normal traffic
python -m src.drift.simulate_drift --n 300 --drifted     # deliberately shifted traffic
docker compose exec airflow-scheduler airflow dags trigger drift_check_dag
```

## Results and metrics

| What was tested | Result |
|---|---|
| Promotion gate — bootstrap / reject / promote (real DAG triggers) | 0.1430 promoted (bootstrap) → 0.0307 correctly rejected → 0.1509 promoted (beat champion by >2%) |
| Serving — live promotion pickup, no redeploy | `/health` moved v2 → v9 automatically within the 30s poll interval |
| Drift detection — normal vs. drifted traffic | Normal: max PSI 0.155 (no trigger). Drifted: PSI up to 2.49 on 5 features, retrain auto-triggered |
| Full closed loop, one demo session | Drift score climbed 0.204 → 8.29 across 5 checks; 2 real promotions, 2 challengers correctly rejected |
| Test suite | 10/10 passing, fully hermetic (no live server or real dataset needed) |
| Serving latency (sequential) | avg ~11ms, p99 ~18ms per prediction |
| Serving latency (20 concurrent clients, 500 requests) | avg ~496ms, p99 ~3.4s, 0 failures — a real single-worker throughput ceiling, see [DESIGN.md §16.6](DESIGN.md#166-load-test-concurrent-vs-sequential-latency) |
| Concurrent-write reliability bug found & fixed | SQLite writes under load initially failed 8/500 requests (`database is locked`); fixed with WAL mode, verified 0/500 failures after |

Full numbers, run IDs, and the reasoning behind each threshold are in [DESIGN.md §16](DESIGN.md#16-results-and-observed-behavior).

## Testing

```bash
pytest tests/ -v
```

Fully hermetic — no live MLflow server, no running Airflow, no real dataset required. `tests/conftest.py` spins up a throwaway local MLflow store and a synthetic model matching the real schema, so the suite runs identically locally and in CI.

- `tests/test_promotion.py` — unit tests for the champion/challenger gate itself: bootstrap, rejection below the margin, promotion above it, the exact-boundary case, a custom margin, and the error path for an unregistered run.
- `tests/test_drift_detector.py` — unit tests for the PSI math.
- `tests/test_serving_integration.py` — API integration tests (`/predict`, `/health`, `/reload-model`, request validation, prediction logging).

CI also validates both Airflow DAGs actually import without error (`.github/workflows/ci.yml`'s `validate-dags` job) — this is a real check, not a formality: it's exactly the class of bug (a TaskFlow parameter name colliding with an Airflow reserved word) that broke the training DAG earlier in development.

**Load testing** (not part of the hermetic suite — hits a real running `serving` container):
```bash
python -m scripts.load_test --concurrency 20 --requests 500
```
This is how the concurrent-write bug in Results and metrics above was actually found.

## Project structure

```
src/training/       data loading, preprocessing, training, promotion gate
src/serving/        FastAPI inference service
src/drift/          drift detection + retraining trigger + traffic simulator
dags/               Airflow DAGs (training_dag, drift_check_dag)
docker/             Dockerfiles + the Docker Compose stack
dashboard/          Streamlit monitoring dashboard
scripts/            operational scripts (load testing)
tests/              hermetic test suite (unit + integration)
.github/workflows/  CI (tests + serving image build + DAG validation)
```

## Design decisions and tradeoffs

A few of the bigger calls — full reasoning and every alternative considered in [DESIGN.md §11](DESIGN.md#11-design-decisions-alternatives-and-tradeoffs):

- **250k-row stratified downsample**, not the full 1M rows — chosen after explicitly comparing 100k/250k/500k for iteration speed vs. metric stability.
- **PR-AUC, not ROC-AUC**, as the promotion metric — ROC-AUC is misleadingly optimistic at ~1% fraud prevalence.
- **2% relative-improvement margin** on the promotion gate, not "any improvement" — filters out run-to-run noise so the gate doesn't churn on chance.
- **`LocalExecutor` for Airflow**, not the default `CeleryExecutor` — lighter weight for a single-machine project, and it means all tasks share one filesystem with no extra distributed-storage setup.
- **MLflow runs as a host process, not a container** — a Windows-specific `uvicorn` bug made this the lower-risk path once the host-based setup was already proven working with Airflow and serving.
- **Raw (pre-encoding) request schema** for the serving API — keeps the client contract stable even as the model's internal encoded feature set shifts slightly between training runs.

## Limitations and future improvements

- **Categorical features aren't monitored for drift** — only numeric features get a PSI score.
- **PSI at small windows produces occasional false positives** on this dataset's heavy-tailed features — mitigated by tuning the window size, not eliminated. A production version would likely require drift to persist across multiple consecutive checks before triggering.
- **No authentication** on any endpoint (Airflow, MLflow, or the serving API) — fine for a local demo, not for anything exposed.
- **No horizontal scaling** — everything is single-instance. Load testing (`scripts/load_test.py`) confirmed a real throughput ceiling under concurrency (single `uvicorn` worker, no `--workers` flag): p99 latency goes from ~18ms sequential to ~3.4s at 20 concurrent clients. This is a legitimate scaling limit, not a bug — fixing it would mean multiple worker processes or an async inference path, neither implemented.
- **No registry retention policy** — every trained model version is kept forever.
- Kubernetes and cloud deployment weren't attempted; everything runs locally via Docker Compose.

Full list with context in [DESIGN.md §21](DESIGN.md#21-limitations-known-issues-technical-debt-and-future-work).

## Deployment

Everything currently runs locally:
- **MLflow** as a host process (`mlflow server ...`, see Installation above).
- **Postgres, Airflow, and the serving API** as Docker containers on one Compose network (`docker/docker-compose.yaml`).
- **The dashboard** as a local Streamlit process.

**CI:** `.github/workflows/ci.yml` runs the test suite on every push/PR, and builds the serving Docker image on every push to `main` to catch a broken build before it would affect a deploy. There is no automatic deploy step — CI validates correctness and buildability only.

## Contributing and license

Personal portfolio project — not currently accepting external contributions. No license has been set; treat as all-rights-reserved unless a `LICENSE` file is added.
