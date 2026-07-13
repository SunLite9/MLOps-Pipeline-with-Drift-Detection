# MLOps Pipeline with Drift Detection — Design Document

A closed-loop MLOps system built around a deliberately simple fraud classifier: tracked training, a gated model registry, containerized serving, PSI-based drift detection that automatically triggers retraining, and CI/CD — the full production lifecycle around a model, not the model itself.

## 1. Executive Overview

This project is a working demonstration of the "production ML lifecycle" — everything that sits around a trained model in a real deployment, as opposed to a notebook that ends at `model.fit()`. The model is intentionally boring (XGBoost on tabular fraud data); the system around it is the actual deliverable.

The pipeline: raw fraud data → stratified downsample → temporal train/val/test split → tracked training run (MLflow) → a champion/challenger promotion gate that only accepts genuinely better models → a Dockerized FastAPI service that serves the current Production model and logs every prediction → an hourly drift check that compares live traffic against the training distribution using the Population Stability Index (PSI) → if drift crosses a threshold, an automatic Airflow DAG-triggers-DAG retraining run, which (if the new model clears the gate) gets promoted, and which the serving API picks up within 30 seconds with zero manual intervention → a Streamlit dashboard showing volume, latency, drift score, and the retrain/promotion timeline → GitHub Actions running the test suite and a Docker build on every push.

It was built incrementally, milestone by milestone, each one producing real, measured, verified output rather than a description of intended behavior — every claim in this document is something that was actually run and observed, including the failures encountered along the way and how they were diagnosed.

**Live proof point, produced in the final end-to-end demo (Section 16):** a drift score climbing from 0.204 → 0.208 → 0.208 → 0.216 → **8.29** across five checks, two automatic promotions (v1 → v5 → v6), and two challengers correctly rejected by the gate — all without a human touching the registry, the serving container, or a deploy pipeline.

## 2. Problem Definition and Context

### 2.1 The real problem

Most ML portfolios stop at "I trained a model and it got X% accuracy." That demonstrates modeling skill but says nothing about whether the author can operate a model in production — the actual job of an ML engineer at most companies with an existing data science function. Models decay: the world the model was trained on changes, and a model deployed once and never revisited silently degrades. The real engineering problem is not "can you fit a classifier" but:

- Can you track and compare experiments reproducibly, not just eyeball a notebook cell's printed metric?
- Can you prevent a worse model from silently replacing a better one in production?
- Can you serve a model as a real network service that survives being promoted without a redeploy?
- Can you detect, automatically and without a human polling a dashboard, when the live world has diverged from the training data?
- Can you close the loop — detection triggers action, action produces a new model, and the new model reaches production — with no manual step in between?
- Can you prove all of the above happened, with logs and numbers, not just a claim that it works?

### 2.2 Why it matters

In a real deployment, a fraud model trained on last year's fraud patterns degrades as fraud rings adapt — this is adversarial data, not naturally stationary data like retail demand. A production fraud system that doesn't monitor for drift and doesn't have an automated retraining path either (a) needs a human to notice a metrics dashboard trending badly and intervene manually — slow, error-prone, and doesn't scale past a few models per team — or (b) silently keeps serving a stale model until someone investigates a spike in missed fraud. Both are real failure modes at companies without this kind of system. The project exists to demonstrate the mechanism that avoids both.

### 2.3 Why this resists the "obvious" solution

The naive version of this project is "write a cron job that retrains nightly." That's not the same problem: unconditional scheduled retraining doesn't know *whether* retraining was warranted, doesn't protect against a worse retrain silently replacing a better model, and doesn't demonstrate the actual sensing mechanism (drift detection) that a real system needs to justify *when* to retrain rather than retraining on a fixed, arbitrary schedule regardless of whether the data actually moved. The interesting engineering is in the conditional trigger and the promotion gate, not the retraining code itself (which is a normal `model.fit()` call).

## 3. Goals, Success Criteria, and Scope

### 3.1 Goals

1. Demonstrate the full ML lifecycle: train → track → gate → serve → monitor → detect drift → auto-retrain → re-promote → re-serve, with every step actually running, not just described.
2. Make the closed loop (drift → retrain → promote → serve) the centerpiece, since that's the single hardest and most valuable thing to demonstrate — everything else in the pipeline exists to support proving that loop closes.
3. Produce real, reproducible, measured results at every phase — not "should work," but "ran this, got this number."
4. Keep the model itself minimal effort — this is a systems project, not a Kaggle leaderboard entry.

### 3.2 Success criteria

- Push code → CI builds and tests it.
- The training DAG runs on schedule (and on demand).
- Feeding the system deliberately shifted data causes the drift detector to fire and trigger an automatic retraining run.
- That automatic retraining promotes a new model (when warranted) that the live API picks up — all without a manual step, and all visible on a dashboard.

Every one of these was independently verified during the build; the exact runs and numbers are in Sections 14–16.

### 3.3 Explicit non-goals

- **State-of-the-art fraud detection accuracy.** The model is a single XGBoost classifier on a downsampled dataset with no hyperparameter search infrastructure, no ensembling, no feature engineering beyond one-hot encoding. This is deliberate (see Section 4.2).
- **Multi-tenant / multi-model serving.** One model (`fraud-model`), one registry, one serving process.
- **Horizontal scaling / high availability.** Single-instance services throughout; Kubernetes was considered as a possible extension but was not attempted (Section 21).
- **Categorical-feature drift detection.** PSI is computed only on numeric raw features (Section 9.3); categorical drift is a documented gap.
- **Cloud deployment.** Everything runs locally (Docker Desktop + a host MLflow process) on a Windows development machine. No AWS/GCP/Azure component was built — cloud deployment was out of scope from the start.

## 4. Requirements and Constraints

### 4.1 Functional requirements

- Track every training run's hyperparameters, metrics, and model artifact.
- Maintain a model registry with named stages (`Production` / `Archived`), and gate promotion on a quantified improvement over the current champion.
- Orchestrate training as a scheduled, manually-triggerable DAG with discrete, independently testable steps.
- Serve the current Production model over HTTP with request validation, and pick up newly-promoted models without a redeploy.
- Log every prediction (inputs, output, timestamp, model version, latency) for downstream monitoring.
- Detect distributional drift in live traffic relative to the training distribution, on a schedule, and automatically trigger retraining when drift crosses a defined threshold.
- Provide a dashboard showing volume, latency, drift trend, and the retrain/promotion history.
- Run the test suite and build the serving image automatically on every push.

### 4.2 Constraints that shaped the design

- **The model was deliberately kept simple and fast-training, by design** — the point of this project was never to maximize model accuracy but to prove out the system around the model. This directly drove the choice of a single XGBoost classifier with no hyperparameter search, and the decision to downsample the dataset (Section 11.1) rather than train on the full 1M rows.
- **Windows 11 development machine, Docker Desktop.** Two real technical constraints followed from this and consumed significant debugging time: (a) `mlflow server`'s multi-worker `uvicorn` mode has a socket-binding bug on Windows (`OSError: [WinError 10022]`) that doesn't occur on Linux, forcing a single-worker MLflow process; (b) Docker Desktop's `host.docker.internal` networking, combined with MLflow's DNS-rebinding protection (`--allowed-hosts`) and its artifact-serving model (`--serve-artifacts`), required explicit configuration that wouldn't be needed if MLflow and its clients were all on the same host filesystem.
- **The raw dataset (~200MB CSV) is not committed to git.** This is a hard constraint on the test/CI design: nothing that runs in CI can assume the real dataset is present, which is why the test suite had to become fully synthetic-data-driven (Section 14.1) rather than relying on the real BAF data.
- **No message queue / Celery infrastructure wanted.** Airflow's default quick-start uses `CeleryExecutor` (Postgres + Redis + webserver + scheduler + worker + triggerer). `LocalExecutor` was chosen instead — a deliberate scope reduction, documented in Section 11.4.

## 5. System Architecture

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

### 5.1 Process/deployment topology

| Component | Runs as | Why |
|---|---|---|
| MLflow tracking server | Host process (not containerized) | Windows `uvicorn` multi-worker bug forced a single-worker process; rather than fight that inside a container too, it was kept simple as a host process reachable from containers via `host.docker.internal` |
| Postgres | Docker container (Airflow's metadata DB) | Standard Airflow requirement |
| Airflow webserver + scheduler | Docker containers, `LocalExecutor` | Same container runs both scheduling and task execution — no separate worker/broker needed |
| Serving API | Docker container | Needed to be genuinely containerized, not run as a dev-mode process |
| Dashboard | Host process (`streamlit run`) | Read-only monitoring tool; no reason to containerize for this project's scope |
| GitHub Actions | Remote CI runner | Standard |

All Docker services share one Docker Compose network and one compose file (`docker/docker-compose.yaml`) — Postgres, Airflow (webserver + scheduler), and the serving API. This consolidation happened mid-project (Section 12) once serving needed to reach the same Airflow-adjacent network as the rest of the stack, and the cleanest way to get there without re-litigating the MLflow-in-Windows problem was to merge the previously-separate Airflow-only compose file with a new serving service, rather than stand up MLflow as a third network member.

### 5.2 Why MLflow was *not* containerized

This was a deliberate, reconsidered decision (see Section 11.9 for the full alternatives analysis). Short version: containerizing MLflow would have required re-solving the Windows socket bug *inside* a Linux container (probably fine) but also migrating already-verified registry state and re-proving the artifact-proxying configuration in a new context, for a benefit (perfect topological symmetry) that wasn't load-bearing — Airflow and serving both already had a working, tested pattern for reaching a host-based MLflow via `host.docker.internal`. Extending that proven pattern to a third consumer was lower-risk than introducing a new failure surface.

## 6. End-to-End System Flow

### 6.1 Training flow (`training_dag`, also runnable manually via `python -m src.training.train`)

1. **`ingest_data()`** — loads the raw 1M-row CSV, stratified-downsamples to 250,000 rows (preserving the ~1.1% fraud rate), one-hot encodes the 5 categorical columns, splits temporally by the `month` column (0–5 train / 6 val / 7 test), writes each split to Parquet under `data/processed/`.
2. **`validate_data(paths)`** — sanity checks: required columns present, no nulls in required columns, train row count within `[150_000, 350_000]`, fraud rate within `(0%, 10%)`. Raises (fails loudly) on any violation.
3. **`train_model(paths, n_estimators, max_depth, learning_rate)`** — fits an `XGBClassifier` with `scale_pos_weight` corrected for class imbalance, logs params + train metrics + the model artifact to MLflow, registers it under `fraud-model` (unstaged), and logs two additional artifacts: `feature_columns.json` (the exact post-encoding column set, for serving) and `training_distribution.json` (per-feature quantile bins, for drift detection). Returns the MLflow `run_id`.
4. **`evaluate_model(run_id, paths)`** — reloads the model from the run, scores it on the validation split, logs `val_roc_auc` and `val_pr_auc` back to the same run, returns `val_pr_auc` as the promotion metric.
5. **`register_if_better(run_id, metric)`** — the gate (Section 9.2): promotes to `Production` (archiving the prior champion) only if the challenger beats the current champion's `val_pr_auc` by ≥2% relative, or unconditionally if there is no current champion.

### 6.2 Serving flow (`src/serving/api.py`)

1. At container startup, `reload_if_new_version()` loads whatever model is currently `Production`, plus its `feature_columns.json`.
2. `POST /predict` accepts a raw (pre-encoding) JSON body validated against a Pydantic schema mirroring the dataset's original columns, one-hot encodes it identically to training, reindexes to the loaded model's exact `feature_columns`, runs inference, logs the request/response/latency to `data/predictions.db`, and returns the prediction plus which model version served it.
3. A background thread re-checks the registry every 30 seconds and swaps in a new `Production` version if one has been promoted since the last check — no restart. `POST /reload-model` does the same check synchronously, on demand.

### 6.3 Drift-and-retrain flow (`drift_check_dag`, hourly)

1. **`check_drift`** — calls `src.drift.detector.check_for_drift()`, which reads the most recent 300 rows from `data/predictions.db`, downloads the current Production model's `training_distribution.json`, computes PSI per numeric feature, and returns whether any feature exceeded 0.2.
2. **`drift_detected`** — an `@task.short_circuit` gate: if `check_drift`'s result says no drift, every downstream task is skipped and the DAG run ends cleanly.
3. **`trigger_retraining`** (`TriggerDagRunOperator`) — only reached if drift was detected; fires a new `training_dag` run.
4. The triggered `training_dag` run executes the full flow in 6.1, including a **fresh, un-seeded data sample** (Section 11.7) rather than the exact same 250k rows every time, so repeated retrains see genuinely new data.
5. If the resulting challenger clears the promotion gate, it becomes the new Production model; the serving API's background poller picks it up within 30 seconds.

### 6.4 Monitoring flow (dashboard)

Every `check_for_drift()` call also logs itself as an MLflow run under a `drift-monitoring` experiment (`max_psi`, `n_samples`, `drifted` tag). The dashboard queries `data/predictions.db` for volume/latency, the `drift-monitoring` experiment for the PSI trend, and the `fraud-model` registry's version history for the retrain/promotion timeline — no separate metrics pipeline exists; the dashboard reads the same sources everything else writes to.

## 7. Component-Level Design

### 7.1 `src/training/data.py` — data loading and splitting

Pure functions, no side effects beyond reading the CSV: `load_raw_data()`, `downsample()`, `preprocess()`, `temporal_split()`, and the convenience `load_splits()` used by the standalone `train.py` script. Owns the two central schema constants reused everywhere else in the codebase: `CATEGORICAL_COLS` (the 5 columns that get one-hot encoded) and `TARGET_COL`/`MONTH_COL`. Every other module that needs to know "what are the categorical columns" or "what's the target column name" imports from here rather than redefining it — this is why `src/serving/api.py` and `src/drift/detector.py` both import `CATEGORICAL_COLS` from `src.training.data` instead of hardcoding their own copy.

### 7.2 `src/training/pipeline.py` — Airflow-callable discrete steps

The four functions in Section 6.1 (`ingest_data`, `validate_data`, `train_model`, `evaluate_model`), designed specifically so each is independently unit-callable and passes only JSON-serializable values between steps (paths, run IDs, floats) — a deliberate constraint for Airflow XCom compatibility (Section 9.1). Also owns the artifact-logging contract that the serving and drift-detection components depend on (`feature_columns.json`, `training_distribution.json`) — this is a cross-cutting dependency worth calling out explicitly: **`pipeline.py` imports from `src.drift.detector`** (`compute_training_distribution`) even though drift detection is conceptually a "later" phase of the project. This is intentional coupling, not an accident: the training step is the only place that has the raw training DataFrame in memory, so it's the only place that can compute a training-distribution artifact for later drift comparison.

### 7.3 `src/training/promotion.py` — the gate

A single function, `register_if_better()`, deliberately kept separate from `pipeline.py` (rather than folded into `train_model()` or `evaluate_model()`) because it's the one piece of business logic in the whole training flow that encodes a *policy decision* (the 2% margin) rather than a mechanical step — keeping it isolated makes the promotion rule a one-file, one-function unit that's trivial to point at, explain, and unit-test in isolation from the rest of training.

### 7.4 `src/serving/api.py` — the serving process

A FastAPI app with three endpoints (`/predict`, `/health`, `/reload-model`) and one background thread. The `ModelState` class exists purely to make the shared mutable state (the loaded model, its version, its feature columns) explicitly lock-guarded, since it's read by every `/predict` request and written by both the startup path and the background poller concurrently.

### 7.5 `src/drift/detector.py` — PSI computation and the drift decision

Pure-function core (`compute_training_distribution`, `compute_drift`, `is_drifted` — no I/O, fully unit-testable with synthetic data, Section 14.1) wrapped by an I/O layer (`_load_recent_predictions`, `_load_production_training_distribution`, `_log_drift_check`) that talks to SQLite and MLflow. `check_for_drift()` is the single function everything else calls — `drift_check_dag` calls it directly; nothing downstream needs to know about bin edges or PSI math.

### 7.6 `src/drift/simulate_drift.py` — the test/demo traffic generator

Not part of the production system — a tool for proving the system works. Samples real rows from the raw dataset (so "normal" traffic is genuinely representative) and, in `--drifted` mode, multiplicatively shifts five behavioral features before sending. Exists because waiting for real-world drift to occur is not a viable way to test or demo a drift detector.

### 7.7 `dags/training_dag.py` and `dags/drift_check_dag.py`

Thin Airflow wrappers around the `src/` modules — each task's body is a two-line `from src.X import Y; return Y(...)` call. This is deliberate: the DAG files contain almost no logic of their own, which means the actual pipeline logic is testable without Airflow at all (as the unit/integration tests do) and the DAG files' only job is orchestration wiring (task ordering, XCom, the conditional trigger).

### 7.8 `dashboard/app.py` — the monitoring UI

Read-only. Four `st.cache_data(ttl=15)`-cached query functions (`load_predictions`, `load_drift_history`, `load_model_timeline`) each independently reading from either SQLite or MLflow, rendered as Altair charts. No write path — the dashboard cannot affect the system, only observe it.

## 8. Data Design

### 8.1 Source dataset

[Bank Account Fraud (BAF) Dataset Suite](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) (NeurIPS 2022), **Base variant** specifically (not Variants I–V — see Section 11.1 for why). ~1M rows, 30 raw features, binary target `fraud_bool`, ~1.1% fraud prevalence, synthetic but modeled on real bank account opening fraud data. Includes a `month` column (0–7) giving the data a genuine temporal structure.

### 8.2 Sampling

Stratified downsample to 250,000 rows (`SAMPLE_SIZE` in `data.py`), preserving the ~1.1% fraud rate via per-class `.sample(frac=...)` then a shuffle. 250k was chosen after an explicit three-way comparison against 100k and 500k (Section 11.2) as the point where iteration stays fast without the training signal becoming too thin.

### 8.3 Splitting

**Temporal, not random.** Months 0–5 → train (~198k rows), month 6 → validation (~27k rows), month 7 → test (~24k rows, currently unused by the pipeline — held out for a true final holdout evaluation that the project didn't need, since promotion decisions are made on validation). This choice is doing double duty: it's the methodologically correct way to evaluate a model that will be deployed on future data (train on the past, evaluate on data that comes after), and it's what makes the drift-detection story coherent — a model trained on months 0–5 and evaluated on month 6 has, by construction, already been tested against *some* natural distribution shift, and the same temporal logic extends naturally to "live traffic is data from even further in the future, which may have shifted further still."

### 8.4 Feature encoding

Five categorical columns (`payment_type`, `employment_status`, `housing_status`, `source`, `device_os`) are one-hot encoded via `pd.get_dummies(..., drop_first=False)`. All other columns, including the `-1` "missing" sentinel values baked into several BAF numeric columns (e.g. `prev_address_months_count`), are passed through unchanged — XGBoost's tree splits handle a sentinel value like any other number without needing explicit missing-value imputation. **`month` is retained as a real numeric feature**, not just a split key (Section 11.7 for the consequences of this decision on retraining variance).

### 8.5 Data at rest

- **`data/processed/{train,val,test}.parquet`** — regenerated by `ingest_data()` on every training run (not versioned; ephemeral working data).
- **`data/predictions.db`** (SQLite) — the prediction log: `timestamp`, `model_version`, `input_json` (the full raw request as JSON), `fraud_probability`, `fraud_prediction`, `latency_ms`. This is the *only* durable record of live traffic and is what both the drift detector and the dashboard read.
- **MLflow's own store** (`mlflow_store/`, SQLite backend + a local artifact directory) — experiments, runs, params, metrics, and the three logged artifacts per training run (`model`, `feature_columns.json`, `training_distribution.json`).

### 8.6 Data contracts between components

The tightest coupling in the system is the **feature schema**, which crosses three components (training, serving, drift detection) and must stay in exact agreement:

- Training determines the real column set (raw + one-hot dummies) and freezes it into `feature_columns.json` per run.
- Serving reads that file for whichever run is currently `Production` and reindexes every incoming request to match it exactly — this is what makes it safe for the *set* of one-hot dummy columns to differ slightly between training runs (e.g. if a rare category happens not to appear in one 250k sample) without breaking inference: missing dummies get filled with 0, unexpected ones get dropped.
- Drift detection reads `training_distribution.json` for the same run and compares only the numeric (non-dummy) columns, identified structurally by `numeric_columns()` (anything not prefixed by a known categorical column name) rather than by a separate hardcoded list — so this too stays in sync automatically with whatever `CATEGORICAL_COLS` says, without a second source of truth.

## 9. Algorithms, Models, and Technical Methods

### 9.1 The model

`xgboost.XGBClassifier`, `objective="binary:logistic"`, `eval_metric="aucpr"`, `scale_pos_weight` set to the train-split negative:positive ratio (correcting for ~1.1% fraud prevalence), `random_state=42` fixed **inside** the model (the *data sampling* seed is a separate, deliberately different decision — Section 11.7). Hyperparameters (`n_estimators`, `max_depth`, `learning_rate`) are the only tunable surface, exposed via the DAG's `dag_run.conf` for manual override during testing (Section 14 uses this extensively to force specific promotion outcomes).

No hyperparameter search infrastructure exists — hyperparameters were hand-picked and hand-tuned by direct trial (Sections 14 and 16 document several rounds of manually searching for combinations that would cross a specific promotion threshold, purely to produce a demonstrable promotion event). This is consistent with the project's explicit non-goal of state-of-the-art accuracy (Section 3.3).

### 9.2 The promotion gate

```
promoted = True                                     if no current Production model exists
promoted = (challenger.val_pr_auc >= champion.val_pr_auc * 1.02)   otherwise
```

**Metric choice: PR-AUC, not ROC-AUC.** At ~1.1% fraud prevalence, ROC-AUC is a known-misleading metric under heavy class imbalance (a classifier that's mediocre at ranking the rare positive class can still post a deceptively high ROC-AUC, because the metric is dominated by the vast majority-negative population). PR-AUC directly reflects precision/recall tradeoffs on the minority class, which is what actually matters for a fraud model.

**Margin choice: 2% relative, not "any improvement."** A zero-margin gate ("promote if strictly better") would promote on pure training noise — two runs with identical hyperparameters and slightly different random data samples will produce slightly different metrics essentially always, and a zero-margin gate would treat that as a real signal, causing promotion churn with no underlying improvement. 2% was chosen as a threshold large enough to filter out that noise (empirically, repeat runs on this dataset/model combination showed metric variance in the low single-digit percent range) while still being small enough to accept genuine, modest improvements rather than requiring an implausibly large jump.

**Bootstrap case.** The very first model ever trained has no champion to compare against; it's promoted unconditionally. This is the correct behavior but has a side effect worth naming: it means the *first* promotion in any fresh registry carries zero quality guarantee — anything, however weak, becomes Production if it's first. This is acceptable for this project (there's always a next retraining cycle to correct it) but would be a real gap in a system where the first deployed model needed to meet some absolute bar.

**Held-out test-set evaluation, added after the fact (Section 20.6).** `evaluate_test_set()` scores the trained model on the month-7 test split — the one temporal split that was defined from the very first training script onward but never actually read by anything until this was added — and logs `test_roc_auc`/`test_pr_auc` to the run purely for reporting. It is deliberately *not* passed to `register_if_better()`: the promotion decision is made on `val_pr_auc` alone, so the test set stays an honest, unused-by-the-decision-process check on generalization, rather than becoming a second metric the gate could end up implicitly tuned against if hyperparameters were ever chosen by watching it.

### 9.3 Population Stability Index (PSI)

Standard formula, computed per feature:

```
PSI = Σ (actual_i - expected_i) × ln(actual_i / expected_i)     over bins i
```

**Binning.** 10 quantile bins computed from the training data (`np.quantile` at deciles), stored as bin edges. Live data is bucketed into the *same* edges (not re-binned from scratch) — this is what makes the comparison meaningful; comparing two independently-binned histograms would conflate "the bins are different" with "the distribution is different." Values falling outside the training range entirely are folded into the nearest boundary bin rather than dropped, because falling outside the observed training range at all is itself a meaningful drift signal that dropping the values would silently discard.

**Epsilon smoothing.** `+1e-4` added to every proportion before the log/ratio, to avoid `log(0)` or division by zero when a bin has zero mass in either distribution (common with small live windows).

**Scope: numeric features only.** The one-hot-encoded categorical dummy columns are excluded by construction (`numeric_columns()` filters out anything prefixed by a `CATEGORICAL_COLS` name). PSI's quantile-binning approach doesn't naturally suit a small, fixed-cardinality categorical variable the way it suits a continuous one; a categorical frequency/chi-square test would be the correct follow-up and is an explicit, documented gap (Section 21).

**Threshold: PSI > 0.2, any single feature.** 0.2 is the standard industry rule of thumb (< 0.1: no significant shift; 0.1–0.2: moderate, worth watching; > 0.2: significant). The **any-single-feature** trigger condition (rather than requiring several features to drift together) was a deliberate design choice: an isolated shift in one important feature — e.g. transaction velocity alone — is a real, actionable fraud signal on its own, and waiting for corroboration from unrelated features would mean missing it.

**Window size: 300 (was 200).** This number was not chosen a priori — it was *discovered* during the Section 16 demo run, when normal (non-drifted) traffic repeatedly produced borderline PSI crossings (0.204–0.216) on the dataset's heavy-tailed velocity features at a 200-sample window. Diagnosis and fix are documented in full in Section 20.3; the short version is that 10 bins over 200 points is ~20 points/bin, and multinomial sampling noise at that bin size was large enough to cross 0.2 by chance on skewed features. 300 was verified empirically (Section 16) to settle that same noise below threshold on this dataset.

**Minimum sample size: 30.** Below 30 live samples, `check_for_drift()` returns `drifted=False` with an explicit "insufficient data" reason rather than computing a PSI on too few points to be meaningful.

### 9.4 Fresh-data-per-retrain

`training_dag`'s `ingest` task calls `ingest_data(random_state=None)` — deliberately *not* the module's default fixed seed (42). This was a bug-driven design change (Section 20.2): with a fixed seed, every retraining run — including auto-triggered ones — would deterministically reproduce the exact same 250k-row sample and therefore the exact same trained model, meaning an auto-retrain could only ever tie the existing champion, never genuinely beat it. `random_state=None` (pandas draws from the global RNG) means each retrain sees a genuinely different stratified sample of the underlying 1M rows, which is both a bug fix and a more realistic simulation of what "new data has arrived" means in a real retraining pipeline.

## 10. APIs, Interfaces, and Data Contracts

### 10.1 `POST /predict`

Request body (`PredictRequest`, Pydantic-validated): 30 raw fields matching the original dataset schema exactly (see `src/serving/api.py` for the full field list) — numeric fields as `int`/`float`, five categorical fields as `str` (`payment_type`, `employment_status`, `housing_status`, `source`, `device_os`), plus `month: int`. Missing or wrong-typed fields return `422` with a field-level error list (FastAPI/Pydantic default behavior, verified in Section 14).

Response (`PredictResponse`):
```json
{"fraud_probability": 0.0377, "fraud_prediction": 0, "model_version": "6"}
```

### 10.2 `GET /health`

```json
{"status": "ok", "model_version": "6"}
```
`status` is `"no_production_model"` if the registry has no `Production`-stage version at all (e.g. before the first promotion).

### 10.3 `POST /reload-model`

Forces an immediate registry check (rather than waiting up to 30s for the background poller). Returns `{"reloaded": bool, "model_version": str}`. `503` if no Production model exists.

### 10.4 Internal artifact contract (MLflow)

Every training run logs exactly three artifacts under a consistent naming scheme, read by name (not by index or convention) by downstream consumers:
- `model` (MLflow's `mlflow.xgboost` flavor) — loaded by both serving (`models:/fraud-model/{version}`) and drift-check evaluation (`runs:/{run_id}/model`).
- `feature_columns.json` — `{"feature_columns": [...]}`, read by serving.
- `training_distribution.json` — `{feature_name: {"bin_edges": [...], "expected_proportions": [...]}}`, read by the drift detector.

### 10.5 Airflow DAG interfaces

`training_dag` accepts an optional `dag_run.conf` of `{"n_estimators": int, "max_depth": int, "learning_rate": float}` to override the training step's defaults — used throughout testing to deliberately force weak or strong challengers (Section 14). `drift_check_dag` takes no parameters; it always operates on the current `Production` model and the most recent 300 predictions.

## 11. Design Decisions, Alternatives, and Tradeoffs

This section is the record of every consequential fork in the road, what was chosen, what was rejected, and why — reconstructed from the actual decision points that came up during the build.

### 11.1 Dataset variant: BAF Base, not Variants I–V

The BAF suite ships six variants: Base (representative sample) and five variants (I–V) that each introduce a specific fairness/robustness challenge (group size disparity, prevalence disparity, separability differences between groups). **Considered:** using one of the fairness variants, since they're the "interesting" part of the BAF paper. **Rejected because:** this project's centerpiece is drift detection via `simulate_drift.py`-injected shifts — a deliberately, controllably introduced distribution change. Using a variant with *built-in* disparities would confound the deliberately-injected drift signal with pre-existing dataset skew unrelated to the question being tested, and would pull the project's scope toward a fairness audit, which was never the goal here. Base was chosen specifically because it's the cleanest baseline to inject drift into on my own terms.

### 11.2 Dataset size: 250k rows, chosen from an explicit 100k/250k/500k comparison

Decided before any code was written. Running all three sizes in parallel and comparing was considered and rejected as unnecessary overhead — the model's own accuracy was never the goal, so spending build time optimizing a sample-size choice would have been effort in the wrong place. 100k rows (~1,000 positive examples at 1.1% prevalence) would already have been plenty for a system whose point is the pipeline, but 250k was judged the better tradeoff: materially more stable metrics for the promotion-gate story (a promotion gate's margin comparisons get noisier with fewer examples) at only a modest iteration-speed cost over 100k. 500k and the full 1M were rejected outright: full-size data actively works against the fast iteration this project depends on (every stage of the build involved triggering training runs repeatedly for verification and demos), for no benefit to the actual goal.

### 11.3 Model choice: single XGBoost classifier, no hyperparameter search

**Considered:** logistic regression (simpler, matches "deliberately simple" instruction most literally); a small hyperparameter search (Optuna/grid search) to make the promotion-gate story more realistic. **Rejected logistic regression** because BAF's feature set has meaningful nonlinear structure (velocity/behavioral features) that a tree model captures without manual feature engineering, and XGBoost was already needed for the fast native handling of the dataset's `-1` missing-value sentinels. **Rejected a hyperparameter search** because it directly conflicts with the "spend minimal effort on model accuracy" instruction — a search infrastructure is itself a form of investment in model quality that this project explicitly deprioritizes. Manual hyperparameter selection, including some rounds of deliberately hunting for a combination that would clear a specific promotion threshold purely to produce a demonstrable test case, was used instead (visible throughout Sections 14 and 16).

### 11.4 Airflow executor: `LocalExecutor`, not `CeleryExecutor`

Airflow's official quick-start docker-compose defaults to `CeleryExecutor` (Postgres + Redis + webserver + scheduler + worker + triggerer + optional Flower). **Rejected** in favor of `LocalExecutor` (Postgres + webserver + scheduler only) for two reasons: (1) resource footprint on a single development machine already running Docker Desktop, MLflow, and a Streamlit process; (2) `LocalExecutor` means every task in a DAG run executes as a subprocess of the scheduler container, sharing its filesystem — this directly simplified the `ingest → validate → train` data hand-off (Parquet files on a shared bind mount) without needing a separate distributed-filesystem or object-store solution that `CeleryExecutor`'s multiple worker containers would have required. The tradeoff accepted: no horizontal task-execution scaling, and a scheduler crash takes down in-flight tasks too — acceptable for a single-machine demo project, not acceptable as-is for a real multi-team production Airflow deployment.

### 11.5 Serving request schema: raw features, not pre-encoded

**Considered:** having `/predict` accept the model's actual post-one-hot-encoding feature vector directly (simpler server-side code, no `_preprocess()` function needed). **Rejected** because it would push the one-hot-encoding logic onto every API client, coupling every caller to the *current* model's exact encoded schema — which changes slightly between training runs depending on which categories happened to appear in that run's 250k sample. Instead, the server accepts the natural, human-readable raw schema and does the encoding + reindexing internally, using the `feature_columns.json` artifact to align to whatever the *currently loaded* model actually expects. This is more server-side complexity in exchange for a stable, model-version-independent client contract — judged worth it since a real API consumer shouldn't need to know or care about the model's internal encoding.

### 11.6 MLflow Model Registry stages API, despite deprecation warnings

MLflow 2.9+ deprecated the classic `Production`/`Staging`/`Archived` stage model in favor of registered-model **aliases**, and the installed MLflow version (3.14) emits `FutureWarning`s on every `get_latest_versions`/`transition_model_version_stage` call. **Considered:** switching to the alias-based API (`set_registered_model_alias`, `get_model_version_by_alias`) to avoid the deprecation noise. **Kept the stages API** because the promotion workflow this project is built around is naturally described in stage terms ("promote to Production, demote the old champion to Archived"), and the stage-based mental model (one designated "current" version, others explicitly archived, full history preserved) maps directly onto that promotion-gate narrative. This is a knowingly-accepted piece of technical debt (Section 21) — a future MLflow major version could remove stages entirely, at which point this would need to migrate to aliases.

### 11.7 `month` retained as a real predictive feature, not just a split key

The BAF `month` column serves two roles simultaneously in this codebase: it drives the temporal train/val/test split, *and* it's left in the feature matrix as an ordinary numeric input the model trains on. **Considered:** dropping it from the feature set after using it to split, on the reasoning that "the split key shouldn't also be a feature." **Kept it in** because a real fraud model plausibly benefits from seasonality (fraud patterns can vary by month), and because a serving client can supply "the current month" as a genuinely known value at inference time — it's not leaking future information the way it would if e.g. the target label were derived from it. The consequence discovered later (Section 20.2) is that this, combined with a fixed data-sampling seed, made repeat training runs deterministic and therefore unable to naturally out-perform each other — which is what drove the `random_state=None` fix in Section 9.4/20.2. Documented here as a case where a reasonable modeling choice had a non-obvious downstream interaction with a different part of the system (the promotion gate's reliance on run-to-run variance).

### 11.8 Prediction-log schema: raw JSON blob, not a normalized table

`data/predictions.db`'s `predictions` table stores the full request as a single `input_json` TEXT column rather than one column per feature. **Considered:** a fully normalized schema (one column per feature) for easier SQL querying. **Rejected** because the feature schema itself isn't fixed across the system's lifetime (Section 8.6 — the encoded column set can vary slightly between training runs), and a normalized table would need a schema migration every time that happened. A JSON blob column is schema-flexible by construction; the cost is that both the drift detector and the dashboard have to `json.loads()` each row rather than running a direct SQL aggregate over feature columns — an accepted tradeoff since the volumes involved (hundreds to low thousands of rows in this project's testing) make that cost negligible.

### 11.9 MLflow topology: host process, not a fourth Docker container

Covered in Section 5.2 at the architecture level; the decision history in full: the *first* attempt did try running `mlflow server` with its default multi-worker `uvicorn` mode, which crashed intermittently on Windows (`OSError: [WinError 10022]`, Section 20.1) — traced to a Windows-specific limitation in how `uvicorn`'s multi-process mode shares a listening socket (POSIX `SO_REUSEPORT`-style fd passing doesn't work the same way on Windows). The fix (`--workers 1`) was applied and verified stable. At that point, **containerizing MLflow was reconsidered** (to make the "everything on one Docker network" story more literal) but rejected: it would mean re-deriving the single-worker fix inside a fresh Linux container context (probably a non-issue on Linux, but unverified), re-establishing the artifact-proxying and allowed-hosts configuration in a new context, and migrating already-tested registry state — real risk and rework for a topological purity gain that wasn't load-bearing, since the already-proven `host.docker.internal` pattern (used successfully by Airflow) extended cleanly to a third consumer (serving) with no new problem class.

### 11.10 Hermetic test fixtures instead of a live-server test dependency

The original `tests/test_serving_integration.py` assumed a real, already-running MLflow server with an already-registered Production model — convenient to write, but meant the test suite could never run in CI (no dataset, no running server there) and was fragile to the local dev server's exact state. **Rejected staying with that design.** Replaced with `tests/conftest.py`'s `mlflow_test_env` fixture: a throwaway local SQLite-backed MLflow store, a synthetic dataset generated to match the real schema exactly (same columns, same categorical value pools, random values), a tiny model trained and promoted within the fixture. This is real engineering cost (writing and validating a synthetic-data generator, discovering and fixing the file-store-deprecation and `str(version)`-type issues it surfaced — Section 20.4) taken on specifically to make "tests pass" mean the same thing locally and in CI, which is the actual point of having CI at all.

## 12. Implementation and Project Evolution

The project grew through five natural milestones, each one a working, independently-verified increment building on the last. This section is the honest build order, including the parts that didn't work the first time.

**Tracked training.** Repo scaffolding, `src/training/data.py`, a standalone `train.py` script, a local MLflow server. First real friction: MLflow's run-summary log line contains an emoji that crashes on Windows' default `cp1252` console encoding after an otherwise-successful run (`UnicodeEncodeError`) — worked around with `PYTHONIOENCODING=utf-8`. Verified with two tracked runs compared side by side in the MLflow UI.

**Airflow orchestration and the promotion gate.** `pipeline.py` (discrete steps), `promotion.py` (the gate), the Airflow Docker Compose stack, `training_dag.py`. This milestone absorbed the bulk of the infrastructure debugging (Section 20.1): the Windows `uvicorn` socket bug, MLflow's DNS-rebinding Host-header rejection of `host.docker.internal` requests, and the local-artifact-store-isn't-reachable-from-a-container problem. Also hit a genuine Airflow gotcha: naming a TaskFlow parameter `run_id` collided with Airflow's own reserved context variable of the same name, breaking the DAG's signature at parse time — renamed to `mlflow_run_id`. Verified via three actual DAG triggers (not direct function calls): a bootstrap promotion, a deliberately-crippled challenger correctly rejected, and a genuinely-better challenger correctly promoted.

**Serving.** `src/serving/api.py`, `docker/serving/Dockerfile`, consolidation of the previously Airflow-only compose file into one shared-network `docker/docker-compose.yaml`. Required adding `feature_columns.json` artifact logging to training (a change to the orchestration milestone's `pipeline.py`), which meant resetting the registry to get a Production version that actually had the artifact. Verified with 5 integration tests and a live promotion-takes-effect test against the running container.

**Drift detection and the closed loop.** `src/drift/detector.py`, `simulate_drift.py`, `drift_check_dag.py`. Required adding a second artifact (`training_distribution.json`), another registry reset. Discovered and fixed the deterministic-retrain bug (Section 9.4/20.2) that would otherwise have made the closed loop unable to ever demonstrate a genuine drift-triggered promotion. Verified end-to-end through the real DAGs: normal traffic correctly skipped retraining, drifted traffic correctly triggered it, the resulting model was promoted, and the serving container picked it up automatically.

**CI/CD and the dashboard.** `.github/workflows/ci.yml`, `tests/conftest.py`'s hermetic fixture, `tests/test_drift_detector.py`, `dashboard/app.py`, latency instrumentation, drift-check MLflow logging. Discovered (via the new hermetic tests, which is exactly what they're for) a real bug: `MlflowClient` version objects don't always return `.version` as a string, which broke Pydantic response validation — fixed by explicit `str()` casts in both `api.py` and `detector.py`. The final end-to-end demo (Section 16) surfaced the PSI window-size noise issue (Section 20.3), fixed live during the same session.

**A process note, not a technical one:** partway through building the closed loop and the CI/CD milestone, it was discovered that the git commits for the orchestration and serving milestones had never actually been pushed — only the tracked-training milestone and (confusingly) a commit containing the closed-loop milestone's files existed on the remote, meaning the pushed `training_dag.py` referenced a `promotion.py` module that didn't exist in the repository. This was caught, diagnosed, and fixed with a catch-up commit; final commit history is `MLflow` → `drift trigger` → `Airflow promotion gate and Dockerized FastAPI serving` → `CI/CD pipeline and monitoring dashboard` — chronologically out of order on GitHub relative to when the code was actually written, but the final tree state is correct and was explicitly re-verified.

## 13. Operational Guide

### 13.1 Prerequisites

- Python 3.11, Docker Desktop (with Linux containers), the raw `Base.csv` placed at `Bank Account Fraud Dataset Suite (NeurIPS 2022)/Base.csv` (not included in the repo).

### 13.2 Bringing up the full stack

```bash
# 1. Python env
python -m venv .venv && ./.venv/Scripts/activate && pip install -r requirements.txt

# 2. MLflow tracking server (host process)
mkdir -p mlflow_store/artifacts
mlflow server --host 127.0.0.1 --port 5000 --workers 1 --serve-artifacts \
  --artifacts-destination ./mlflow_store/artifacts \
  --backend-store-uri sqlite:///mlflow_store/mlflow.db \
  --allowed-hosts "localhost:5000,127.0.0.1:5000,host.docker.internal:5000"

# 3. Postgres + Airflow + serving, one Docker network
cd docker
docker compose build
docker compose up -d postgres
docker compose up airflow-init
docker compose up -d airflow-webserver airflow-scheduler serving

# 4. Dashboard
cd ..
streamlit run dashboard/app.py
```

| Service | URL |
|---|---|
| MLflow UI | http://127.0.0.1:5000 |
| Airflow UI | http://localhost:8080 (`admin` / `admin`) |
| Serving API | http://localhost:8000 |
| Dashboard | http://localhost:8501 |

### 13.3 Common operations

- **Trigger training manually:** `docker compose exec airflow-scheduler airflow dags trigger training_dag`, optionally with `--conf '{"n_estimators": N, "max_depth": N, "learning_rate": F}'`.
- **Trigger a drift check manually:** `docker compose exec airflow-scheduler airflow dags trigger drift_check_dag`.
- **Simulate traffic:** `python -m src.drift.simulate_drift --n 300 [--drifted]`.
- **Force an immediate model reload in serving:** `curl -X POST http://localhost:8000/reload-model`.
- **Run tests:** `pytest tests/ -v` (no live services required — see Section 14.1).

### 13.4 Known environment quirks (Windows-specific)

- MLflow's console output includes an emoji that crashes on `cp1252`; set `PYTHONIOENCODING=utf-8` if training scripts crash *after* apparently completing successfully.
- Git Bash mangles absolute Unix-style paths passed to `docker compose exec` (e.g. turns `/opt/airflow/...` into `C:/Program Files/Git/opt/airflow/...`); prefix such commands with `MSYS_NO_PATHCONV=1`.

## 14. Testing and Validation

### 14.1 Test suite design: fully hermetic

`pytest tests/ -v` runs 16 tests with **no live MLflow server, no running Airflow, and none of the real (200MB, gitignored) dataset** — verified explicitly by running the suite with `MLFLOW_TRACKING_URI` and `PREDICTION_LOG_DB` unset. This is load-bearing for CI (Section 4.2's dataset-not-in-git constraint) and was a deliberate mid-project refactor (Section 11.10) away from an earlier version that assumed a live server.

The mechanism: `tests/conftest.py`'s session-scoped `mlflow_test_env` fixture points `MLFLOW_TRACKING_URI` at a throwaway `sqlite:///{tmp}/mlflow.db`, generates a synthetic 500-row dataset matching the real schema (same column names/types, same categorical value pools, random values via `numpy`'s `default_rng`), trains a tiny (`n_estimators=10, max_depth=2`) XGBoost model on it, logs the same two artifacts real training does (`feature_columns.json`, `training_distribution.json`), registers and promotes it — all before any test that needs a live model runs. `tests/test_promotion.py` uses a similar but independent pattern (its own function-scoped local SQLite store per test, via a `mlflow_local` fixture) rather than sharing `mlflow_test_env`, since each promotion test needs to construct an exact, controlled champion/challenger metric pair rather than reuse one shared registered model.

### 14.2 Test inventory

| File | Tests | What they cover |
|---|---|---|
| `tests/test_promotion.py` | 6 | The promotion gate itself: bootstrap promotes unconditionally, a challenger below the 2% margin is rejected, one that clears it is promoted (and the prior champion archived), the exact-boundary case promotes (`>=`, not `>`), a custom margin is respected, and an unregistered run_id raises |
| `tests/test_drift_detector.py` | 5 | Pure PSI math: dummy columns excluded from binning, bin proportions sum to 1, identical distributions score ~0 PSI, a shifted feature is correctly flagged, threshold logic respects a custom cutoff |
| `tests/test_serving_integration.py` | 5 | `/health` reports a loaded version; `/predict` returns a well-formed response; malformed input returns `422`; predictions are correctly written to SQLite; `/reload-model` reports the current version |

**Result:** 16/16 passing, confirmed both with a live host MLflow server present and explicitly with it absent (`env -u MLFLOW_TRACKING_URI -u PREDICTION_LOG_DB pytest`).

`test_promotion.py` was added after the fact, specifically because the original test suite — despite good hermetic-testing hygiene elsewhere — had zero coverage of `register_if_better()`, the single function that decides whether any model reaches production. That gap was caught in a self-review, not by CI or by any process built into the project; see Section 20.6 for the fuller account of why that's worth naming rather than quietly fixing.

### 14.3 What the hermetic tests actually caught

Not a hypothetical benefit — real bugs were found this way, more than once:
1. MLflow's file-store backend (`file:///...`) is in maintenance mode as of the installed MLflow version and now raises unless `MLFLOW_ALLOW_FILE_STORE=true` is set — discovered when the fixture first tried a `file:` tracking URI; fixed by switching the fixture to a `sqlite:///` backend instead.
2. `ModelVersion.version` from a fresh local registry was returned as a Python `int`, not `str`, which failed Pydantic validation on `PredictResponse.model_version: str` — fixed with explicit `str()` casts in `reload_if_new_version()` and `_load_production_training_distribution()`.
3. The *same* `int`-vs-`str` version issue reappeared independently when `test_promotion.py` was first written — its `_current_stage()` helper indexed a dict by `v.version` directly and got a `KeyError`, because a fresh local registry's version numbers came back as `int` there too. This wasn't a regression of bug 2 (a different file, a different code path — a test helper, not application code) — it's the same underlying MLflow behavior surfacing a second time in a second place, which is itself worth noting: a single fix in `api.py`/`detector.py` didn't make the *pattern* go away, only fixed the two spots already known about.

### 14.4 Manual / integration verification (against the real running stack)

Beyond the automated suite, every phase was independently verified against the actual running Docker stack and MLflow registry (not mocked), documented with exact run IDs and metric values throughout Section 16 and the project's build history — DAG runs triggered via the real Airflow CLI, registry state checked via the real MLflow REST API, serving responses checked via real `curl` requests against the running container.

### 14.5 What is covered now that wasn't, and what still isn't

**Closed since the initial build** (Section 20.6 has the fuller story of why these were found *after* the project was declared "done"):
- Load/stress testing of the serving API — `scripts/load_test.py`, Section 16.6. Found and led to fixing a real concurrent-write bug.
- Airflow DAG import validation in CI — `.github/workflows/ci.yml`'s `validate-dags` job (Section 19.2). Would have caught the `run_id`-collision bug from the orchestration milestone (Section 12) automatically, at commit time, instead of by manual DAG-trigger testing.
- The held-out test split (month 7) is now actually evaluated — `evaluate_test_set()` (Section 9, updated), logged per-run for reporting, deliberately not fed into the promotion decision.

**Still not covered:**
- No test of Airflow's `LocalExecutor` failure/retry behavior under a killed task.
- No CI job actually deploys or runs the trained pipeline against the real dataset — CI validates code correctness, DAG parseability, and Docker buildability, not a full real-data training run (a direct consequence of the dataset-not-in-git constraint, Section 4.2).
- The load test (Section 16.6) covers one endpoint (`/predict`) at one concurrency level (20) for one duration; no sustained soak test, no test of Airflow or MLflow under concurrent load, no test of what happens if `/predict` load and a training DAG run overlap.

## 15. Experimental Methodology

"Experiments" in this project are of two kinds: model-training runs (comparing hyperparameter choices via MLflow) and system-behavior verifications (proving the closed loop and other mechanisms work by triggering them and observing the outcome). Both follow the same basic method: **run the real thing, record the real output, don't report an expected or typical result as if it were observed.**

### 15.1 Model-comparison methodology

Each training run is one MLflow run: fixed data split (per that run's sample), a specific hyperparameter triple (`n_estimators`, `max_depth`, `learning_rate`), `val_pr_auc`/`val_roc_auc` computed on the held-out validation split (month 6, never used for training). Runs are compared directly via their logged metrics in the MLflow UI/API — no statistical significance testing was applied (a single validation split per run, not cross-validation); this is a deliberate scope simplification consistent with the project's non-goal of rigorous model quality (Section 3.3).

### 15.2 System-behavior verification methodology

For each closed-loop claim ("the gate rejects a worse model," "drift triggers retraining," "serving picks up a promotion with no redeploy"), the method was consistently: (1) put the system in a known starting state, (2) perform the real action via the real interface (an actual `airflow dags trigger`, an actual `curl` to `/predict`, never a direct Python function call standing in for the DAG), (3) query the real resulting state (the actual MLflow registry, the actual `/health` response, the actual Airflow task states), (4) report exactly what was observed, including when it wasn't what was expected (Section 16.2, 20.3).

### 15.3 The drift-simulation methodology

`simulate_drift.py`'s "normal" traffic is not synthetic — it's a genuine random sample of real BAF rows, sent through the real `/predict` endpoint, to genuinely test the drift detector against traffic that *should* match the training distribution (since it's literally drawn from the same population). "Drifted" traffic uses the same real sampling as its base, with five specific behavioral features (`velocity_6h`, `velocity_24h`, `velocity_4w`, `session_length_in_minutes`, `customer_age`) multiplicatively scaled before sending — chosen because they're plausible axes for genuine behavioral drift (a change in transaction velocity patterns, session behavior, or the age distribution of the customer base) rather than an arbitrary distortion.

## 16. Results and Observed Behavior

This section reports actual measured outcomes, in the order they were produced.

### 16.1 Tracked training (two compared runs)

| run | n_estimators | max_depth | learning_rate | val_roc_auc | val_pr_auc |
|---|---|---|---|---|---|
| `f73d0fa2...` | 250 | 4 | 0.08 | 0.8694 | 0.1402 |
| `7e17a641...` | 120 | 3 | 0.15 | 0.8757 | 0.1497 |

Both logged and registered correctly; visible side by side in the MLflow UI.

### 16.2 The promotion gate, via three real DAG triggers

| trigger | hyperparams | val_pr_auc | outcome |
|---|---|---|---|
| 1 (defaults) | 200/4/0.1 | 0.1430 | **Promoted** (bootstrap, no prior champion) → v1 Production |
| 2 (`--conf`, deliberately crippled) | 1/1/0.01 | 0.0307 | **Rejected** — v1 correctly held as Production |
| 3 (`--conf`, tuned stronger) | 1000/3/0.02 | 0.1509 | **Promoted** — beats v1 by >2% (threshold 0.1459) → v3 Production, v1 Archived |

All 5 DAG tasks (`ingest`, `validate`, `train`, `evaluate`, `register`) reported `success` on every run — including the rejection case, since a correct rejection is a successful `register` task outcome, not a failure.

### 16.3 Serving: integration tests and live promotion pickup

5/5 integration tests passed. Live promotion-takes-effect test: `serving` container running and serving v2; a new challenger (v9, val_pr_auc 0.1573, beating v2's 0.1509 by >2%) trained and promoted via `register_if_better`; `/health`, re-checked with no restart and no redeploy, reported v9. The background poller picked this up automatically within its 30-second interval in every test run — the manual `/reload-model` endpoint was never actually needed to observe the update.

### 16.4 The closed loop, via the real DAGs

**Normal traffic (150 requests):** `check_drift` → max PSI 0.155 (all features), below threshold → `drift_detected` gate evaluated `False` → `trigger_retraining` **skipped**.

**Drifted traffic (150 requests, mixed into the same window):** `check_drift` → 5 features exceeded PSI 0.2 (`customer_age` 1.75, `velocity_6h` 1.56, `velocity_24h` 2.39, `velocity_4w` 2.49, `session_length_in_minutes` 2.12) → `drift_detected` evaluated `True` → `trigger_retraining` fired `training_dag` automatically. The triggered run (on a freshly-drawn data sample) scored val_pr_auc 0.1549 vs. the prior champion's 0.1430, clearing the 2% margin — **promoted**. The `serving` container's background poller picked up the new version automatically; a live `/predict` request confirmed `model_version: 5` with zero manual redeploy steps.

### 16.5 The final end-to-end demo (drift climbing, dashboard-verified)

Starting from a clean prediction log, full stack up (MLflow, Postgres, Airflow, serving, dashboard):

**Drift score history**, read directly from the `drift-monitoring` MLflow experiment (exactly what the dashboard's drift panel plots):

| check | max_psi | drifted | n_samples | note |
|---|---|---|---|---|
| 1 | 0.204 | True | 200 | normal traffic; borderline PSI noise at the (then-default) 200-sample window |
| 2 | 0.208 | True | 200 | same noise, reproduced |
| 3 | 0.208 | True | 200 | same noise, reproduced a third time — this is what motivated the window-size fix |
| 4 | 0.216 | True | 300 | normal traffic, post-fix window size — still borderline on this run |
| 5 | **8.29** | True | 300 | deliberately drifted traffic — unambiguous |

**Registry timeline** over the same session:

| version | stage | val_pr_auc |
|---|---|---|
| 1 | Archived | 0.1430 |
| 5 | Archived | 0.1549 |
| 6 | **Production** | **0.2096** |
| 9 | unstaged | 0.1456 (correctly rejected — below v6's 2% margin) |
| 10 | unstaged | 0.1636 (correctly rejected — below v6's 2% margin) |

**Volume/latency** over the session's ~600 sent requests: average latency ~11ms, p99 ~18ms, all served by whichever version was current `Production` at request time — confirmed via direct SQL query against `data/predictions.db`, the same source the dashboard's volume/latency panels read.

**Interpretation:** this run is a more honest and more interesting result than a clean, noise-free demo would have been — it directly shows the gate doing its protective job (checks 1–3 triggered retrains from noise, not real drift, and every one of those challengers either got fairly promoted on its own merits or correctly rejected; nothing false-positive ever corrupted the registry), and check 5 shows the detector responding to unambiguous drift exactly as designed. See Section 20.3 for the full diagnosis of *why* checks 1–3 happened.

### 16.6 Load test: concurrent vs sequential latency

The latency figures in Section 16.5 (~11ms avg, ~18ms p99) came from `simulate_drift.py`'s traffic — real requests, but sent sequentially by a single client. That's a meaningfully different measurement from load, and presenting it as *the* latency number without that caveat would have been a real gap: this section exists because that gap was identified during a self-review of the project and closed with an actual load test, not because it was planned from the start.

`scripts/load_test.py` sends genuinely concurrent requests via a thread pool and measures wall-clock latency per request plus overall throughput. First run, 20 concurrent clients, 500 requests against the real running `serving` container:

| metric | value |
|---|---|
| requests | 500 |
| **failed** | **8** |
| avg latency | 659ms |
| p50 | 89ms |
| p95 | 3755ms |
| p99 | 5160ms |
| throughput | 29.3 req/s |

**The 8 failures were not noise.** `docker compose logs serving` showed exactly 8 occurrences of `sqlite3.OperationalError: database is locked` — SQLite's default rollback-journal mode serializes writers, and 20 concurrent `/predict` requests each trying to `INSERT` into `data/predictions.db` at once produced real, reproducible write contention. This was a genuine reliability bug that no prior test (sequential traffic, single-request integration tests) had ever exercised, because none of them wrote concurrently.

**Fix:** `_init_log_db()`/`_log_prediction()` now connect via a shared `_connect()` helper that sets `PRAGMA journal_mode=WAL` (allows one writer and many readers to proceed concurrently, instead of a writer blocking everyone) and `PRAGMA busy_timeout=5000` (any remaining contention retries for up to 5s before failing, instead of failing immediately). Re-ran the identical load test after rebuilding and restarting the `serving` container:

| metric | before (rollback journal) | after (WAL) |
|---|---|---|
| failed | 8 | **0** |
| avg latency | 659ms | 496ms |
| p50 | 89ms | 228ms |
| p95 | 3755ms | 2283ms |
| p99 | 5160ms | 3445ms |
| throughput | 29.3 req/s | 39.2 req/s |

**What the fix did and didn't solve.** The reliability bug (dropped requests) is fully resolved — 0 failures, verified. Throughput and per-request latency under load both improved but remain far worse than the sequential numbers. That gap is a *separate*, still-open limitation: the serving container runs a single `uvicorn` worker process with no `--workers` flag (`docker/serving/Dockerfile`), and FastAPI's synchronous `/predict` handler runs in that one process's thread pool — 20 concurrent CPU-bound XGBoost inference calls genuinely queue for that pool. Raising `--workers` or moving to an async inference path would address this; neither was implemented (Section 21.4).

### 16.7 Summary table, all five milestones

| Milestone | Verified | Result |
|---|---|---|
| Tracked training | Two compared runs | Both logged, both registered |
| Airflow + gate | Bootstrap / reject / promote, via real DAG triggers | 0.1430 promoted → 0.0307 rejected → 0.1509 promoted, v1 archived |
| Serving | 5 integration tests; live promotion pickup | 5/5 passed; v2→v9 picked up automatically in ≤30s |
| Drift + auto-retrain | No false trigger on normal traffic; real trigger on drifted traffic; loop closes to serving | Normal: PSI 0.155, skipped. Drifted: PSI up to 2.49, retrain triggered, 0.1549 promoted, serving updated automatically |
| CI/CD + dashboard | 10 hermetic tests; full demo with dashboard-sourced data | 10/10 passed with no live infra; PSI climbed 0.204→8.29 across 5 checks; 2 promotions, 2 correct rejections |
| Post-launch hardening | Promotion-gate unit tests added; CI validates DAG imports; held-out test-set metric wired in; concurrent load test run | 16/16 total tests passing (6 new promotion tests); load test found and fixed a real SQLite write-contention bug (8/500 failures → 0/500) |

## 17. Security, Reliability, and Failure Handling

### 17.1 Security posture (explicitly minimal — this is a local demo project)

- **Airflow:** default `admin`/`admin` credentials, `basic_auth` backend. Not acceptable outside a local demo context.
- **MLflow:** no authentication at all on the tracking server; `--allowed-hosts` protects against DNS-rebinding attacks specifically (a real, if narrow, threat model) but does not gate *who* can read/write experiments and models — anyone who can reach port 5000 has full registry access, including the ability to promote/archive models.
- **Serving API:** no authentication on `/predict`, `/reload-model`, or `/health`. Pydantic validation on `/predict` provides input-shape safety (rejects malformed requests) but not access control.
- **No secrets management:** the project has no credentials to manage (no cloud provider, no external API keys), which sidesteps rather than solves this category.

None of this was hardened because the project's scope is a local, single-user development/demo environment, not a multi-tenant or internet-facing deployment — but it's worth being explicit that "no auth anywhere" is a real gap, not an oversight to gloss over, if this were to move toward any shared or exposed deployment.

### 17.2 Reliability mechanisms actually built

- **`validate_data()`** fails loudly (raises, not logs-and-continues) on schema/row-count/fraud-rate anomalies, so a corrupted ingest can never silently reach training.
- **The promotion gate** is the system's core reliability mechanism: it makes bad retraining outcomes cheap. A false-positive drift trigger, or a retrain that happens to draw an unlucky data sample, costs one wasted training run — it can never regress the serving model, because promotion requires clearing a real bar. This was directly observed in Section 16.5 (checks 1–4 all triggered retrains; only one, which happened to be genuinely better, was promoted).
- **The serving background poller's exception handling** (`_background_poll_loop`) swallows exceptions from `reload_if_new_version()` so a transient MLflow-connectivity blip can't kill the polling thread and silently freeze the API on a stale model forever.
- **`_log_drift_check()`'s exception handling** similarly ensures a monitoring-log failure (e.g. MLflow temporarily unreachable) can never block the actual drift decision that `drift_check_dag` needs to make.
- **`predictions.db` runs in WAL mode** (`PRAGMA journal_mode=WAL`, `PRAGMA busy_timeout=5000`) specifically so concurrent `/predict` writers don't serialize against each other — added after a load test (Section 16.6) found this failing 8/500 requests with `database is locked` under 20 concurrent clients using SQLite's default rollback-journal mode. Re-verified at 0/500 failures under identical load after the fix.

### 17.3 Failure modes not handled

- **No retries** anywhere in the Airflow DAGs (`DEFAULT_ARGS = {"retries": 0}` explicitly) — a transient failure (e.g. a momentary MLflow connection drop during `train_model`) fails the whole DAG run rather than retrying. Acceptable for a project where DAG runs are manually observed during testing; not acceptable for unattended production scheduling.
- **No dead-letter or alerting path** if `drift_check_dag` itself fails (as opposed to succeeding with `drifted=False`) — a broken drift check would silently stop protecting against drift with no notification.
- **No handling for the serving API's SQLite log growing unbounded** — `predictions.db` has no retention/rotation policy; it grows indefinitely.
- **Concurrent-write *failures* are fixed (WAL mode, above), but concurrent-write *scale* is not** — WAL resolves lock contention for the load tested here (single serving instance, 20 concurrent clients), but SQLite is still a single-file store; a genuinely scaled-out serving deployment (multiple replicas writing to the same log) would need a real database, not just a better SQLite journal mode.

## 18. Performance, Scalability, and Cost

### 18.1 Observed performance

- **Training:** a full `ingest → validate → train → evaluate → register` DAG run completes in roughly 30–50 seconds end to end on the 250k-row sample (observed directly in Airflow task timestamps throughout Section 16 — e.g. one closed-loop run: `ingest` ~9s, `validate` ~3s, `train` ~25s, `evaluate` ~3s, `register` ~2s).
- **Serving latency, sequential:** average ~11ms, p99 ~18ms per `/predict` request (Section 16.5), measured end-to-end inside the FastAPI handler (`time.perf_counter()` around preprocessing + inference), one client, no concurrency.
- **Serving latency, 20 concurrent clients (Section 16.6):** average ~496ms, p99 ~3.4s, throughput ~39 req/s, 0 failures after the WAL-mode fix (8/500 failed before it). The gap between the sequential and concurrent numbers is real and explained in Section 16.6 — a single `uvicorn` worker process, not a remaining bug.
- **Drift check:** a `check_drift` task completes in 2–5 seconds (reading up to 300 SQLite rows, downloading one MLflow artifact, computing PSI over ~25 numeric features).

### 18.2 Scalability — mostly not addressed, one concurrency bug found and fixed

This project runs entirely on a single machine with single-instance services throughout. Load testing (Section 16.6) went further than "not tested" for one specific dimension — concurrent writes to the prediction log — found a real bug there, and fixed it (SQLite WAL mode). Beyond that specific fix, no component was tested or designed for horizontal scale:
- Serving is one FastAPI process, one `uvicorn` worker; no load balancer, no replica set, no autoscaling. The single-worker throughput ceiling observed in Section 16.6 (p99 latency ~190x worse under 20 concurrent clients than sequential) is a direct, measured consequence of this, not a hypothetical one.
- The SQLite prediction log's *lock-contention* failure mode is fixed (WAL mode); its *single-file, single-machine* nature is not — it would not survive a genuinely multi-replica serving deployment without moving to a real database.
- Airflow's `LocalExecutor` runs all tasks in the scheduler process — no distributed task execution.
- Kubernetes deployment was considered as a possible extension and was not attempted.

### 18.3 Cost

Zero infrastructure cost — everything runs locally (Docker Desktop + host processes) with no cloud resources provisioned. This was never a design goal being optimized for; it's simply a consequence of the project never being deployed to the cloud.

## 19. Deployment, Monitoring, and Maintenance

### 19.1 Deployment model

There is no "deployment" in the cloud-release sense — the project's entire lifecycle runs locally. "Deploying" a new model means the promotion gate marking a new registry version `Production`; the serving API's background poller is the mechanism that makes that take effect, which is the closest thing this project has to a deploy pipeline, and it was deliberately built to require zero manual action (Section 6.2, verified repeatedly in Section 16).

### 19.2 CI/CD

`.github/workflows/ci.yml` has three jobs:

1. **`test`** — on every push and pull request, installs `requirements.txt` and runs `pytest tests/ -v` (the hermetic suite, Section 14.1 — runs identically on a GitHub-hosted runner as locally, with no dataset and no live services).
2. **`validate-dags`** — on every push and pull request, installs only `apache-airflow==2.9.3` (pinned to match `docker/airflow/Dockerfile`'s base image, with Airflow's own constraints file) — deliberately *not* the full `requirements.txt` — and runs `python dags/training_dag.py` / `python dags/drift_check_dag.py` directly. Both DAG files import every `src/*` dependency lazily inside each task's function body rather than at module top level specifically so this check stays lightweight: parsing/instantiating the DAG only needs Airflow itself, not xgboost, mlflow, or any training/serving dependency. Verified against the real Airflow container that this check would have caught the actual `run_id`-collision bug from the orchestration milestone (Section 12) — re-running the buggy version of `training_dag.py` this way reproduces the exact historical `ValueError` at DAG-definition time.
3. **`build-serving-image`** — on push to `main` only, after `test` passes, builds the serving Docker image (`docker build -f docker/serving/Dockerfile`) to catch a broken Dockerfile or dependency conflict before it would affect a deploy.

No image push to a registry, no automatic redeploy — CI validates buildability, DAG parseability, and test correctness; it does not itself deploy anything.

### 19.3 Monitoring

The Streamlit dashboard (`dashboard/app.py`) is the monitoring surface: prediction volume (hourly), latency (avg/p99, hourly), drift score history (from the `drift-monitoring` MLflow experiment, with the 0.2 threshold drawn as a reference line), and the model registry's version/promotion timeline. All four panels are cached for 15 seconds (`st.cache_data(ttl=15)`) and read directly from the same data stores every other component writes to — there is no separate metrics-aggregation pipeline to fall out of sync.

### 19.4 Maintenance considerations

- **Registry growth:** every training run (including drift-triggered ones that don't get promoted) creates a permanent, never-deleted model version. Over a long deployment this would need a retention/cleanup policy that doesn't currently exist.
- **Prediction log growth:** `data/predictions.db` has no rotation (Section 17.3) — an operational task that would be needed before any long-running deployment.
- **MLflow stages deprecation:** the registry API this project depends on (Section 11.6) is deprecated upstream; a future MLflow upgrade could require a migration to the alias-based API.

## 20. Interpretation and Lessons Learned

This section reconstructs the actual debugging narrative behind the results in Section 16 — not because the bugs are interesting in themselves, but because each one changed a design decision documented elsewhere in this file, and the reasoning is easier to evaluate with the failure that motivated it attached.

### 20.1 Windows/Docker infrastructure friction (Airflow orchestration)

Three distinct, compounding issues surfaced in sequence when first standing up Airflow-talks-to-MLflow:

1. **MLflow's multi-worker `uvicorn` mode crashed intermittently on Windows** (`OSError: [WinError 10022]: An invalid argument was supplied`), traced to how Windows handles socket sharing across `multiprocessing`-spawned worker processes differently from POSIX (`SO_REUSEPORT`-style fd passing isn't equivalent). Sometimes one worker would still bind successfully, making the server *appear* to work while actually running in a degraded, unreliable multi-process state — the failure was silent enough that it wasn't caught until a later request against the "working" server also failed. **Fix:** `--workers 1`.
2. **MLflow's DNS-rebinding protection (`--allowed-hosts`) rejected requests from the Airflow container**, since it sends `Host: host.docker.internal`, which isn't in MLflow's default allowed-hosts list (`localhost` + private IP ranges by default). Returned a generic `403 Invalid Host header` with no immediately obvious connection to the actual cause. **Fix:** explicit `--allowed-hosts` including `host.docker.internal:5000` — critically, **with the port included**; an earlier attempt using bare hostnames without ports still failed, because the middleware's allowlist matching (once you override the default) requires exact `host:port` entries.
3. **The local-filesystem artifact store wasn't reachable from a separate container**, since a `file://`-scheme artifact root is accessed *directly by the client*, not proxied through the server — the Airflow container tried to `os.makedirs()` a Windows path (`C:\Users\...`) as if it were a Linux path, producing `PermissionError: [Errno 13] Permission denied: '/C:'`. **Fix:** `--serve-artifacts --artifacts-destination` instead of `--default-artifact-root`, which proxies artifact I/O through the tracking server's own REST API rather than requiring direct filesystem access from every client.

**The general lesson**, stated once here rather than three times: a "same machine, dev-mode" configuration accumulates implicit assumptions (only localhost will ever connect, the filesystem is shared, one process is enough) that don't survive the first real multi-consumer, multi-process, cross-container deployment — and each of those assumptions failed independently and had to be diagnosed one at a time rather than as a single root cause.

### 20.2 The deterministic-retrain bug (closed-loop testing)

The closed loop's very first real test (drift detected → `training_dag` auto-triggered → check whether promotion resulted) produced a *tie*: the auto-triggered challenger scored the exact same `val_pr_auc` as the existing champion. Root cause: `ingest_data()`'s default `random_state=42` meant every retraining run — auto-triggered or not — deterministically reproduced the identical 250k-row sample from the same 1M-row source, and the model's own `random_state=42` was also fixed. Same data, same model config, same model, every time. A gate that requires beating the champion by a margin can, by construction, never accept a tied challenger — so the closed loop, as originally built, could detect drift and trigger retraining correctly, but could *never actually complete a drift-triggered promotion*, which is the single most important claim the whole project is supposed to demonstrate.

This was not a cosmetic bug — it directly threatened the core deliverable. Diagnosed by checking the exact metric values of two consecutive triggered runs and noticing they matched to the same six decimal places (impossible by chance with real sampling variance). **Fix:** `training_dag`'s `ingest` task explicitly calls `ingest_data(random_state=None)`, deliberately overriding the module's default fixed seed, so pandas draws from the global RNG and each retrain sees a genuinely different stratified sample. Re-tested immediately after the fix and observed a real, different, and in that case better-scoring challenger, which did clear the promotion margin (Section 16.4).

### 20.3 The PSI window-size noise (final demo)

The Section 16.5 final demo's first several "normal traffic" checks came back `drifted: True` with `velocity_4w`/`velocity_24h` PSI values of 0.204–0.216 — just over the 0.2 threshold, on traffic that was, by construction (drawn from the real dataset with no injected shift), not supposed to be drifted at all. The first hypothesis (a bug in the PSI computation itself) was ruled out by unit-testing the pure `compute_drift()` function directly against known synthetic distributions (Section 14.2's `test_identical_distribution_yields_near_zero_psi` — which passed, confirming the math itself was correct).

The actual cause: 10 quantile bins over a 200-sample live window is roughly 20 points per bin on average, and *far fewer* than that near the tails of a skewed distribution — the BAF velocity features (`velocity_6h/24h/4w`) are heavy-tailed, spanning roughly 0–20,000 with most mass concentrated well below the max. At that bin population, ordinary multinomial sampling variance is large enough to occasionally push a bin's observed proportion far enough from its expected proportion to cross PSI 0.2, even when the underlying population hadn't moved at all. This was confirmed directly: re-running `check_for_drift()` with `window_size=600` (using all 300 available samples instead of only the most recent 200) on the *exact same* traffic dropped `velocity_4w`'s PSI from 0.208 to 0.178 — same data, more of it, lower apparent drift, which is the signature of a sample-size artifact rather than a real signal.

**Fix:** `DEFAULT_WINDOW_SIZE` raised from 200 to 300, verified to settle the same feature's noise below threshold. **What this doesn't fix:** it's a threshold shift, not an elimination of the underlying phenomenon — a sufficiently unlucky 300-sample window could still cross 0.2 by chance, just less often. This is documented as an open limitation (Section 21), not a solved problem, because it isn't actually solved, only made less frequent.

### 20.4 What the hermetic-test refactor caught

Covered technically in Section 14.3; the interpretive point here is that both bugs found this way (`file:` store deprecation, `.version` returning `int`) were **pre-existing** in code that had already "worked" in every manual test up to that point — they surfaced specifically *because* the new fixture exercised a code path (a fresh, from-scratch local registry) that the manual testing flow, which always ran against an already-long-running, already-string-typed server, had never actually hit. This is the concrete argument for hermetic tests in this project specifically: they don't just make CI possible, they exercise different, previously-untested code paths than manual testing does, and found real bugs by doing so.

### 20.5 The git-history gap

Documented factually in Section 12's final paragraph. The interpretive lesson: verifying "is the deliverable actually complete" required checking git history against the working directory, not just checking the working directory — the working directory always looked complete (every file existed locally through every phase), which is exactly why the gap between "built" and "actually pushed to the remote" went unnoticed until it was explicitly checked. A file existing on disk and a file existing in the repository a collaborator would clone are not the same claim, and only one of them was verified by default.

### 20.6 The post-launch hardening pass: a self-review found four real gaps after the project was "done"

Once the closed loop, the dashboard, and CI were all working and pushed, the project was deliberately reviewed a second time from a colder, more skeptical angle: not "does this work," but "what would a rigorous outside engineer notice in the first few minutes of opening this repository." That review surfaced four concrete, specific gaps, none of them hypothetical:

1. **`register_if_better()` — the promotion gate — had zero test coverage**, confirmed by grepping `tests/` for any reference to it and finding none. Everything *around* the gate was tested (the drift math that feeds a retraining decision, the API that serves whatever the gate promotes); the gate's own decision logic was not. Fixed with `tests/test_promotion.py` (Section 14.2) — 6 tests covering the bootstrap case, rejection below the margin, promotion above it, the exact boundary, a custom margin, and the error path.
2. **The held-out test split was computed but never evaluated.** `temporal_split()` had produced a `test` split (month 7) since the very first training script; nothing ever read it. Fixed by adding `evaluate_test_set()` and wiring it into `training_dag` as a reporting-only step (Section 9), explicitly not fed into `register_if_better()`.
3. **CI never validated the DAGs themselves**, only the training/serving/drift *logic* those DAGs call into. The exact bug class that broke `training_dag.py` during the orchestration milestone (a TaskFlow parameter colliding with an Airflow reserved word, Section 12) would not have been caught by the CI that existed at the time — it was only caught because a human manually triggered the DAG and read the traceback. Fixed by adding a `validate-dags` CI job (Section 19.2) that runs each DAG file directly with just Airflow installed (no training/serving dependencies needed, since both DAGs import `src/*` lazily inside task bodies, not at module level) — verified against the actual Airflow container to confirm it would have caught the real historical bug, not just a synthetic one.
4. **The headline serving-latency number was sequential, not concurrent, and wasn't labeled as such.** ~11ms avg / ~18ms p99 is a true number, but reporting it as "the" latency without qualifying "single sequential client" risked overstating what had actually been measured. Fixed by writing an actual load test (`scripts/load_test.py`, Section 16.6) — which, in the process of being run for the first time, found a genuine, previously-unknown reliability bug (SQLite lock contention under concurrent writes, 8/500 requests failing), which was then also fixed (WAL mode) and re-verified (0/500 failures).

**Why this section exists at all, rather than just quietly updating the numbers above:** items 1–3 are a specific, recognizable failure pattern — testing and validating the parts of a system that are easy to test, while the part that makes the single most consequential decision in the whole pipeline (what becomes production) went unchecked. That pattern is more informative than any individual bug it produced. Item 4 is a related but distinct pattern: reporting a real, honestly-measured number without the caveat that made it only partially representative, which is a subtler failure than reporting a wrong number, and arguably more likely to slip past both the author and a reviewer, since nothing about it looks incorrect in isolation. Both patterns are worth being able to name and recognize, independent of this specific project — which is the actual reason a "why didn't this get caught the first time" section belongs in a design document at all, rather than treating a hardening pass as if it had been the plan from the start.

## 21. Limitations, Known Issues, Technical Debt, and Future Work

Every limitation encountered or knowingly accepted during the build, in one place.

### 21.1 Model quality

- No hyperparameter search; hyperparameters were hand-picked, including some picked specifically to force a demonstrable promotion outcome rather than to maximize genuine model quality (Section 9.1, 11.3).
- Single train/val split per run, no cross-validation — metric comparisons between runs don't account for evaluation variance.
- The model has never been evaluated against the held-out month-7 test split at all; it exists in the pipeline but nothing currently reads it.

### 21.2 Drift detection

- **Categorical features are not monitored for drift at all** (Section 9.3) — only numeric raw features. A fraud pattern that shifted purely in categorical space (e.g. a sudden shift toward one `payment_type`) would be invisible to the current detector.
- **PSI at moderate window sizes produces real false positives on heavy-tailed features** (Section 20.3) — mitigated by raising the window to 300, not eliminated. A more robust fix (not implemented) would be requiring drift to persist across 2+ consecutive hourly checks before triggering retraining, or widening bins specifically for high-variance features.
- The any-single-feature trigger rule (Section 9.3) is deliberately sensitive; it was never compared against a multi-feature-consensus alternative to see which produces fewer false positives in practice at the chosen window size.
- No drift monitoring on the model's *output* distribution (prediction/probability drift) — only input-feature drift.

### 21.3 Registry and promotion

- **The MLflow stages API used throughout is deprecated upstream** (Section 11.6) and will require a migration to the alias-based API at some future MLflow version.
- The promotion gate's 2% margin was chosen by inspection of this dataset/model's observed run-to-run variance, not derived from a formal statistical test (e.g. a paired significance test between challenger and champion metrics) — a more rigorous gate would compute a confidence interval on the metric difference rather than a fixed percentage.
- No registry cleanup/retention policy — every trained version (promoted or not) is kept forever (Section 19.4).
- The bootstrap case (Section 9.2) promotes the very first model unconditionally, with no quality floor.

### 21.4 Serving and reliability

- No authentication on any endpoint (Section 17.1).
- No rate limiting, no request batching, no horizontal scaling, no multiple `uvicorn` workers — single-instance, single-process serving only. Load testing (Section 16.6) measured the real cost of this directly: p99 latency degrades from ~18ms sequential to ~3.4s at 20 concurrent clients.
- No retries anywhere in the Airflow DAGs (Section 17.3) — a transient failure fails the whole DAG run.
- No alerting if `drift_check_dag` itself fails (as distinct from succeeding with `drifted=False`).
- `data/predictions.db` (SQLite) has no rotation/retention policy. Its concurrent-write *lock-contention* bug was found and fixed (WAL mode, Section 16.6/17.2) — but it remains a single-file store, not a real answer for a multi-replica serving deployment.

### 21.5 Reasonable extensions, not attempted

- Kubernetes deployment (kind/minikube locally, or a cloud provider) — the project runs entirely on Docker Compose.
- Any cloud provider component — the project is 100% local.
- Grafana/Prometheus-style monitoring — a Streamlit dashboard was built instead, reading directly from SQLite/MLflow rather than a metrics time-series database.

### 21.6 Process debt

- Commit history is chronologically inconsistent with actual build order (Section 12, 20.5) — cosmetic, but a real artifact of how the project was actually developed and worth being upfront about rather than silently rewriting history to hide it.

## 22. Conclusion

This project set out to prove something narrower and harder than "I can train a fraud model": that a system can sense its own staleness and correct itself without a human in the loop, while a gate protects it from correcting itself into something worse. That specific claim — drift detected, retraining triggered automatically, a genuinely better model promoted, serving updated with zero manual steps — was demonstrated multiple times against the real running system (Sections 16.4, 16.5), not asserted from a design doc.

The build process surfaced real, non-obvious engineering problems that a description-only version of this project would never have encountered: a Windows-specific socket bug that silently degraded a "working" server, a Host-header security feature that had to be understood rather than bypassed, an artifact-storage assumption that broke the moment a second container needed the same files, a deterministic-sampling bug that would have made the entire closed-loop story impossible to actually demonstrate, and a statistical sampling-noise effect in the drift detector itself that only appeared once the system was pushed hard enough to hit it. Every one of those is documented here with its root cause and its fix, not just its resolution — because the reasoning behind each fix is the part of this project that doesn't survive in the code alone.

What remains genuinely unfinished is documented in Section 21 without hedging: categorical drift isn't monitored, the registry has no retention policy, nothing is authenticated, and the PSI false-positive rate at the current window size is reduced, not eliminated. None of that was hidden or discovered by someone else after the fact — it's the honest edge of what a project scoped and built the way this one was can responsibly claim to have proven.
