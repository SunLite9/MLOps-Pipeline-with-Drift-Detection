"""Training DAG: ingest -> validate -> train -> evaluate -> register_if_better.

Each task wraps a discrete function from src/training/pipeline.py and
src/training/promotion.py. Only small, JSON-serializable values (paths,
run ids, metric floats) cross tasks via XCom; the actual data splits live
as parquet files on the shared filesystem mounted into the Airflow
containers.
"""

from datetime import datetime

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

DEFAULT_ARGS = {"owner": "mlops", "retries": 0}


@dag(
    dag_id="training_dag",
    description="Ingest, validate, train, evaluate, and conditionally promote a challenger fraud model.",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["training"],
)
def training_dag():
    @task
    def ingest():
        from src.training.pipeline import ingest_data

        # No fixed random_state: each DAG run draws a fresh stratified sample
        # rather than reusing the exact same rows forever, so repeated
        # retrains (e.g. from drift_check_dag) see genuinely new data instead
        # of deterministically tying with the existing champion every time.
        return ingest_data(random_state=None)

    @task
    def validate(paths: dict):
        from src.training.pipeline import validate_data

        return validate_data(paths)

    @task
    def train(paths: dict):
        from src.training.pipeline import train_model

        # Allows `airflow dags trigger training_dag --conf '{"n_estimators": 1, ...}'`
        # to force a deliberately weak challenger, for testing the promotion gate.
        conf = get_current_context()["dag_run"].conf or {}
        hyperparams = {
            k: conf[k]
            for k in ("n_estimators", "max_depth", "learning_rate")
            if k in conf
        }
        return train_model(paths, **hyperparams)

    @task
    def evaluate(mlflow_run_id: str, paths: dict):
        from src.training.pipeline import evaluate_model

        return evaluate_model(mlflow_run_id, paths)

    @task
    def evaluate_test(mlflow_run_id: str, paths: dict):
        from src.training.pipeline import evaluate_test_set

        # Reporting only — never feeds the promotion decision.
        return evaluate_test_set(mlflow_run_id, paths)

    @task
    def register(mlflow_run_id: str, metric: float):
        from src.training.promotion import register_if_better

        decision = register_if_better(mlflow_run_id, metric)
        print(decision)
        return decision

    ingested_paths = ingest()
    validated_paths = validate(ingested_paths)
    mlflow_run_id = train(validated_paths)
    metric = evaluate(mlflow_run_id, validated_paths)
    register(mlflow_run_id, metric)


training_dag()
