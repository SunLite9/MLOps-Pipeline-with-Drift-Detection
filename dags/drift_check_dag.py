"""Drift check DAG: reads recent prediction logs, computes PSI drift against
the currently-Production model's training distribution, and — if drift
exceeds the threshold — automatically triggers training_dag. This automatic
trigger is the closed loop that is the centerpiece of the whole project:
drift in, retraining out, with zero manual steps.
"""

from datetime import datetime

from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

DEFAULT_ARGS = {"owner": "mlops", "retries": 0}


@dag(
    dag_id="drift_check_dag",
    description="Checks live inference traffic for drift against the Production model's training distribution; triggers retraining if drift exceeds the threshold.",
    schedule="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["drift"],
)
def drift_check_dag():
    @task
    def check_drift() -> dict:
        from src.drift.detector import check_for_drift

        result = check_for_drift()
        print(result)
        return result

    @task.short_circuit
    def drift_detected(result: dict) -> bool:
        return result["drifted"]

    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="training_dag",
    )

    drift_detected(check_drift()) >> trigger_retraining


drift_check_dag()
