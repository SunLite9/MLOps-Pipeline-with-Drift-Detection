"""Champion/challenger promotion gate for the model registry.

Promotion rule: a challenger is promoted to the "Production" stage only if
its primary metric (val_pr_auc) beats the current Production model's
val_pr_auc by at least MIN_RELATIVE_IMPROVEMENT (2%) relative improvement.
If no Production model exists yet, the first challenger is promoted
unconditionally (bootstrap case).

PR-AUC, not ROC-AUC, is the gating metric — see pipeline.evaluate_model for
why (heavy class imbalance in fraud data).

A small positive margin (rather than "any improvement promotes") avoids
promotion churn from run-to-run noise that doesn't reflect a real gain.
"""

import os

from mlflow.tracking import MlflowClient

MODEL_NAME = "fraud-model"
METRIC_NAME = "val_pr_auc"
MIN_RELATIVE_IMPROVEMENT = 0.02


def _tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def _get_production_metric(client: MlflowClient, model_name: str, metric_name: str):
    """Returns (metric_value, version) for the current Production model, or
    (None, None) if there isn't one yet."""
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not prod_versions:
        return None, None
    prod_version = prod_versions[0]
    run = client.get_run(prod_version.run_id)
    return run.data.metrics.get(metric_name), prod_version


def register_if_better(
    run_id: str,
    new_metric: float,
    model_name: str = MODEL_NAME,
    metric_name: str = METRIC_NAME,
    min_relative_improvement: float = MIN_RELATIVE_IMPROVEMENT,
) -> dict:
    """The promotion gate. `run_id` must already have a registered model
    version (train_model() registers it, unstaged). Looks up the current
    Production model's metric, compares, and promotes the challenger to
    Production (archiving the previous Production version) only if it wins.

    Returns a dict describing the decision, for logging/assertions.
    """
    client = MlflowClient(tracking_uri=_tracking_uri())

    challenger_versions = client.search_model_versions(f"run_id='{run_id}'")
    if not challenger_versions:
        raise ValueError(f"No registered model version found for run_id={run_id}")
    challenger_version = challenger_versions[0]

    champion_metric, champion_version = _get_production_metric(client, model_name, metric_name)

    if champion_metric is None:
        promoted = True
        reason = "no existing Production model — bootstrapping challenger as first Production version"
    else:
        threshold = champion_metric * (1 + min_relative_improvement)
        promoted = new_metric >= threshold
        reason = (
            f"challenger {metric_name}={new_metric:.4f} "
            f"{'>=' if promoted else '<'} threshold={threshold:.4f} "
            f"(champion={champion_metric:.4f}, required margin={min_relative_improvement:.0%})"
        )

    if promoted:
        client.transition_model_version_stage(
            name=model_name,
            version=challenger_version.version,
            stage="Production",
            archive_existing_versions=True,
        )

    return {
        "promoted": promoted,
        "reason": reason,
        "challenger_version": challenger_version.version,
        "challenger_metric": new_metric,
        "champion_metric": champion_metric,
        "champion_version": champion_version.version if champion_version else None,
    }
