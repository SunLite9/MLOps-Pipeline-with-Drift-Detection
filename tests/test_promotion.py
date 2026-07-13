"""Unit tests for the champion/challenger promotion gate
(src/training/promotion.py) — the single piece of logic that decides
whether a model actually reaches production, and previously had zero test
coverage.

Each test gets its own throwaway local MLflow store (function-scoped
`tmp_path`), so tests don't share registry state and don't depend on the
session-scoped `mlflow_test_env` fixture used by the other test files.
"""

import mlflow
import mlflow.sklearn
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression

from src.training.promotion import register_if_better

MODEL_NAME = "test-fraud-model"
METRIC_NAME = "val_pr_auc"


@pytest.fixture
def mlflow_local(tmp_path, monkeypatch):
    """A fresh, isolated local MLflow store per test."""
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("promotion-gate-tests")
    yield tracking_uri


def _register_run(metric_value: float, model_name: str = MODEL_NAME) -> str:
    """Logs a trivial run with a given val_pr_auc and registers a model
    version against it — the minimum needed for register_if_better() to
    have something to promote."""
    with mlflow.start_run() as run:
        mlflow.log_metric(METRIC_NAME, metric_value)
        model = LogisticRegression().fit([[0], [1]], [0, 1])
        mlflow.sklearn.log_model(model, name="model", registered_model_name=model_name)
        return run.info.run_id


def _current_stage(model_name: str = MODEL_NAME) -> dict:
    client = MlflowClient()
    return {str(v.version): v.current_stage for v in client.search_model_versions(f"name='{model_name}'")}


def test_bootstrap_promotes_first_model_unconditionally(mlflow_local):
    run_id = _register_run(0.10)

    decision = register_if_better(run_id, 0.10, model_name=MODEL_NAME)

    assert decision["promoted"] is True
    assert "bootstrap" in decision["reason"]
    assert decision["champion_metric"] is None
    assert _current_stage()["1"] == "Production"


def test_rejects_challenger_below_the_margin(mlflow_local):
    champion_run = _register_run(0.50)
    register_if_better(champion_run, 0.50, model_name=MODEL_NAME)  # bootstrap champion

    challenger_run = _register_run(0.509)  # +1.8%, below the 2% margin
    decision = register_if_better(challenger_run, 0.509, model_name=MODEL_NAME)

    assert decision["promoted"] is False
    assert decision["champion_metric"] == 0.50
    stages = _current_stage()
    assert stages["1"] == "Production"  # champion untouched
    assert stages["2"] == "None"  # challenger not promoted


def test_promotes_challenger_that_clears_the_margin(mlflow_local):
    champion_run = _register_run(0.50)
    register_if_better(champion_run, 0.50, model_name=MODEL_NAME)

    challenger_run = _register_run(0.52)  # +4%, clears the 2% margin
    decision = register_if_better(challenger_run, 0.52, model_name=MODEL_NAME)

    assert decision["promoted"] is True
    stages = _current_stage()
    assert stages["1"] == "Archived"  # prior champion archived
    assert stages["2"] == "Production"  # challenger promoted


def test_boundary_exact_margin_promotes(mlflow_local):
    """The gate uses >=, not >, so a challenger exactly at the threshold
    should promote, not tie."""
    champion_metric = 0.50
    champion_run = _register_run(champion_metric)
    register_if_better(champion_run, champion_metric, model_name=MODEL_NAME)

    exact_threshold = champion_metric * 1.02
    challenger_run = _register_run(exact_threshold)
    decision = register_if_better(challenger_run, exact_threshold, model_name=MODEL_NAME)

    assert decision["promoted"] is True


def test_custom_margin_is_respected(mlflow_local):
    champion_run = _register_run(0.50)
    register_if_better(champion_run, 0.50, model_name=MODEL_NAME)

    # +4% beats the default 2% margin but not a 10% margin.
    challenger_run = _register_run(0.52)
    decision = register_if_better(
        challenger_run, 0.52, model_name=MODEL_NAME, min_relative_improvement=0.10
    )

    assert decision["promoted"] is False


def test_raises_if_run_has_no_registered_model_version(mlflow_local):
    with mlflow.start_run() as run:
        mlflow.log_metric(METRIC_NAME, 0.5)
        unregistered_run_id = run.info.run_id

    with pytest.raises(ValueError, match="No registered model version"):
        register_if_better(unregistered_run_id, 0.5, model_name=MODEL_NAME)
