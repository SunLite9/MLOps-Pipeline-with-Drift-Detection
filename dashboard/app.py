"""Monitoring dashboard: prediction volume, latency, drift score over time,
and a timeline of retraining/promotion events. Reads directly from the
prediction log (SQLite) and MLflow (registry + the drift-monitoring
experiment) — no separate metrics pipeline needed.

Run with: streamlit run dashboard/app.py
"""

import os
import sqlite3
from pathlib import Path

import altair as alt
import mlflow
import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient

MODEL_NAME = "fraud-model"
DRIFT_EXPERIMENT_NAME = "drift-monitoring"
PSI_THRESHOLD = 0.2
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
PREDICTION_LOG_DB = Path(
    os.environ.get("PREDICTION_LOG_DB", Path(__file__).resolve().parents[1] / "data" / "predictions.db")
)

st.set_page_config(page_title="Fraud Detection — Monitoring", layout="wide")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@st.cache_data(ttl=15)
def load_predictions() -> pd.DataFrame:
    if not PREDICTION_LOG_DB.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(PREDICTION_LOG_DB)
    df = pd.read_sql_query(
        "SELECT timestamp, model_version, fraud_probability, fraud_prediction, latency_ms FROM predictions",
        conn,
    )
    conn.close()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(ttl=15)
def load_drift_history() -> pd.DataFrame:
    client = MlflowClient()
    try:
        experiment = client.get_experiment_by_name(DRIFT_EXPERIMENT_NAME)
    except Exception:
        return pd.DataFrame()
    if experiment is None:
        return pd.DataFrame()

    runs = client.search_runs([experiment.experiment_id], order_by=["start_time ASC"], max_results=2000)
    rows = []
    for run in runs:
        rows.append(
            {
                "timestamp": pd.to_datetime(run.info.start_time, unit="ms"),
                "max_psi": run.data.metrics.get("max_psi"),
                "n_samples": run.data.metrics.get("n_samples"),
                "drifted": run.data.tags.get("drifted"),
                "production_version": run.data.tags.get("production_version"),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=15)
def load_model_timeline() -> pd.DataFrame:
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except Exception:
        return pd.DataFrame()

    rows = []
    for v in versions:
        try:
            run = client.get_run(v.run_id)
            metric = run.data.metrics.get("val_pr_auc")
        except Exception:
            metric = None
        rows.append(
            {
                "version": v.version,
                "created": pd.to_datetime(v.creation_timestamp, unit="ms"),
                "stage": v.current_stage,
                "val_pr_auc": metric,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("created")
    return df


st.title("Fraud Detection — Monitoring")

predictions = load_predictions()
drift_history = load_drift_history()
model_timeline = load_model_timeline()

# --- Prediction volume ---
st.header("Prediction volume")
if predictions.empty:
    st.info("No predictions logged yet.")
else:
    volume = predictions.set_index("timestamp").resample("1h").size().reset_index(name="requests")
    chart = (
        alt.Chart(volume)
        .mark_bar()
        .encode(x=alt.X("timestamp:T", title="Hour"), y=alt.Y("requests:Q", title="Requests"))
        .properties(height=250)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(f"{len(predictions)} total predictions logged.")

# --- Latency ---
st.header("Prediction latency")
if predictions.empty or predictions["latency_ms"].isna().all():
    st.info("No latency data logged yet.")
else:
    latency = predictions.dropna(subset=["latency_ms"]).set_index("timestamp")
    latency_stats = (
        latency["latency_ms"]
        .resample("1h")
        .agg(avg_ms="mean", p99_ms=lambda s: s.quantile(0.99))
        .reset_index()
    )
    latency_long = latency_stats.melt(id_vars="timestamp", var_name="stat", value_name="ms")
    chart = (
        alt.Chart(latency_long)
        .mark_line(point=True)
        .encode(x=alt.X("timestamp:T", title="Hour"), y=alt.Y("ms:Q", title="Latency (ms)"), color="stat:N")
        .properties(height=250)
    )
    st.altair_chart(chart, use_container_width=True)
    col1, col2 = st.columns(2)
    col1.metric("Avg latency (overall)", f"{latency['latency_ms'].mean():.2f} ms")
    col2.metric("p99 latency (overall)", f"{latency['latency_ms'].quantile(0.99):.2f} ms")

# --- Drift score ---
st.header("Drift score (PSI)")
if drift_history.empty:
    st.info("No drift checks logged yet — run drift_check_dag at least once.")
else:
    threshold_df = pd.DataFrame({"threshold": [PSI_THRESHOLD]})
    line = (
        alt.Chart(drift_history)
        .mark_line(point=True)
        .encode(
            x=alt.X("timestamp:T", title="Drift check time"),
            y=alt.Y("max_psi:Q", title="Max PSI across features"),
            color=alt.Color("drifted:N", title="Drifted?"),
            tooltip=["timestamp", "max_psi", "n_samples", "production_version", "drifted"],
        )
        .properties(height=300)
    )
    rule = alt.Chart(threshold_df).mark_rule(color="red", strokeDash=[4, 4]).encode(y="threshold:Q")
    st.altair_chart(line + rule, use_container_width=True)
    st.caption("Red dashed line: PSI drift threshold (0.2).")

# --- Retraining / promotion timeline ---
st.header("Retraining and promotion timeline")
if model_timeline.empty:
    st.info("No model versions registered yet.")
else:
    display = model_timeline.rename(
        columns={"version": "Version", "created": "Trained at", "stage": "Stage", "val_pr_auc": "val_pr_auc"}
    )
    st.dataframe(display, use_container_width=True, hide_index=True)

    chart = (
        alt.Chart(model_timeline)
        .mark_circle(size=120)
        .encode(
            x=alt.X("created:T", title="Trained at"),
            y=alt.Y("val_pr_auc:Q", title="val_pr_auc"),
            color=alt.Color("stage:N", title="Registry stage"),
            tooltip=["version", "created", "stage", "val_pr_auc"],
        )
        .properties(height=250)
    )
    st.altair_chart(chart, use_container_width=True)
