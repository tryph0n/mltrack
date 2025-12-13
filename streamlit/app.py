"""Streamlit dashboard for MLtrack metrics comparison.

This dashboard connects to the MLflow tracking server and displays
a comparison of all tracked model runs with interactive visualizations.
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import mlflow
import streamlit as st

# Page config
st.set_page_config(page_title="MLtrack Dashboard", page_icon="📊", layout="wide")

# Connect to MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

st.title("📊 MLtrack Dashboard")
st.markdown(f"**MLflow Tracking URI:** `{MLFLOW_TRACKING_URI}`")

# Auto-refresh option
col1, col2 = st.columns([3, 1])
with col2:
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        st.rerun()


@st.cache_data(ttl=30)
def fetch_runs():
    """Fetch all runs from all experiments."""
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()

    all_runs = []
    for exp in experiments:
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        all_runs.extend(runs)

    return all_runs


def parse_runs_to_df(runs):
    """Parse MLflow runs into a pandas DataFrame."""
    data = []
    for run in runs:
        metrics = run.data.metrics
        params = run.data.params
        data.append(
            {
                "run_name": run.info.run_name or run.info.run_id[:8],
                "model_type": params.get("model_type", "Unknown"),
                "accuracy": metrics.get("training_score", metrics.get("accuracy", 0)),
                "f1_score": metrics.get("training_f1_score", 0),
                "training_time": metrics.get("training_time", 0),
                "run_id": run.info.run_id,
                "start_time": pd.to_datetime(run.info.start_time, unit="ms"),
            }
        )
    return pd.DataFrame(data)


# Fetch and parse runs
try:
    runs = fetch_runs()
    if not runs:
        st.warning("No runs found. Train some models to see results!")
        st.stop()

    df = parse_runs_to_df(runs)

    # Summary metrics
    st.header("📈 Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Runs", len(df))
    with col2:
        best_model = df.loc[df["accuracy"].idxmax(), "run_name"]
        st.metric("Best Model", best_model)
    with col3:
        avg_accuracy = df["accuracy"].mean()
        st.metric("Avg Accuracy", f"{avg_accuracy:.4f}")
    with col4:
        avg_time = df["training_time"].mean()
        st.metric("Avg Training Time", f"{avg_time:.4f}s")

    # Comparison table
    st.header("📋 Model Comparison Table")
    st.dataframe(
        df[["run_name", "model_type", "accuracy", "f1_score", "training_time"]].sort_values(
            "accuracy", ascending=False
        ),
        use_container_width=True,
    )

    # Line chart: Metrics evolution
    st.header("📉 Metrics Evolution Over Time")
    fig_line = px.line(
        df.sort_values("start_time"),
        x="start_time",
        y="accuracy",
        color="run_name",
        markers=True,
        title="Accuracy Over Time",
        labels={"start_time": "Timestamp", "accuracy": "Accuracy", "run_name": "Model"},
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # Bar chart: Model comparison
    st.header("📊 Model Performance Comparison")
    fig_bar = go.Figure(
        data=[
            go.Bar(name="Accuracy", x=df["run_name"], y=df["accuracy"]),
            go.Bar(name="F1 Score", x=df["run_name"], y=df["f1_score"]),
        ]
    )
    fig_bar.update_layout(
        barmode="group",
        title="Accuracy vs F1 Score by Model",
        xaxis_title="Model",
        yaxis_title="Score",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Best model details
    st.header("🏆 Best Model Details")
    best_idx = df["accuracy"].idxmax()
    best_row = df.loc[best_idx]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Name", best_row["run_name"])
    with col2:
        st.metric("Accuracy", f"{best_row['accuracy']:.4f}")
    with col3:
        st.metric("F1 Score", f"{best_row['f1_score']:.4f}")

except Exception as e:
    st.error(f"Error connecting to MLflow: {e}")
    st.info("Make sure MLflow server is running and accessible.")
