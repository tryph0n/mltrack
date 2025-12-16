"""MLflow data fetching and parsing utilities."""

import pandas as pd

import mlflow


def fetch_experiments() -> list:
    """Fetch all MLflow experiments.

    Returns:
        List of MLflow Experiment objects
    """
    client = mlflow.tracking.MlflowClient()
    return client.search_experiments()


def fetch_runs(experiment_id: str | None = None) -> pd.DataFrame:
    """Fetch runs and parse to DataFrame.

    Args:
        experiment_id: Optional experiment ID. If None, fetches from all experiments.

    Returns:
        DataFrame with columns:
            - run_name: Name or ID prefix
            - model_type: Model type from params
            - accuracy: Training score or accuracy metric
            - f1_score: F1 score metric
            - training_time: Training duration in seconds
            - run_id: Full run ID
            - start_time: Run start timestamp
    """
    client = mlflow.tracking.MlflowClient()

    # Fetch runs
    if experiment_id is None:
        # Fetch from all experiments
        experiments = client.search_experiments()
        all_runs = []
        for exp in experiments:
            runs = client.search_runs(experiment_ids=[exp.experiment_id])
            all_runs.extend(runs)
    else:
        all_runs = client.search_runs(experiment_ids=[experiment_id])

    # Parse to DataFrame
    data = []
    for run in all_runs:
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
