"""Training pipeline orchestration with MLflow."""

import time
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

from mltrack.loaders import get_loader
from mltrack.logger import get_logger
from mltrack.models import get_model

logger = get_logger(__name__)


@dataclass
class TrainingPipeline:
    """Configuration for a training pipeline.

    Attributes:
        loader_name: Name of the data loader to use
        model_names: List of model names to train
        test_size: Fraction of data to use for testing (0.0-1.0)
        random_state: Random seed for reproducibility
    """

    loader_name: str
    model_names: list[str]
    test_size: float = 0.2
    random_state: int = 42


def train_single_model(
    model_name: str,
    X_train,
    X_test,
    y_train,
    y_test,
) -> dict:
    """Train a single model and log to MLflow.

    Args:
        model_name: Name of the model from registry
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels

    Returns:
        Dictionary with training metrics
    """
    with mlflow.start_run(run_name=model_name):
        start_time = time.time()

        # Get model instance
        model = get_model(model_name)

        # Train model
        model.fit(X_train, y_train)

        # Calculate training time
        training_time = time.time() - start_time

        # Log custom metrics
        mlflow.log_metric("training_time", training_time)

        # MLflow autolog automatically logs model params, metrics, artifacts

        logger.info(f"✓ {model_name} trained in {training_time:.4f}s")

        return {
            "model_name": model_name,
            "training_time": training_time,
        }


def run_pipeline(
    pipeline: TrainingPipeline,
    experiment_name: str = "Default",
) -> list[dict]:
    """Run a complete training pipeline.

    Args:
        pipeline: TrainingPipeline configuration
        experiment_name: MLflow experiment name

    Returns:
        List of results dictionaries (one per model)

    Raises:
        ValueError: If loader or model names are invalid
    """
    # Setup MLflow experiment
    mlflow.set_experiment(experiment_name)

    # Enable autologging
    mlflow.sklearn.autolog()  # pyright: ignore[reportPrivateImportUsage]

    # Load dataset
    logger.info(f"Loading dataset: {pipeline.loader_name}")
    try:
        loader = get_loader(pipeline.loader_name)
        X, y, feature_names, target_names = loader.load()
    except ValueError as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=pipeline.test_size,
        random_state=pipeline.random_state,  # pyright: ignore[reportAttributeAccessIssue]
    )
    logger.info(f"Dataset split: {len(X_train)} train, {len(X_test)} test samples")

    # Train all models
    logger.info(f"Training {len(pipeline.model_names)} models...")
    results = []
    for model_name in pipeline.model_names:
        result = train_single_model(
            model_name,
            X_train,
            X_test,
            y_train,
            y_test,
        )
        results.append(result)

    logger.info("✓ All models trained successfully!")
    logger.info(f"View results at: {mlflow.get_tracking_uri()}")

    return results
