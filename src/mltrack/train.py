"""Training script for Iris dataset with multiple classifiers.

This script trains three different models on the Iris dataset and logs
all experiments to MLflow. Each model is tracked as a separate run with
metrics, parameters, and artifacts automatically logged via MLflow autolog.

Usage:
    # Local (with .env file)
    uv run python src/mltrack/train.py

    # External compute (Colab, AWS, etc.)
    Set environment variables then run:
    python src/mltrack/train.py
"""

import time

import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import mlflow
from mltrack.config import setup_mlflow


def train_model(model, model_name: str, X_train, X_test, y_train, y_test):
    """Train a single model and log to MLflow.

    Args:
        model: Sklearn classifier instance
        model_name: Descriptive name for the run
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
    """
    with mlflow.start_run(run_name=model_name):
        start_time = time.time()

        # Train model
        model.fit(X_train, y_train)

        # Calculate training time
        training_time = time.time() - start_time

        # Log custom metrics
        mlflow.log_metric("training_time", training_time)

        # MLflow autolog automatically logs:
        # - Model parameters
        # - Training metrics (accuracy, precision, recall, f1)
        # - Model artifacts

        print(f"✓ {model_name} trained in {training_time:.4f}s")


def main():
    """Main training pipeline."""
    try:
        # Setup MLflow tracking
        setup_mlflow()
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

        # Enable autologging for sklearn
        mlflow.sklearn.autolog()

        # Load Iris dataset
        print("\nLoading Iris dataset...")
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )
        print(f"Dataset split: {len(X_train)} train, {len(X_test)} test samples")

        # Define models
        models = [
            (LogisticRegression(max_iter=200, random_state=42), "Logistic Regression"),
            (RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest"),
            (SVC(kernel="rbf", random_state=42), "SVM (RBF kernel)"),
        ]

        # Train all models
        print("\nTraining models...")
        for model, name in models:
            train_model(model, name, X_train, X_test, y_train, y_test)

        print("\n✓ All models trained successfully!")
        print(f"View results at: {mlflow.get_tracking_uri()}")

    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
        print("Please check your environment variables and try again.")
        raise
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        print("Please check MLflow server connection and S3 credentials.")
        raise


if __name__ == "__main__":
    main()
