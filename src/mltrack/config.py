"""Configuration module for MLflow and S3 connections.

This module provides a simple abstraction layer for configuring MLflow tracking
and S3 storage. It works both locally (Docker setup) and on external compute
platforms (Colab, AWS SageMaker, Lambda AI, etc.).

Usage for external compute (e.g., Google Colab):
    ```python
    import os
    from mltrack.config import setup_mlflow, get_s3_path

    # Set environment variables
    os.environ["MLFLOW_TRACKING_URI"] = "http://your-mlflow-server:5000"
    os.environ["AWS_ACCESS_KEY_ID"] = "your-key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "your-secret"
    os.environ["AWS_REGION"] = "eu-west-1"
    os.environ["S3_BUCKET_MLFLOW"] = "mltrack-mlflow"
    os.environ["S3_BUCKET_DATA"] = "mltrack-data"

    # Setup MLflow
    setup_mlflow()

    # Train your model - MLflow will automatically track to your server
    ```
"""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

import mlflow

# Load .env file if present (local development)
load_dotenv()


@dataclass
class MLtrackConfig:
    """Configuration for MLtrack system.

    Attributes:
        tracking_uri: MLflow tracking server URI
        s3_bucket_mlflow: S3 bucket for MLflow artifacts
        s3_bucket_data: S3 bucket for training/inference data
        aws_region: AWS region for S3 buckets
    """

    tracking_uri: str
    s3_bucket_mlflow: str
    s3_bucket_data: str
    aws_region: str

    @classmethod
    def from_env(cls) -> "MLtrackConfig":
        """Load configuration from environment variables.

        Returns:
            MLtrackConfig instance populated from environment

        Raises:
            ValueError: If required environment variables are missing
        """
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        s3_bucket_mlflow = os.getenv("S3_BUCKET_MLFLOW")
        s3_bucket_data = os.getenv("S3_BUCKET_DATA")
        aws_region = os.getenv("AWS_REGION", "eu-west-1")

        if not tracking_uri:
            raise ValueError("MLFLOW_TRACKING_URI environment variable is required")
        if not s3_bucket_mlflow:
            raise ValueError("S3_BUCKET_MLFLOW environment variable is required")
        if not s3_bucket_data:
            raise ValueError("S3_BUCKET_DATA environment variable is required")

        return cls(
            tracking_uri=tracking_uri,
            s3_bucket_mlflow=s3_bucket_mlflow,
            s3_bucket_data=s3_bucket_data,
            aws_region=aws_region,
        )


def setup_mlflow(tracking_uri: Optional[str] = None) -> None:
    """Configure MLflow tracking URI.

    Args:
        tracking_uri: Optional tracking URI. If not provided, will use
                     MLFLOW_TRACKING_URI from environment variables.

    Example:
        >>> setup_mlflow()  # Uses env var
        >>> setup_mlflow("http://localhost:5000")  # Explicit URI
    """
    if tracking_uri is None:
        config = MLtrackConfig.from_env()
        tracking_uri = config.tracking_uri

    mlflow.set_tracking_uri(tracking_uri)


def get_s3_path(bucket: str, *path_parts: str) -> str:
    """Construct an S3 URI from bucket name and path components.

    Args:
        bucket: S3 bucket name
        *path_parts: Variable number of path components

    Returns:
        Complete S3 URI (s3://bucket/path/to/object)

    Example:
        >>> get_s3_path("mltrack-data", "training", "iris.csv")
        's3://mltrack-data/training/iris.csv'
        >>> get_s3_path("mltrack-mlflow", "artifacts", "model.pkl")
        's3://mltrack-mlflow/artifacts/model.pkl'
    """
    path = "/".join(path_parts)
    return f"s3://{bucket}/{path}"
