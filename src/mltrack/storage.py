"""S3 storage abstraction for datasets and models."""

import io
import pickle
from typing import Any

import boto3
import numpy as np

from mltrack.config import MLtrackConfig
from mltrack.logger import get_logger

logger = get_logger(__name__)


class S3Storage:
    """S3 storage handler for ML artifacts.

    Attributes:
        bucket: S3 bucket name
        client: Boto3 S3 client
    """

    def __init__(self, bucket: str | None = None):
        """Initialize S3 storage.

        Args:
            bucket: S3 bucket name. If None, uses config from environment.
        """
        if bucket is None:
            config = MLtrackConfig.from_env()
            bucket = config.s3_bucket_data

        self.bucket = bucket
        self.client = boto3.client("s3")
        logger.info(f"S3Storage initialized with bucket: {self.bucket}")

    def save_preprocessed(
        self,
        X: np.ndarray,
        y: np.ndarray,
        path: str,
    ) -> str:
        """Save preprocessed data to S3.

        Args:
            X: Feature matrix
            y: Target vector
            path: S3 key path (without bucket)

        Returns:
            S3 URI of saved data

        Example:
            >>> storage = S3Storage()
            >>> uri = storage.save_preprocessed(X, y, "training/iris_preprocessed.pkl")
        """
        data = {"X": X, "y": y}
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        buffer.seek(0)

        self.client.upload_fileobj(buffer, self.bucket, path)
        s3_uri = f"s3://{self.bucket}/{path}"

        logger.info(f"Saved preprocessed data to {s3_uri}")
        return s3_uri

    def save_model(self, model: Any, path: str) -> str:
        """Save trained model to S3.

        Args:
            model: Trained scikit-learn model
            path: S3 key path (without bucket)

        Returns:
            S3 URI of saved model

        Example:
            >>> storage = S3Storage()
            >>> uri = storage.save_model(trained_model, "models/iris_rf.pkl")
        """
        buffer = io.BytesIO()
        pickle.dump(model, buffer)
        buffer.seek(0)

        self.client.upload_fileobj(buffer, self.bucket, path)
        s3_uri = f"s3://{self.bucket}/{path}"

        logger.info(f"Saved model to {s3_uri}")
        return s3_uri

    def load_preprocessed(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        """Load preprocessed data from S3.

        Args:
            path: S3 key path (without bucket)

        Returns:
            Tuple of (X, y)
        """
        buffer = io.BytesIO()
        self.client.download_fileobj(self.bucket, path, buffer)
        buffer.seek(0)

        data = pickle.load(buffer)
        logger.info(f"Loaded preprocessed data from s3://{self.bucket}/{path}")

        return data["X"], data["y"]

    def load_model(self, path: str) -> Any:
        """Load trained model from S3.

        Args:
            path: S3 key path (without bucket)

        Returns:
            Trained model instance
        """
        buffer = io.BytesIO()
        self.client.download_fileobj(self.bucket, path, buffer)
        buffer.seek(0)

        model = pickle.load(buffer)
        logger.info(f"Loaded model from s3://{self.bucket}/{path}")

        return model


def get_storage(bucket: str | None = None) -> S3Storage:
    """Get an S3Storage instance.

    Args:
        bucket: Optional bucket name. Uses config if not provided.

    Returns:
        S3Storage instance
    """
    return S3Storage(bucket=bucket)
