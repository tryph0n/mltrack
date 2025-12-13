"""Test imports and basic functionality of mltrack modules."""

import os
from unittest.mock import patch


def test_config_module_imports():
    """Test that mltrack.config module imports successfully."""
    from mltrack.config import MLtrackConfig, get_s3_path, setup_mlflow

    assert MLtrackConfig is not None
    assert setup_mlflow is not None
    assert get_s3_path is not None


def test_train_module_imports():
    """Test that mltrack.train module imports successfully."""
    from mltrack.train import main, train_model

    assert main is not None
    assert train_model is not None


def test_mltrack_config_instantiation():
    """Test that MLtrackConfig can be instantiated with mock env vars."""
    from mltrack.config import MLtrackConfig

    # Test direct instantiation
    config = MLtrackConfig(
        tracking_uri="http://localhost:5000",
        s3_bucket_mlflow="test-mlflow",
        s3_bucket_data="test-data",
        aws_region="eu-west-1",
    )

    assert config.tracking_uri == "http://localhost:5000"
    assert config.s3_bucket_mlflow == "test-mlflow"
    assert config.s3_bucket_data == "test-data"
    assert config.aws_region == "eu-west-1"


def test_mltrack_config_from_env():
    """Test that MLtrackConfig.from_env() works with mock env vars."""
    from mltrack.config import MLtrackConfig

    mock_env = {
        "MLFLOW_TRACKING_URI": "http://test:5000",
        "S3_BUCKET_MLFLOW": "test-mlflow-bucket",
        "S3_BUCKET_DATA": "test-data-bucket",
        "AWS_REGION": "us-east-1",
    }

    with patch.dict(os.environ, mock_env):
        config = MLtrackConfig.from_env()

        assert config.tracking_uri == "http://test:5000"
        assert config.s3_bucket_mlflow == "test-mlflow-bucket"
        assert config.s3_bucket_data == "test-data-bucket"
        assert config.aws_region == "us-east-1"


def test_get_s3_path():
    """Test S3 path construction utility."""
    from mltrack.config import get_s3_path

    path = get_s3_path("my-bucket", "folder", "subfolder", "file.txt")
    assert path == "s3://my-bucket/folder/subfolder/file.txt"

    path = get_s3_path("bucket", "single-file.csv")
    assert path == "s3://bucket/single-file.csv"
