"""Test imports and basic functionality of mltrack modules."""

import os
from unittest.mock import patch


# ============================================================================
# CORE MODULE IMPORTS
# ============================================================================


def test_config_module_imports():
    """Test config module imports."""
    from mltrack.config import MLtrackConfig, get_s3_path, setup_mlflow

    assert MLtrackConfig is not None
    assert setup_mlflow is not None
    assert get_s3_path is not None


def test_train_module_imports():
    """Test train module imports."""
    from mltrack.train import main, train_model

    assert main is not None
    assert train_model is not None


# ============================================================================
# NEW MODULE IMPORTS
# ============================================================================


def test_loader_imports():
    """Test loader module imports."""
    from mltrack.loaders import DataLoader, IrisLoader, get_loader

    assert DataLoader is not None
    assert IrisLoader is not None
    assert get_loader is not None


def test_model_imports():
    """Test model module imports."""
    from mltrack.models import MODELS, get_model, list_models

    assert MODELS is not None
    assert get_model is not None
    assert list_models is not None


def test_pipeline_imports():
    """Test pipeline module imports."""
    from mltrack.pipelines import TrainingPipeline, run_pipeline, train_single_model

    assert TrainingPipeline is not None
    assert run_pipeline is not None
    assert train_single_model is not None


def test_storage_imports():
    """Test storage module imports."""
    from mltrack.storage import S3Storage, get_storage

    assert S3Storage is not None
    assert get_storage is not None


def test_logger_imports():
    """Test logger module imports."""
    from mltrack.logger import get_logger

    assert get_logger is not None


# ============================================================================
# CORE MODULE FUNCTIONALITY TESTS
# ============================================================================


def test_mltrack_config_instantiation():
    """Test MLtrackConfig direct instantiation."""
    from mltrack.config import MLtrackConfig

    config = MLtrackConfig(
        tracking_uri="http://localhost:5000",
        s3_bucket_mlflow="test-mlflow",
        s3_bucket_data="test-data",
        aws_region="eu-west-3",
    )

    assert config.tracking_uri == "http://localhost:5000"
    assert config.s3_bucket_mlflow == "test-mlflow"
    assert config.s3_bucket_data == "test-data"
    assert config.aws_region == "eu-west-3"


def test_mltrack_config_from_env():
    """Test MLtrackConfig.from_env() with mocked environment."""
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
    """Test get_s3_path utility function."""
    from mltrack.config import get_s3_path

    path = get_s3_path("my-bucket", "folder", "subfolder", "file.txt")
    assert path == "s3://my-bucket/folder/subfolder/file.txt"

    path = get_s3_path("bucket", "single-file.csv")
    assert path == "s3://bucket/single-file.csv"


# ============================================================================
# NEW MODULE FUNCTIONALITY TESTS
# ============================================================================


def test_loader_functionality():
    """Test basic loader functionality."""
    from mltrack.loaders import get_loader

    loader = get_loader("iris")
    X, y, feature_names, target_names = loader.load()

    assert X is not None
    assert y is not None
    assert len(feature_names) == 4
    assert len(target_names) == 3
    assert X.shape[0] == 150


def test_model_functionality():
    """Test basic model functionality."""
    from mltrack.models import list_models, get_model

    models = list_models()
    assert len(models) > 0
    assert "logistic_regression" in models
    assert "random_forest" in models
    assert "svm_rbf" in models

    model = get_model("logistic_regression")
    assert model is not None
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_model_override_params():
    """Test get_model with parameter overrides."""
    from mltrack.models import get_model

    model = get_model("random_forest", n_estimators=50)
    assert model.n_estimators == 50


def test_logger_functionality():
    """Test basic logger functionality."""
    from mltrack.logger import get_logger

    logger = get_logger(__name__)
    assert logger is not None
    assert logger.name == __name__
    assert hasattr(logger, "info")
    assert hasattr(logger, "debug")
    assert hasattr(logger, "error")


def test_storage_initialization():
    """Test S3Storage initialization with bucket."""
    from mltrack.storage import S3Storage

    storage = S3Storage(bucket="test-bucket")
    assert storage is not None
    assert storage.bucket == "test-bucket"
    assert hasattr(storage, "client")


def test_get_storage():
    """Test get_storage factory function."""
    from mltrack.storage import get_storage

    storage = get_storage(bucket="test-bucket")
    assert storage is not None
    assert storage.bucket == "test-bucket"


def test_training_pipeline_instantiation():
    """Test TrainingPipeline dataclass instantiation."""
    from mltrack.pipelines import TrainingPipeline

    pipeline = TrainingPipeline(
        loader_name="iris",
        model_names=["logistic_regression", "random_forest"],
        test_size=0.2,
        random_state=42,
    )

    assert pipeline.loader_name == "iris"
    assert pipeline.model_names == ["logistic_regression", "random_forest"]
    assert pipeline.test_size == 0.2
    assert pipeline.random_state == 42


def test_training_pipeline_defaults():
    """Test TrainingPipeline with default parameters."""
    from mltrack.pipelines import TrainingPipeline

    pipeline = TrainingPipeline(
        loader_name="iris",
        model_names=["logistic_regression"],
    )

    assert pipeline.test_size == 0.2
    assert pipeline.random_state == 42
