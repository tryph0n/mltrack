"""Main entry point for MLtrack training pipeline.

Usage:
    # Default (train all models on Iris)
    uv run python -m mltrack.main

    # Custom experiment name
    uv run python -m mltrack.main --experiment-name "My Experiment"

    # Train specific models
    uv run python -m mltrack.main --models logistic_regression random_forest

    # Custom test size
    uv run python -m mltrack.main --test-size 0.3
"""

import argparse
import sys

import mlflow

from mltrack.config import setup_mlflow
from mltrack.logger import get_logger
from mltrack.models import list_models
from mltrack.pipelines import TrainingPipeline, run_pipeline

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Train ML models with MLflow tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="Iris Classification",
        help="MLflow experiment name",
    )

    parser.add_argument(
        "--loader",
        type=str,
        default="iris",
        help="Dataset loader name",
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Model names to train (space-separated). If not specified, trains all available models.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (0.0-1.0)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Parse arguments
        args = parse_args()

        # Setup MLflow
        setup_mlflow()
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

        # Determine which models to train
        model_names = args.models if args.models else list_models()

        # Create pipeline configuration
        pipeline = TrainingPipeline(
            loader_name=args.loader,
            model_names=model_names,
            test_size=args.test_size,
        )

        logger.info(f"Starting training pipeline: {args.experiment_name}")
        logger.info(f"Models: {', '.join(model_names)}")

        # Run pipeline
        results = run_pipeline(pipeline, experiment_name=args.experiment_name)

        # Summary
        logger.info(f"\n{'='*50}")
        logger.info("Training completed successfully!")
        logger.info(f"Trained {len(results)} models")
        logger.info(f"View results at: {mlflow.get_tracking_uri()}")
        logger.info(f"{'='*50}")

        return 0

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please check your arguments and environment variables.")
        return 1

    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error("Please check MLflow server connection and configuration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
