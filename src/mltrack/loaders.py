"""Data loaders for machine learning datasets."""

from abc import ABC, abstractmethod

import numpy as np
from sklearn.datasets import load_iris as sklearn_load_iris


class DataLoader(ABC):
    """Abstract base class for dataset loaders.

    All loaders must implement the load() method which returns
    features, targets, feature names, and target names.
    """

    @abstractmethod
    def load(self) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
        """Load the dataset.

        Returns:
            Tuple of (X, y, feature_names, target_names):
                - X: Feature matrix (n_samples, n_features)
                - y: Target vector (n_samples,)
                - feature_names: List of feature names
                - target_names: List of target class names
        """
        pass


class IrisLoader(DataLoader):
    """Loader for the Iris dataset."""

    def load(self) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
        """Load the Iris dataset from sklearn.

        Returns:
            Tuple of (X, y, feature_names, target_names)
        """
        iris = sklearn_load_iris()
        return (
            iris.data,
            iris.target,
            list(iris.feature_names),
            list(iris.target_names),
        )


def get_loader(name: str) -> DataLoader:
    """Get a data loader by name.

    Args:
        name: Name of the dataset ("iris")

    Returns:
        DataLoader instance

    Raises:
        ValueError: If loader name is not recognized
    """
    loaders = {
        "iris": IrisLoader,
    }

    if name not in loaders:
        raise ValueError(f"Unknown loader: {name}. Available: {list(loaders.keys())}")

    return loaders[name]()
