"""Model registry for scikit-learn classifiers."""

from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# Registry: {name: (ClassifierClass, default_params)}
MODELS: dict[str, tuple[type[ClassifierMixin], dict[str, Any]]] = {
    "logistic_regression": (
        LogisticRegression,
        {"max_iter": 200, "random_state": 42},
    ),
    "random_forest": (
        RandomForestClassifier,
        {"n_estimators": 100, "random_state": 42},
    ),
    "svm_rbf": (
        SVC,
        {"kernel": "rbf", "random_state": 42},
    ),
}


def get_model(name: str, **kwargs: Any) -> ClassifierMixin:
    """Get a configured model instance.

    Args:
        name: Model name from registry
        **kwargs: Override default parameters

    Returns:
        Configured classifier instance

    Raises:
        ValueError: If model name is not recognized

    Example:
        >>> model = get_model("random_forest")
        >>> model = get_model("svm_rbf", kernel="linear")
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list_models()}")

    model_class, default_params = MODELS[name]

    # Merge default params with overrides
    params = {**default_params, **kwargs}

    return model_class(**params)


def list_models() -> list[str]:
    """List all available model names.

    Returns:
        List of model names in the registry
    """
    return list(MODELS.keys())
