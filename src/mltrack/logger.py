"""Logging configuration for MLtrack."""

import logging
import os


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Set level based on environment
    env = os.getenv("ENV", "production")
    level = logging.DEBUG if env == "local" else logging.INFO
    logger.setLevel(level)

    # Console handler (if not already configured)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)

        # Format: timestamp - name - level - message
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
