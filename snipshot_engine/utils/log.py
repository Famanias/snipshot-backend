"""Logging utilities for snipshot_engine."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``snipshot_engine`` namespace."""
    return logging.getLogger(f"snipshot_engine.{name}")
