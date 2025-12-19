"""Logging utilities for consistent application-wide logging."""

import logging
import os
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Configure and return a logger with a consistent format.

    Log level can be controlled via the LOG_LEVEL environment variable.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(log_level)
    logger.propagate = False

    return logger