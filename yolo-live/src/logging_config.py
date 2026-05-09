"""
src/logging_config.py

Simple logging configuration helper. Call setup_logging() early in app startup.
"""
import os
import logging
from logging.config import dictConfig
from typing import Dict, Any


def setup_logging() -> None:
    """
    Configure Python logging using a dictionary config.
    Reads LOG_LEVEL from environment (default INFO).
    """
    level = os.environ.get("LOG_LEVEL", "INFO").upper()

    config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": level,
                "stream": "ext://sys.stdout",
            }
        },
        "root": {
            "handlers": ["console"],
            "level": level,
        },
    }

    dictConfig(config)


# Usage: call setup_logging() early in application startup, e.g. in src/main.py or src/app.py
