import logging
import os
from logging.config import dictConfig

def setup_logging():
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
        config = {
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
                                                                                                                                                                }
                                                                                                                                                                        },
                                                                                                                                                                                "root": {
                                                                                                                                                                                            "handlers": ["console"],
                                                                                                                                                                                                        "level": level,
                                                                                                                                                                                                                },
                                                                                                                                                                                                                    }
                                                                                                                                                                                                                        dictConfig(config)

                                                                                                                                                                                                                        # Usage: call setup_logging() early in app startup