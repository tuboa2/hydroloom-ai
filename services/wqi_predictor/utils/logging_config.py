from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Final

from ..config import LOG_DIR

_DEFAULT_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(filename)s:%(lineno)d | %(message)s"
)
_DEFAULT_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

_LOG_FILE: Final[str] = "wqi_predictor.log"
_MAX_BYTES: Final[int] = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT: Final[int] = 5

_CONFIGURED_LOGGERS: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """Return a logger with console and rotating file handlers.

    Calling this function multiple times with the same ``name`` will
    return the same logger **without** adding duplicate handlers.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    if name in _CONFIGURED_LOGGERS:
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent propagation to the root logger to avoid duplicate messages
    # when multiple modules share a common parent logger.
    logger.propagate = False

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATE_FORMAT)

    # Console handler — INFO and above to stderr
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating file handler — DEBUG and above
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=str(LOG_DIR / _LOG_FILE),
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # If the log directory is not writable (e.g. CI, read-only fs),
        # fall back to console-only logging without failing.
        logger.warning(
            "Could not create rotating file handler at %s — falling back to console-only logging.",
            LOG_DIR / _LOG_FILE,
        )

    _CONFIGURED_LOGGERS.add(name)
    return logger
