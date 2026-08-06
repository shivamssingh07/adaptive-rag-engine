"""Structured logging configuration.

Provides one entry point, :func:`configure_logging`, that wires up the root
logger with:

    * A console handler — human-readable in development, JSON in production
      (controlled by ``settings.log_json``).
    * A rotating file handler writing to ``settings.log_dir`` so logs
      survive process restarts and don't grow unbounded.

Every log record includes a ``request_id`` field (defaulting to ``"-"``
when logged outside of an HTTP request) so log lines can be correlated to a
specific chat request across the router, retrieval, grading, and
generation nodes. See `backend.api.middleware.logging_middleware` for how
``request_id`` is bound during request handling.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from pythonjsonlogger import json as jsonlogger

from backend.config.constants import LOG_FILE_BACKUP_COUNT, LOG_FILE_MAX_BYTES, LOG_FILE_NAME
from backend.config.settings import Settings

_CONSOLE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | request_id=%(request_id)s | %(message)s"
)


class _RequestIdFilter(logging.Filter):
    """Ensure every log record has a `request_id` attribute.

    Log calls made outside of an active HTTP request (e.g. during startup,
    background indexing) don't have a request ID bound to them; without
    this filter, the formatter above would raise a `KeyError`.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


def configure_logging(settings: Settings) -> None:
    """Configure the root logger for the entire process.

    Idempotent: safe to call more than once (e.g. once at module import and
    again during the FastAPI lifespan startup hook) — existing handlers are
    removed before new ones are attached, so log lines are never duplicated.

    Args:
        settings: The active application settings, used to determine log
            level, output directory, and whether to emit JSON or
            human-readable console output.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level)

    # Remove any handlers from a previous call to keep this idempotent.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    request_id_filter = _RequestIdFilter()

    # --- Console handler ---------------------------------------------------
    console_handler = logging.StreamHandler(stream=sys.stdout)
    if settings.log_json:
        console_formatter: logging.Formatter = jsonlogger.JsonFormatter(
            fmt=("%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s"),
            rename_fields={"asctime": "timestamp", "levelname": "level"},
        )
    else:
        console_formatter = logging.Formatter(_CONSOLE_FORMAT)
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(request_id_filter)
    root_logger.addHandler(console_handler)

    # --- Rotating file handler ----------------------------------------------
    file_path = settings.log_dir / LOG_FILE_NAME
    file_handler = RotatingFileHandler(
        filename=file_path,
        maxBytes=LOG_FILE_MAX_BYTES,
        backupCount=LOG_FILE_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_formatter = jsonlogger.JsonFormatter(
        fmt=("%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s"),
        rename_fields={"asctime": "timestamp", "levelname": "level"},
    )
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(request_id_filter)
    root_logger.addHandler(file_handler)

    # Third-party libraries are frequently far noisier than our own code;
    # keep them at WARNING unless the deployer explicitly wants DEBUG
    # across the board.
    if settings.log_level != "DEBUG":
        for noisy_logger in ("httpx", "httpcore", "sentence_transformers", "urllib3"):
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured (level=%s, json=%s, dir=%s)",
        settings.log_level,
        settings.log_json,
        settings.log_dir,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Thin convenience wrapper around `logging.getLogger` so call sites don't
    need to import the standard library `logging` module directly, keeping
    the option open to attach project-specific behavior here later without
    touching every call site.

    Args:
        name: Logger name, conventionally `__name__` of the calling module.
    """
    return logging.getLogger(name)
