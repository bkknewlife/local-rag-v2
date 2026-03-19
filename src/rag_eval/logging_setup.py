"""Centralized logging configuration for the RAG evaluation platform.

Provides:
  - Console handler (RichHandler) for human-friendly terminal output
  - Rotating file handler for persistent application-level logs
  - Per-run file handler that can be attached/detached for evaluation or
    ingestion sessions

Log directory: ``data/logs/``
Per-run logs:  ``data/results/<run_id>.log``

Usage from entry points::

    from rag_eval.logging_setup import setup_logging, attach_run_log, detach_run_log
    setup_logging(verbose=True)            # call once at startup
    handler = attach_run_log("my-run-1")   # optional: per-run file
    ...
    detach_run_log(handler)                # cleanup
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOG_DIR = _PROJECT_ROOT / "data" / "logs"
_APP_LOG = _LOG_DIR / "rag_eval.log"
_RESULTS_DIR = _PROJECT_ROOT / "data" / "results"

_FILE_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"
_initialized = False


def setup_logging(verbose: bool = False) -> None:
    """Configure the root logger with console + rotating file handlers.

    Safe to call multiple times; only the first call takes effect.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    level = logging.DEBUG if verbose else logging.INFO
    root.setLevel(level)

    # Console: RichHandler for pretty terminal output
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        markup=True,
    )
    console_handler.setLevel(level)
    root.addHandler(console_handler)

    # Rotating file: application-level log (10 MB x 5 backups)
    file_handler = RotatingFileHandler(
        _APP_LOG,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_FILE_DATEFMT))
    root.addHandler(file_handler)

    # Quiet noisy libraries
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "posthog",
                  "sentence_transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def attach_run_log(run_id: str, directory: Path | None = None) -> logging.FileHandler:
    """Add a per-run file handler to the root logger.

    Returns the handler so the caller can remove it later with
    ``detach_run_log()``.
    """
    out_dir = directory or _RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{run_id}.log"

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_FILE_DATEFMT))
    logging.getLogger().addHandler(handler)
    logging.getLogger(__name__).info("Per-run log started: %s", log_path)
    return handler


def detach_run_log(handler: logging.FileHandler) -> None:
    """Remove a per-run file handler and close its file."""
    logging.getLogger().removeHandler(handler)
    handler.close()
