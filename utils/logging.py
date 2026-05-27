"""Structured logging with rich console output + optional file sink.

Every entry point should call `setup_logging()` once at startup. Use
`get_logger(__name__)` everywhere else.
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler

_FORMAT = "%(message)s"
_FILE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_ROOT_NAME = "vcr"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
) -> Logger:
    """Configure the root `vcr` logger with rich console and optional file sink.

    Idempotent — repeated calls replace handlers rather than stacking them.
    """
    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(level)
    root.handlers.clear()
    root.propagate = False

    console = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        markup=True,
        log_time_format="[%X]",
    )
    console.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        root.addHandler(file_handler)

    return root


def get_logger(name: str) -> Logger:
    """Return a child logger of the `vcr` root."""
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
