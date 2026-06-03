"""Lightweight structured logging + per-phase timing helpers."""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Iterator

_CONFIGURED = False


def get_logger(name: str = "specgraph") -> logging.Logger:
    """Return the package logger, configuring a console handler once."""
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        level = os.environ.get("SPECGRAPH_LOG_LEVEL", "INFO").upper()
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        root = logging.getLogger("specgraph")
        root.handlers[:] = [handler]
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    return logger


@contextmanager
def phase(name: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Log the start/end and wall-clock duration of a pipeline phase."""
    log = logger or get_logger()
    log.info("▶ %s …", name)
    start = time.perf_counter()
    try:
        yield
    except Exception:
        log.error("✖ %s failed after %.1fs", name, time.perf_counter() - start)
        raise
    else:
        log.info("✔ %s done in %.1fs", name, time.perf_counter() - start)
