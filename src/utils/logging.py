"""Structured logging for the pipeline.

Provides JSON-formatted logs with component tagging and timing context.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "component": getattr(record, "component", record.name),
            "msg": record.getMessage(),
        }
        # Include extra fields
        for key in ("model", "latency_ms", "vram_mb", "items", "metric"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


class ConsoleFormatter(logging.Formatter):
    """Readable colored console output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        component = getattr(record, "component", record.name)
        msg = record.getMessage()

        extras = []
        for key in ("model", "latency_ms", "vram_mb", "items"):
            val = getattr(record, key, None)
            if val is not None:
                extras.append(f"{key}={val}")
        extra_str = f" [{', '.join(extras)}]" if extras else ""

        return f"{color}{record.levelname:8s}{self.RESET} {component:20s} | {msg}{extra_str}"


def get_logger(
    name: str,
    level: str = "INFO",
    json_output: bool = False,
) -> logging.Logger:
    """Create a configured logger for a pipeline component.

    Args:
        name: Logger name (typically the component name).
        level: Logging level.
        json_output: If True, use JSON formatting. Otherwise readable console.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(f"pipeline.{name}")

    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    handler = logging.StreamHandler(sys.stderr)
    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(ConsoleFormatter())
    logger.addHandler(handler)

    return logger


def log_with_context(
    logger: logging.Logger,
    level: str,
    msg: str,
    **kwargs: Any,
) -> None:
    """Log a message with extra context fields.

    Args:
        logger: Logger instance.
        level: Log level string.
        msg: Log message.
        **kwargs: Extra fields (component, model, latency_ms, vram_mb, items, metric).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.log(log_level, msg, extra=kwargs)


class TimingContext:
    """Context manager for timing code blocks.

    Usage:
        with TimingContext("ner_inference") as t:
            result = model(input)
        print(f"Took {t.elapsed_ms:.1f} ms")
    """

    def __init__(self, name: str, sync_cuda: bool = True) -> None:
        self.name = name
        self.sync_cuda = sync_cuda
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> TimingContext:
        if self.sync_cuda:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except ImportError:
                pass
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        if self.sync_cuda:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except ImportError:
                pass
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
