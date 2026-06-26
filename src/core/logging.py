"""Structured logging configuration using ``structlog``.

Provides JSON logs in production (machine-readable, ingestible by Loki/ELK) and
human-friendly colourised console logs during development. Every log entry is
enriched with a timestamp, level, logger name and any bound contextual fields
(e.g. ``detection_id`` or ``latency_ms``).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_CONFIGURED = False


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure the standard library and structlog logging stacks.

    Args:
        level: Minimum log level name (``DEBUG``, ``INFO``, ...).
        json_logs: When True, emit JSON lines; otherwise a colour console render.
    """
    global _CONFIGURED

    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
        force=True,
    )

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.MODULE,
                structlog.processors.CallsiteParameter.FUNC_NAME,
            }
        ),
    ]

    if json_logs:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger, configuring defaults on first use.

    Args:
        name: Optional logger name, typically ``__name__``.

    Returns:
        A bound structlog logger instance.
    """
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)
