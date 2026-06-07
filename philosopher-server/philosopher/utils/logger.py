"""Structured logging setup using structlog."""
from __future__ import annotations

import logging
import sys

import structlog


def configure(level: str = "INFO", fmt: str = "rich") -> None:
    handlers = []
    if fmt == "rich":
        try:
            from rich.logging import RichHandler
            h = RichHandler(rich_tracebacks=True, show_path=False)
            h.setFormatter(logging.Formatter("%(message)s"))
        except ImportError:
            h = logging.StreamHandler(sys.stdout)
    else:
        h = logging.StreamHandler(sys.stdout)
    handlers.append(h)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=handlers, force=True,
    )
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    return structlog.get_logger(name)
