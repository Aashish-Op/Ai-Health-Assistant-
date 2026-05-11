from __future__ import annotations

import logging
import sys

import structlog

from config import Settings, get_settings


def configure_logging(settings: Settings | None = None) -> None:
    """Configure structlog for local and deployed environments.

    Args:
        settings: Optional settings instance. If omitted, cached settings are used.

    Returns:
        None.

    Raises:
        None.
    """
    active_settings = settings or get_settings()
    level = getattr(logging, active_settings.log_level.upper(), logging.INFO)

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level, force=True)

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors: list[object] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        timestamper,
    ]

    renderer: object
    if active_settings.environment == "development":
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        environment=active_settings.environment,
        service="clinical-copilot",
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a named structured logger.

    Args:
        name: Logger namespace.

    Returns:
        A structlog bound logger.

    Raises:
        None.
    """
    return structlog.get_logger(name)
