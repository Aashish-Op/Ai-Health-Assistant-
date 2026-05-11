from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from config import Settings, get_settings
from models.exceptions import DatabaseError
from services.logging_config import get_logger

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_logger = get_logger(__name__)


def init_db(settings: Settings | None = None) -> None:
    """Initialise the async SQLAlchemy engine and session factory.

    Args:
        settings: Optional settings instance. If omitted, cached settings are used.

    Returns:
        None.

    Raises:
        None.
    """
    global _engine, _session_factory

    active_settings = settings or get_settings()
    engine_kwargs: dict[str, object] = {
        "echo": active_settings.log_level.upper() == "DEBUG",
        "pool_pre_ping": True,
    }
    if not active_settings.database_url.startswith("sqlite"):
        engine_kwargs.update(
            {
                "pool_size": 10,
                "max_overflow": 20,
                "pool_recycle": 3600,
            }
        )

    _engine = create_async_engine(active_settings.database_url, **engine_kwargs)
    _session_factory = async_sessionmaker(
        bind=_engine,
        expire_on_commit=False,
        class_=AsyncSession,
    )
    _logger.debug("database_initialized")


async def dispose_db() -> None:
    """Dispose of the async database engine.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    global _engine, _session_factory

    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
    _logger.debug("database_disposed")


def get_engine() -> AsyncEngine:
    """Return the current async engine, initialising it when needed.

    Args:
        None.

    Returns:
        Async SQLAlchemy engine.

    Raises:
        DatabaseError: If engine initialisation fails.
    """
    if _engine is None:
        try:
            init_db()
        except Exception as exc:
            raise DatabaseError("Database engine could not be initialised", str(exc)) from exc
    if _engine is None:
        raise DatabaseError("Database engine is unavailable")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the current async session factory.

    Args:
        None.

    Returns:
        Async SQLAlchemy sessionmaker.

    Raises:
        DatabaseError: If the session factory cannot be initialised.
    """
    if _session_factory is None:
        get_engine()
    if _session_factory is None:
        raise DatabaseError("Database session factory is unavailable")
    return _session_factory


async def get_db() -> AsyncIterator[AsyncSession]:
    """Yield a request-scoped async database session.

    Args:
        None.

    Returns:
        Async iterator yielding one session.

    Raises:
        DatabaseError: If session creation fails.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def check_db_connection() -> bool:
    """Check whether PostgreSQL accepts a simple query.

    Args:
        None.

    Returns:
        True when SELECT 1 succeeds.

    Raises:
        None.
    """
    try:
        engine = get_engine()
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        _logger.error("database_healthcheck_failed", error=str(exc))
        return False
