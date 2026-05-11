from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from config import get_settings
from db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url() -> str:
    """Return the database URL for Alembic.

    Args:
        None.

    Returns:
        Async SQLAlchemy database URL.

    Raises:
        pydantic.ValidationError: If settings are invalid.
    """
    return get_settings().database_url


def run_migrations_offline() -> None:
    """Run migrations without opening a database connection.

    Args:
        None.

    Returns:
        None.

    Raises:
        alembic.util.CommandError: If migration configuration fails.
    """
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations on an existing synchronous migration connection.

    Args:
        connection: SQLAlchemy migration connection.

    Returns:
        None.

    Raises:
        alembic.util.CommandError: If migration execution fails.
    """
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations through an AsyncEngine.

    Args:
        None.

    Returns:
        None.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: If database migration fails.
    """
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_url()

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_async_migrations())
