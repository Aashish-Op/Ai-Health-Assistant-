from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import Settings, get_settings
from db.models import Base
from db.session import get_db
from models.patient import PatientContext
from services.fhir_parser import FHIRParser


@pytest.fixture(autouse=True)
def settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """Override app settings with test-safe values.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Iterator yielding test settings.

    Raises:
        pydantic.ValidationError: If test settings are invalid.
    """
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    monkeypatch.setenv("CORS_ORIGINS", '["http://test"]')
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def async_engine(settings: Settings) -> AsyncIterator[object]:
    """Create an async SQLite engine for tests.

    Args:
        settings: Test settings fixture.

    Returns:
        Async iterator yielding an engine.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: If engine creation fails.
    """
    engine = create_async_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def db_session(async_engine: object) -> AsyncIterator[AsyncSession]:
    """Create a clean async database session for each test.

    Args:
        async_engine: Test database engine.

    Returns:
        Async iterator yielding a database session.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: If table setup fails.
    """
    async with async_engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=async_engine,
        expire_on_commit=False,
        class_=AsyncSession,
    )
    async with session_factory() as session:
        yield session

    async with async_engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def test_app() -> object:
    """Create a FastAPI app for API tests.

    Args:
        None.

    Returns:
        FastAPI app instance.

    Raises:
        None.
    """
    from main import create_app

    return create_app()


@pytest_asyncio.fixture
async def test_client(db_session: AsyncSession, test_app: object) -> AsyncIterator[AsyncClient]:
    """Create an HTTP client with database dependency overrides.

    Args:
        db_session: Test database session.

    Returns:
        Async iterator yielding an ASGI test client.

    Raises:
        None.
    """
    async def override_get_db() -> AsyncIterator[AsyncSession]:
        yield db_session

    test_app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    test_app.dependency_overrides.clear()


@pytest.fixture
def fhir_bundle() -> dict[str, object]:
    """Load the sample FHIR bundle fixture.

    Args:
        None.

    Returns:
        Decoded sample FHIR bundle.

    Raises:
        json.JSONDecodeError: If the fixture is invalid JSON.
    """
    fixture_path = BACKEND_DIR / "tests" / "fixtures" / "sample_patient.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


@pytest.fixture
def patient_context(fhir_bundle: dict[str, object]) -> PatientContext:
    """Return a parsed sample patient context.

    Args:
        fhir_bundle: Decoded sample FHIR bundle.

    Returns:
        Parsed patient context.

    Raises:
        FHIRParseError: If fixture parsing fails.
    """
    return FHIRParser(fhir_bundle).extract()
