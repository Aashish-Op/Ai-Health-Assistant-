from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from config import get_settings
from db.session import check_db_connection
from models.api import DBHealthResponse, HealthResponse

router = APIRouter(tags=["health"])
_app_start_time = datetime.utcnow()


@router.get("", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return process health without touching dependencies.

    Args:
        None.

    Returns:
        Application health response.

    Raises:
        None.
    """
    settings = get_settings()
    uptime = (datetime.utcnow() - _app_start_time).total_seconds()
    return HealthResponse(
        status="ok",
        version="0.1.0",
        environment=settings.environment,
        uptime_seconds=uptime,
    )


@router.get("/ready", response_model=DBHealthResponse)
async def ready(request: Request) -> DBHealthResponse | JSONResponse:
    """Return readiness based on database connectivity.

    Args:
        None.

    Returns:
        DB readiness response or a 503 JSON response.

    Raises:
        None.
    """
    postgres_ok = await check_db_connection()
    pinecone_ok = False
    pinecone = getattr(request.app.state, "pinecone_service", None)
    if pinecone is not None:
        try:
            await pinecone.get_index_stats()
            pinecone_ok = True
        except Exception:
            pinecone_ok = False
    response = DBHealthResponse(
        postgres=postgres_ok,
        pinecone=pinecone_ok,
        message=_readiness_message(postgres_ok, pinecone_ok),
    )
    if postgres_ok and pinecone_ok:
        return response
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=response.model_dump(),
    )


def _readiness_message(postgres: bool, pinecone: bool) -> str:
    if postgres and pinecone:
        return "postgres and pinecone reachable"
    if not postgres and not pinecone:
        return "postgres and pinecone unreachable"
    if not postgres:
        return "postgres unreachable"
    return "pinecone unreachable"
