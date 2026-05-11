from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from db.repository import PatientRepository
from db.session import get_db
from models.api import RetrieveRequest, RetrieveResponse
from models.exceptions import PatientNotFoundError, RetrievalServiceUnavailableError
from models.patient import PatientContext
from services.logging_config import get_logger
from services.retrieval_service import RetrievalService

router = APIRouter(tags=["query"])
_logger = get_logger(__name__)


def get_retrieval_service(request: Request) -> RetrievalService:
    """Return the application-scoped retrieval service.

    Args:
        request: Active FastAPI request.

    Returns:
        Retrieval service from app state.

    Raises:
        RetrievalServiceUnavailableError: If retrieval is not configured or failed at startup.
    """
    service = getattr(request.app.state, "retrieval_service", None)
    if service is None:
        missing = get_settings().missing_retrieval_settings()
        detail = "missing settings: " + ", ".join(missing) if missing else "service startup failed"
        _logger.warning("retrieval_service_unavailable", detail=detail)
        raise RetrievalServiceUnavailableError(detail)
    return cast(RetrievalService, service)


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(
    request_body: RetrieveRequest,
    service: RetrievalService = Depends(get_retrieval_service),
    db: AsyncSession = Depends(get_db),
) -> RetrieveResponse:
    """Retrieve literature evidence for a patient-specific clinical question.

    Args:
        request_body: Retrieval request body.
        service: Application-scoped retrieval service.
        db: Request-scoped database session.

    Returns:
        Structured retrieval response.

    Raises:
        PatientNotFoundError: If the patient does not exist.
        RetrievalServiceUnavailableError: If retrieval dependencies are unavailable.
    """
    repo = PatientRepository(db)
    record = await repo.get_by_id(request_body.patient_id)
    if record is None:
        raise PatientNotFoundError(request_body.patient_id)
    patient_context = PatientContext.model_validate(record.context_json)
    result = await service.retrieve(
        question=request_body.question,
        patient_context=patient_context,
        top_k=request_body.top_k,
    )
    return RetrieveResponse(
        patient_id=request_body.patient_id,
        question=request_body.question,
        expanded_query=result.expanded_query,
        results=result.chunks,
        retrieval_duration_ms=result.retrieval_duration_ms,
        reranking_duration_ms=result.reranking_duration_ms,
        total_candidates=result.total_candidates,
    )


@router.get("/index/stats")
async def index_stats(request: Request) -> dict[str, object]:
    """Return Pinecone index statistics.

    Args:
        request: Active FastAPI request.

    Returns:
        Normalized Pinecone index stats.

    Raises:
        RetrievalServiceUnavailableError: If Pinecone is unavailable.
    """
    pinecone = getattr(request.app.state, "pinecone_service", None)
    if pinecone is None:
        raise RetrievalServiceUnavailableError("Pinecone service is unavailable")
    stats = await pinecone.get_index_stats()
    return {
        "vector_count": int(stats.get("total_vector_count", 0) or 0),
        "dimension": int(stats.get("dimension", 0) or 0),
        "index_fullness": float(stats.get("index_fullness", 0.0) or 0.0),
        "namespaces": stats.get("namespaces", {}),
    }
