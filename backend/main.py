from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from time import perf_counter

import structlog
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import get_settings
from db.session import dispose_db, init_db
from models.api import ErrorResponse
from models.exceptions import (
    ClinicalCopilotError,
    DatabaseError,
    DuplicatePatientError,
    FHIRParseError,
    FileTooLargeError,
    InvalidFileTypeError,
    PatientNotFoundError,
    RetrievalServiceUnavailableError,
)
from routers import fhir, health, patients, query
from services.embedder import EmbeddingService
from services.logging_config import configure_logging, get_logger
from services.pinecone_service import PineconeService
from services.reranker import RerankerService
from services.retrieval_service import RetrievalService

configure_logging()
_logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown resources.

    Args:
        _: FastAPI application instance.

    Returns:
        Async iterator for the lifespan context.

    Raises:
        None.
    """
    settings = get_settings()
    configure_logging(settings)
    init_db(settings)
    configure_retrieval_services(app)
    _logger.info("application_started")
    try:
        yield
    finally:
        await dispose_db()
        _logger.info("application_stopped")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        None.

    Returns:
        Configured FastAPI app.

    Raises:
        pydantic.ValidationError: If settings are invalid.
    """
    settings = get_settings()
    app = FastAPI(
        title="FHIR-Integrated Clinical Copilot",
        description=(
            "Phase 1 data foundation for parsing Synthea FHIR R4 patient bundles, "
            "persisting clinical context, and serving patient records."
        ),
        version="0.1.0",
        contact={"name": "MedConnect Clinical Copilot Team"},
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_middleware(app)
    register_exception_handlers(app)
    app.include_router(fhir.router, prefix="/fhir")
    app.include_router(patients.router, prefix="/patients")
    app.include_router(query.router, prefix="/query")
    app.include_router(health.router, prefix="/health")
    return app


def configure_retrieval_services(app: FastAPI) -> None:
    """Create retrieval services once per application lifetime.

    Args:
        app: FastAPI application instance.

    Returns:
        None.

    Raises:
        None.
    """
    settings = get_settings()
    app.state.retrieval_service = None
    app.state.pinecone_service = None
    missing = settings.missing_retrieval_settings()
    if missing:
        _logger.warning("retrieval_services_not_configured", missing_settings=missing)
        return
    try:
        embedder = EmbeddingService(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
        pinecone = PineconeService(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
        )
        reranker = RerankerService(
            api_key=settings.cohere_api_key,
            model=settings.reranker_model,
            top_n=settings.reranker_top_n,
        )
        app.state.pinecone_service = pinecone
        app.state.retrieval_service = RetrievalService(
            embedder=embedder,
            pinecone=pinecone,
            reranker=reranker,
            retrieval_top_k=settings.retrieval_top_k,
        )
    except Exception as exc:
        app.state.retrieval_service = None
        app.state.pinecone_service = None
        _logger.error("retrieval_services_failed_to_start", error=str(exc))


def register_middleware(app: FastAPI) -> None:
    """Register request context and structured logging middleware.

    Args:
        app: FastAPI application instance.

    Returns:
        None.

    Raises:
        None.
    """

    @app.middleware("http")
    async def request_context_middleware(
        request: Request,
        call_next: Callable[[Request], object],
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        start = perf_counter()
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception:
            _logger.error(
                "request_failed",
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_ms=int((perf_counter() - start) * 1000),
                exc_info=True,
            )
            raise
        finally:
            _logger.info(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_ms=int((perf_counter() - start) * 1000),
            )
            structlog.contextvars.unbind_contextvars("request_id")


def register_exception_handlers(app: FastAPI) -> None:
    """Register structured exception handlers.

    Args:
        app: FastAPI application instance.

    Returns:
        None.

    Raises:
        None.
    """
    app.add_exception_handler(FHIRParseError, _domain_exception_handler(status.HTTP_422_UNPROCESSABLE_ENTITY))
    app.add_exception_handler(PatientNotFoundError, _domain_exception_handler(status.HTTP_404_NOT_FOUND))
    app.add_exception_handler(DatabaseError, _domain_exception_handler(status.HTTP_503_SERVICE_UNAVAILABLE))
    app.add_exception_handler(DuplicatePatientError, _domain_exception_handler(status.HTTP_409_CONFLICT))
    app.add_exception_handler(InvalidFileTypeError, _domain_exception_handler(status.HTTP_422_UNPROCESSABLE_ENTITY))
    app.add_exception_handler(FileTooLargeError, _domain_exception_handler(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE))
    app.add_exception_handler(RetrievalServiceUnavailableError, _domain_exception_handler(status.HTTP_503_SERVICE_UNAVAILABLE))
    app.add_exception_handler(ClinicalCopilotError, _domain_exception_handler(status.HTTP_400_BAD_REQUEST))

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = _request_id(request)
        _logger.error("unhandled_exception", request_id=request_id, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error_code="INTERNAL_SERVER_ERROR",
                message="An unexpected error occurred",
                detail=str(exc),
                request_id=request_id,
            ).model_dump(),
        )


def _domain_exception_handler(status_code: int) -> Callable[[Request, ClinicalCopilotError], object]:
    async def handler(request: Request, exc: ClinicalCopilotError) -> JSONResponse:
        request_id = _request_id(request)
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(
                error_code=exc.error_code,
                message=exc.message,
                detail=exc.detail,
                request_id=request_id,
            ).model_dump(),
        )

    return handler


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


app = create_app()
