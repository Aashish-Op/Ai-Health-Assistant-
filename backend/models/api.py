from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from models.retrieval import RetrievedChunk


class FrozenAPIModel(BaseModel):
    """Base immutable API schema."""

    model_config = ConfigDict(frozen=True)


class UploadPatientResponse(FrozenAPIModel):
    patient_id: str
    status: str
    parsed_at: datetime
    conditions_count: int
    medications_count: int
    allergies_count: int
    labs_count: int
    has_critical_labs: bool
    parse_duration_ms: int


class PatientListItem(FrozenAPIModel):
    patient_id: str
    full_name: str
    age: int
    gender: str
    conditions_count: int
    medications_count: int
    has_critical_labs: bool


class PatientListResponse(FrozenAPIModel):
    patients: list[PatientListItem]
    total: int
    page: int
    page_size: int
    total_pages: int


class ErrorResponse(FrozenAPIModel):
    error_code: str
    message: str
    detail: str | None = None
    request_id: str | None = None


class HealthResponse(FrozenAPIModel):
    status: str
    version: str
    environment: str
    uptime_seconds: float


class DBHealthResponse(FrozenAPIModel):
    postgres: bool
    pinecone: bool = False
    message: str


class RetrieveRequest(FrozenAPIModel):
    patient_id: str
    question: str
    top_k: int = 5


class RetrieveResponse(FrozenAPIModel):
    patient_id: str
    question: str
    expanded_query: str
    results: list[RetrievedChunk]
    retrieval_duration_ms: int
    reranking_duration_ms: int
    total_candidates: int
