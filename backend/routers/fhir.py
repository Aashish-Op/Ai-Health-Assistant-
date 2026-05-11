from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, Depends, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from db.repository import PatientRepository
from db.session import get_db
from models.api import UploadPatientResponse
from models.exceptions import FHIRParseError, FileTooLargeError, InvalidFileTypeError
from services.fhir_parser import FHIRParser
from services.logging_config import get_logger

router = APIRouter(tags=["fhir"])
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_logger = get_logger(__name__)


@router.post("/patient/load", response_model=UploadPatientResponse)
async def load_patient(file: UploadFile, db: AsyncSession = Depends(get_db)) -> UploadPatientResponse:
    """Parse and persist one FHIR patient bundle upload.

    Args:
        file: Multipart JSON upload containing a FHIR Bundle.
        db: Request-scoped database session.

    Returns:
        Parsed patient upload summary.

    Raises:
        InvalidFileTypeError: If the upload is not JSON.
        FileTooLargeError: If the upload exceeds the maximum size.
        FHIRParseError: If the FHIR payload is malformed.
    """
    filename = file.filename or "patient_bundle.json"
    content_type = file.content_type or ""
    if "application/json" not in content_type and not filename.lower().endswith(".json"):
        raise InvalidFileTypeError(filename)

    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise FileTooLargeError(MAX_UPLOAD_BYTES)

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise FileTooLargeError(MAX_UPLOAD_BYTES)

    repo = PatientRepository(db)
    upload = await repo.record_upload(
        patient_id=None,
        filename=filename,
        filesize=len(data),
        status="pending",
    )

    start = perf_counter()
    try:
        patient_context = FHIRParser.from_bytes(data).extract()
        await repo.upsert(patient_context)
        duration_ms = int((perf_counter() - start) * 1000)
        await repo.update_upload(
            upload_id=upload.id,
            status="parsed",
            patient_id=patient_context.patient_id,
            duration_ms=duration_ms,
        )
        _logger.info(
            "patient_parsed",
            patient_id=patient_context.patient_id,
            duration_ms=duration_ms,
        )
        return UploadPatientResponse(
            patient_id=patient_context.patient_id,
            status="parsed",
            parsed_at=patient_context.parsed_at,
            conditions_count=len(patient_context.active_conditions),
            medications_count=len(patient_context.active_medications),
            allergies_count=len(patient_context.allergies),
            labs_count=len(patient_context.recent_labs),
            has_critical_labs=patient_context.has_critical_labs,
            parse_duration_ms=duration_ms,
        )
    except FHIRParseError as exc:
        duration_ms = int((perf_counter() - start) * 1000)
        await repo.update_upload(
            upload_id=upload.id,
            status="failed",
            error=exc.message,
            duration_ms=duration_ms,
        )
        raise
