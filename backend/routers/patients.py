from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from db.repository import PatientRepository
from db.session import get_db
from models.api import PatientListItem, PatientListResponse
from models.exceptions import PatientNotFoundError
from models.patient import PatientContext

router = APIRouter(tags=["patients"])


@router.get("", response_model=PatientListResponse)
async def list_patients(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    search: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> PatientListResponse:
    """Return a paginated patient list.

    Args:
        page: One-based page number.
        page_size: Maximum records per page.
        search: Optional name or MRN search term.
        db: Request-scoped database session.

    Returns:
        Paginated patient list response.

    Raises:
        DatabaseError: If the query fails.
    """
    repo = PatientRepository(db)
    records, total = await repo.get_all(page=page, page_size=page_size, search=search)
    items: list[PatientListItem] = []
    for record in records:
        ctx = PatientContext.model_validate(record.context_json)
        items.append(
            PatientListItem(
                patient_id=record.patient_id,
                full_name=ctx.full_name,
                age=ctx.age,
                gender=record.gender,
                conditions_count=record.active_conditions_count,
                medications_count=record.active_medications_count,
                has_critical_labs=record.has_critical_labs,
            )
        )
    return PatientListResponse(
        patients=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=repo.total_pages(total, page_size),
    )


@router.get("/{patient_id}", response_model=PatientContext)
async def get_patient(patient_id: str, db: AsyncSession = Depends(get_db)) -> PatientContext:
    """Return the full parsed patient context.

    Args:
        patient_id: Patient identifier.
        db: Request-scoped database session.

    Returns:
        Parsed patient context.

    Raises:
        PatientNotFoundError: If the patient does not exist.
    """
    repo = PatientRepository(db)
    record = await repo.get_by_id(patient_id)
    if record is None:
        raise PatientNotFoundError(patient_id)
    return PatientContext.model_validate(record.context_json)


@router.get("/{patient_id}/summary")
async def get_patient_summary(patient_id: str, db: AsyncSession = Depends(get_db)) -> dict[str, str]:
    """Return the pre-computed clinical summary for a patient.

    Args:
        patient_id: Patient identifier.
        db: Request-scoped database session.

    Returns:
        Patient ID and clinical summary text.

    Raises:
        PatientNotFoundError: If the patient does not exist.
    """
    repo = PatientRepository(db)
    record = await repo.get_by_id(patient_id)
    if record is None:
        raise PatientNotFoundError(patient_id)
    return {"patient_id": patient_id, "clinical_summary": record.clinical_summary}


@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_patient(patient_id: str, db: AsyncSession = Depends(get_db)) -> Response:
    """Delete a patient record.

    Args:
        patient_id: Patient identifier.
        db: Request-scoped database session.

    Returns:
        Empty 204 response when deleted.

    Raises:
        PatientNotFoundError: If the patient does not exist.
    """
    repo = PatientRepository(db)
    deleted = await repo.delete(patient_id)
    if not deleted:
        raise PatientNotFoundError(patient_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
