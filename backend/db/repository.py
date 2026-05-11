from __future__ import annotations

import uuid
from math import ceil
from time import perf_counter

from sqlalchemy import delete, func, or_, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import PatientRecord, PatientUploadRecord
from models.exceptions import DatabaseError
from models.patient import PatientContext
from services.logging_config import get_logger


class PatientRepository:
    """Repository for patient records and upload audits."""

    def __init__(self, db: AsyncSession) -> None:
        """Create a repository bound to a request session.

        Args:
            db: Async SQLAlchemy session.

        Returns:
            None.

        Raises:
            None.
        """
        self._db = db
        self._logger = get_logger(__name__)

    async def upsert(self, ctx: PatientContext) -> PatientRecord:
        """Insert or update patient context.

        Args:
            ctx: Parsed patient context.

        Returns:
            Persisted patient ORM record.

        Raises:
            DatabaseError: If the database operation fails.
        """
        start = perf_counter()
        values = self._patient_values(ctx)
        self._logger.debug("repository_upsert_started", patient_id=ctx.patient_id)
        try:
            bind = self._db.get_bind()
            if bind.dialect.name == "postgresql":
                stmt = (
                    postgres_insert(PatientRecord)
                    .values(**values)
                    .on_conflict_do_update(
                        index_elements=[PatientRecord.patient_id],
                        set_={
                            key: value
                            for key, value in values.items()
                            if key not in {"patient_id", "created_at"}
                        },
                    )
                    .returning(PatientRecord)
                )
                result = await self._db.execute(stmt)
                await self._db.commit()
                record = result.scalar_one()
            else:
                existing = await self.get_by_id(ctx.patient_id)
                if existing is None:
                    record = PatientRecord(**values)
                    self._db.add(record)
                else:
                    record = existing
                    for key, value in values.items():
                        setattr(record, key, value)
                await self._db.commit()
                await self._db.refresh(record)
            self._logger.debug(
                "repository_upsert_finished",
                patient_id=ctx.patient_id,
                duration_ms=int((perf_counter() - start) * 1000),
            )
            return record
        except Exception as exc:
            await self._db.rollback()
            self._logger.error("repository_upsert_failed", patient_id=ctx.patient_id, exc_info=True)
            raise DatabaseError("Patient could not be persisted", str(exc)) from exc

    async def get_by_id(self, patient_id: str) -> PatientRecord | None:
        """Fetch patient by ID.

        Args:
            patient_id: Patient identifier.

        Returns:
            Patient ORM record or None.

        Raises:
            DatabaseError: If the database query fails.
        """
        try:
            result = await self._db.execute(
                select(PatientRecord).where(PatientRecord.patient_id == patient_id)
            )
            return result.scalar_one_or_none()
        except Exception as exc:
            self._logger.error("repository_get_by_id_failed", patient_id=patient_id, exc_info=True)
            raise DatabaseError("Patient lookup failed", str(exc)) from exc

    async def get_all(
        self,
        page: int = 1,
        page_size: int = 20,
        search: str | None = None,
    ) -> tuple[list[PatientRecord], int]:
        """Paginated patient list with optional name search.

        Args:
            page: One-based page number.
            page_size: Maximum records to return.
            search: Optional first-name, last-name, or MRN filter.

        Returns:
            Tuple of records and total matching count.

        Raises:
            DatabaseError: If the database query fails.
        """
        try:
            page = max(page, 1)
            page_size = max(min(page_size, 100), 1)
            filters = []
            if search:
                pattern = f"%{search}%"
                filters.append(
                    or_(
                        PatientRecord.first_name.ilike(pattern),
                        PatientRecord.last_name.ilike(pattern),
                        PatientRecord.mrn.ilike(pattern),
                    )
                )

            count_stmt = select(func.count()).select_from(PatientRecord)
            list_stmt = select(PatientRecord).order_by(
                PatientRecord.last_name.asc(),
                PatientRecord.first_name.asc(),
            )
            if filters:
                count_stmt = count_stmt.where(*filters)
                list_stmt = list_stmt.where(*filters)

            total_result = await self._db.execute(count_stmt)
            total = int(total_result.scalar_one())
            records_result = await self._db.execute(
                list_stmt.offset((page - 1) * page_size).limit(page_size)
            )
            return list(records_result.scalars().all()), total
        except Exception as exc:
            self._logger.error("repository_get_all_failed", search=search, exc_info=True)
            raise DatabaseError("Patient list query failed", str(exc)) from exc

    async def delete(self, patient_id: str) -> bool:
        """Delete patient by ID.

        Args:
            patient_id: Patient identifier.

        Returns:
            True if a patient was deleted, otherwise False.

        Raises:
            DatabaseError: If the database mutation fails.
        """
        try:
            result = await self._db.execute(
                delete(PatientRecord).where(PatientRecord.patient_id == patient_id)
            )
            await self._db.commit()
            return bool(result.rowcount)
        except Exception as exc:
            await self._db.rollback()
            self._logger.error("repository_delete_failed", patient_id=patient_id, exc_info=True)
            raise DatabaseError("Patient deletion failed", str(exc)) from exc

    async def record_upload(
        self,
        patient_id: str | None,
        filename: str,
        filesize: int,
        status: str,
        error: str | None = None,
        duration_ms: int | None = None,
    ) -> PatientUploadRecord:
        """Create an upload audit record.

        Args:
            patient_id: Optional patient identifier.
            filename: Original upload filename.
            filesize: Uploaded byte size.
            status: Upload status.
            error: Optional error message.
            duration_ms: Optional parse duration in milliseconds.

        Returns:
            Persisted upload audit record.

        Raises:
            DatabaseError: If the audit record cannot be persisted.
        """
        try:
            record = PatientUploadRecord(
                patient_id=patient_id,
                original_filename=filename,
                file_size_bytes=filesize,
                status=status,
                error_message=error,
                parse_duration_ms=duration_ms,
            )
            self._db.add(record)
            await self._db.commit()
            await self._db.refresh(record)
            return record
        except Exception as exc:
            await self._db.rollback()
            self._logger.error("repository_record_upload_failed", filename=filename, exc_info=True)
            raise DatabaseError("Upload audit record could not be created", str(exc)) from exc

    async def update_upload(
        self,
        upload_id: uuid.UUID,
        status: str,
        patient_id: str | None = None,
        error: str | None = None,
        duration_ms: int | None = None,
    ) -> PatientUploadRecord:
        """Update an existing upload audit record.

        Args:
            upload_id: Upload audit identifier.
            status: New upload status.
            patient_id: Optional parsed patient identifier.
            error: Optional error message.
            duration_ms: Optional parse duration in milliseconds.

        Returns:
            Updated upload audit record.

        Raises:
            DatabaseError: If the audit record cannot be updated.
        """
        try:
            result = await self._db.execute(
                select(PatientUploadRecord).where(PatientUploadRecord.id == upload_id)
            )
            record = result.scalar_one()
            record.status = status
            record.patient_id = patient_id
            record.error_message = error
            record.parse_duration_ms = duration_ms
            await self._db.commit()
            await self._db.refresh(record)
            return record
        except Exception as exc:
            await self._db.rollback()
            self._logger.error("repository_update_upload_failed", upload_id=str(upload_id), exc_info=True)
            raise DatabaseError("Upload audit record could not be updated", str(exc)) from exc

    @staticmethod
    def total_pages(total: int, page_size: int) -> int:
        """Calculate a non-negative page count.

        Args:
            total: Total matching rows.
            page_size: Page size.

        Returns:
            Number of pages.

        Raises:
            None.
        """
        if total == 0:
            return 0
        return ceil(total / page_size)

    @staticmethod
    def _patient_values(ctx: PatientContext) -> dict[str, object]:
        return {
            "patient_id": ctx.patient_id,
            "mrn": ctx.mrn,
            "first_name": ctx.first_name,
            "last_name": ctx.last_name,
            "birth_date": ctx.birth_date,
            "gender": ctx.gender,
            "context_json": ctx.model_dump(mode="json"),
            "clinical_summary": ctx.to_clinical_summary(),
            "active_conditions_count": len(ctx.active_conditions),
            "active_medications_count": len(ctx.active_medications),
            "allergies_count": len(ctx.allergies),
            "has_critical_labs": ctx.has_critical_labs,
        }
