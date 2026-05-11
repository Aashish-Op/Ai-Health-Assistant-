from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON, Uuid


class Base(DeclarativeBase):
    """Base class for all ORM mappings."""


class PatientRecord(Base):
    """Persisted patient context and prompt-ready summary."""

    __tablename__ = "patients"
    __table_args__ = (
        Index("idx_patients_last_name", "last_name"),
        Index("idx_patients_birth_date", "birth_date"),
    )

    patient_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    mrn: Mapped[str | None] = mapped_column(String(64), unique=True, nullable=True)
    first_name: Mapped[str] = mapped_column(String(128), nullable=False)
    last_name: Mapped[str] = mapped_column(String(128), nullable=False)
    birth_date: Mapped[date] = mapped_column(Date, nullable=False)
    gender: Mapped[str] = mapped_column(String(32), nullable=False)
    context_json: Mapped[dict[str, object]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
    )
    clinical_summary: Mapped[str] = mapped_column(Text, nullable=False)
    active_conditions_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    active_medications_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    allergies_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    has_critical_labs: Mapped[bool] = mapped_column(default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    uploads: Mapped[list[PatientUploadRecord]] = relationship(
        back_populates="patient",
        passive_deletes=True,
    )


class PatientUploadRecord(Base):
    """Audit record for a FHIR upload attempt."""

    __tablename__ = "patient_uploads"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("patients.patient_id", ondelete="SET NULL"),
        nullable=True,
    )
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    parse_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    patient: Mapped[PatientRecord | None] = relationship(back_populates="uploads")
