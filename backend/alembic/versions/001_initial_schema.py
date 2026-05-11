from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the Phase 1 patient storage schema."""
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')

    op.create_table(
        "patients",
        sa.Column("patient_id", sa.String(length=64), primary_key=True),
        sa.Column("mrn", sa.String(length=64), nullable=True, unique=True),
        sa.Column("first_name", sa.String(length=128), nullable=False),
        sa.Column("last_name", sa.String(length=128), nullable=False),
        sa.Column("birth_date", sa.Date(), nullable=False),
        sa.Column("gender", sa.String(length=32), nullable=False),
        sa.Column("context_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("clinical_summary", sa.Text(), nullable=False),
        sa.Column("active_conditions_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("active_medications_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("allergies_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("has_critical_labs", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("idx_patients_last_name", "patients", ["last_name"])
    op.create_index("idx_patients_birth_date", "patients", ["birth_date"])
    op.create_index("idx_patients_context_json_gin", "patients", ["context_json"], postgresql_using="gin")

    op.create_table(
        "patient_uploads",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("patient_id", sa.String(length=64), nullable=True),
        sa.Column("original_filename", sa.String(length=512), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("parse_duration_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["patient_id"], ["patients.patient_id"], ondelete="SET NULL"),
    )
    op.create_index("idx_patient_uploads_patient_id", "patient_uploads", ["patient_id"])
    op.create_index("idx_patient_uploads_status", "patient_uploads", ["status"])


def downgrade() -> None:
    """Drop the Phase 1 patient storage schema."""
    op.drop_index("idx_patient_uploads_status", table_name="patient_uploads")
    op.drop_index("idx_patient_uploads_patient_id", table_name="patient_uploads")
    op.drop_table("patient_uploads")
    op.drop_index("idx_patients_context_json_gin", table_name="patients", postgresql_using="gin")
    op.drop_index("idx_patients_birth_date", table_name="patients")
    op.drop_index("idx_patients_last_name", table_name="patients")
    op.drop_table("patients")
