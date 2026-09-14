from __future__ import annotations

from datetime import UTC, datetime

import sqlalchemy as sa
from sqlmodel import Field, SQLModel


class CodeControlRecord(SQLModel, table=True):
    """Tenant-namespaced durable record for the Code control plane.

    Execution contracts evolve more quickly than the surrounding Cowork data
    model.  Keeping their versioned Pydantic payload intact gives migrations a
    stable envelope while the composite key and namespace preserve strict
    tenant isolation and efficient collection scans.
    """

    __tablename__ = "code_control_records"
    __table_args__ = (
        sa.Index(
            "ix_code_control_records_namespace_collection",
            "namespace_id",
            "collection",
            "updated_at",
        ),
        sa.Index(
            "ix_code_control_records_run_queue",
            "namespace_id",
            "collection",
            "assigned_computer_id",
            "lifecycle_status",
            "created_at",
        ),
        sa.Index(
            "ix_code_control_records_parent",
            "namespace_id",
            "collection",
            "parent_id",
            "updated_at",
        ),
    )

    namespace_id: str = Field(primary_key=True, max_length=128)
    collection: str = Field(primary_key=True, max_length=32)
    document_id: str = Field(primary_key=True, max_length=160)
    payload: dict = Field(sa_column=sa.Column(sa.JSON, nullable=False))
    # Scheduler projections keep lease claims indexable without duplicating the
    # complete versioned TaskRun contract into a second persistence model.
    assigned_computer_id: str | None = Field(default=None, max_length=128)
    lifecycle_status: str | None = Field(default=None, max_length=32)
    # Run/task ownership is projected out of the JSON payload so 20 Hz runtime
    # polling never degrades into a full tenant collection scan.
    parent_id: str | None = Field(default=None, max_length=160)
    revision: int = Field(default=1, ge=1)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=sa.DateTime(timezone=True),  # type: ignore[arg-type]
        sa_column_kwargs={"server_default": sa.func.now()},
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=sa.DateTime(timezone=True),  # type: ignore[arg-type]
        sa_column_kwargs={"server_default": sa.func.now(), "onupdate": sa.func.now()},
    )
