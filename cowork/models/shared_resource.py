from __future__ import annotations

from datetime import datetime

import sqlalchemy as sa
from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class SharedResourceAttribution(BaseSQLModel, table=True):
    """Server-owned authorship metadata for org-shared resources.

    Some shared resources live on disk rather than in SQL (skills, project
    memory, and project instructions).  Keeping their authorization identity in
    SQL prevents an agent with write access to the project volume from forging
    ownership by editing a sidecar file.
    """

    __tablename__ = "shared_resource_attributions"
    __table_args__ = (
        sa.UniqueConstraint(
            "org_id",
            "resource_kind",
            "resource_key",
            name="uq_shared_resource_attribution_key",
        ),
    )

    org_id: str = Field(index=True, max_length=36)
    resource_kind: str = Field(index=True, max_length=32)
    resource_key: str = Field(max_length=255)
    created_by_id: str | None = Field(default=None, max_length=36)
    created_by_email: str | None = Field(default=None, max_length=320)
    updated_by_id: str | None = Field(default=None, max_length=36)
    updated_by_email: str | None = Field(default=None, max_length=320)
    pending_claim_token: str | None = Field(default=None, max_length=36)
    pending_claim_expires_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
    )


class SharedResourceMutation(BaseSQLModel, table=True):
    """Append-only audit event for an allowed shared-resource mutation."""

    __tablename__ = "shared_resource_mutations"
    __table_args__ = (
        sa.Index(
            "ix_shared_resource_mutations_resource",
            "org_id",
            "resource_kind",
            "resource_key",
        ),
    )

    org_id: str = Field(index=True, max_length=36)
    resource_kind: str = Field(max_length=32)
    resource_key: str = Field(max_length=255)
    action: str = Field(max_length=32)
    actor_id: str = Field(max_length=36)
    actor_email: str | None = Field(default=None, max_length=320)
