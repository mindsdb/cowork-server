"""Short-lived browser preview leases, separate from live app launch history."""
from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

import sqlalchemy as sa
from sqlmodel import Field

from cowork.models.base import BaseSQLModel

_revision = sa.Column("revision", sa.Integer, nullable=False)


class LiveAppPreview(BaseSQLModel, table=True):
    """One of eight reusable leases per person, bound to an immutable launch.

    The service filters these cross-origin records by owner explicitly. ORM
    version checks make intent and grant consumption atomic across workers on
    both SQLite and PostgreSQL: a losing transaction cannot return a token.
    """

    __tablename__ = "live_app_previews"
    __table_args__ = (
        sa.UniqueConstraint("owner_partition_id", "owner_user_id", "position", name="uq_live_app_preview_position"),
        sa.CheckConstraint("position >= 0 AND position < 8", name="ck_live_app_preview_position"),
    )
    __mapper_args__ = {"version_id_col": _revision}

    owner_partition_id: str = Field(max_length=36)
    owner_user_id: str = Field(max_length=36)
    owner_email: str = Field(max_length=254)
    viewer_cookie: str | None = Field(default=None, sa_type=sa.Text)
    position: int
    launch_id: UUID
    origin: str = Field(max_length=256)
    intent_hash: str = Field(max_length=64, unique=True, index=True)
    grant_hash: str | None = Field(default=None, max_length=64, unique=True, index=True)
    session_hash: str | None = Field(default=None, max_length=64, unique=True, index=True)
    state: Literal["pending", "granted", "active", "revoked"] = Field(sa_type=sa.String(16))
    expires_at: datetime = Field(sa_type=sa.DateTime(timezone=True))
    revision: int = Field(default=1, sa_column=_revision)


_owner_revision = sa.Column("revision", sa.Integer, nullable=False)


class LiveAppPreviewOwner(BaseSQLModel, table=True):
    """A versioned owner row serializes lease creation with owner-wide revocation."""

    __tablename__ = "live_app_preview_owners"
    __mapper_args__ = {"version_id_col": _owner_revision}

    marker: int = 0
    revision: int = Field(default=1, sa_column=_owner_revision)
