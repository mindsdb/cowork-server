from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from pydantic import BaseModel
from sqlalchemy import JSON
from sqlmodel import Column, Field

from cowork.models.base import BaseSQLModel


class ApprovalToken(BaseSQLModel, table=True):
    """One-shot execution token issued when an approval resolves.

    Carries the FULL approved payload (tool + args) so what executes is
    bit-identical to what was approved — the gated tool path must present a
    token whose stored payload matches its own arguments exactly. Hash-only
    storage (the raw token is returned once, never persisted)."""

    __tablename__ = "approval_tokens"

    approval_id: UUID = Field(
        foreign_key="approvals.id",
        index=True,
        description="Approval this token executes",
    )
    token_hash: str = Field(
        sa_column=Column(sa.String(length=64), unique=True, nullable=False),
        description="SHA-256 hex of the raw token (the raw value is never stored)",
    )
    payload: dict[str, Any] | BaseModel | str | list[Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON),
        description="Bound payload: {tool, args, snapshot_v} — execution must match it exactly",
    )
    consumed_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="UTC instant the token was spent (null = still usable)",
    )
    expires_at: datetime = Field(
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="UTC instant after which the token is worthless",
    )
