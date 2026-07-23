from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from pydantic import BaseModel
from sqlalchemy import JSON
from sqlmodel import Column, Field

from cowork.models.base import BaseSQLModel


class Approval(BaseSQLModel, table=True):
    """A consequential action parked for human review (approve-before-act).

    The descriptor is a versioned discriminated union (schemas/approvals.py)
    carrying EXACTLY what executes on approve — the harness runs it
    deterministically, so executed == approved. Receipts are written after
    execution, not derived: what I'll do → what you approved → what happened.
    """

    __tablename__ = "approvals"

    conversation_id: UUID = Field(
        foreign_key="conversations.id",
        index=True,
        description="Conversation the proposal belongs to",
    )
    kind: str = Field(description="Approval kind: action | auth (more kinds ride the versioned descriptor)")
    status: str = Field(
        default="pending",
        index=True,
        description="pending | resolving | approved | edited | skipped | expired | failed (re-resolvable)",
    )
    action_descriptor: dict[str, Any] | BaseModel | str | list[Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON),
        description="Versioned action descriptor (union v1) — exactly what executes on approve",
    )
    draft: str = Field(
        default="",
        sa_column=Column(sa.Text),
        description="User-inspectable draft content (e.g. the email body being approved)",
    )
    receipt: dict[str, Any] | BaseModel | str | list[Any] | None = Field(
        default=None,
        sa_column=Column(JSON),
        description="What actually happened, written after execution (artifact refs, diffs)",
    )
    ttl_seconds: int = Field(
        default=259200,
        description="Time-to-live in seconds before the sweep expires it (default 72h)",
    )
    expires_at: datetime = Field(
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        index=True,
        description="UTC expiry instant — swept to status=expired by the scheduler",
    )
    resolved_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="UTC instant resolved (approved/edited/skipped/expired)",
    )
