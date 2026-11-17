from __future__ import annotations

from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy import Index
from sqlmodel import Field

from .base import BaseSQLModel


class Notice(BaseSQLModel, table=True):
    """One entry in a person's notification inbox (cowork.services.notices).

    Features post notices; the inbox lists them. A notice stores what
    happened (``kind``), what it is about (``subject``, such as a task's
    name) and what opening it shows (``target``). The client turns the kind
    and subject into text in the reader's language, so the server never
    stores display copy.

    Owned like the schedules that post them: ``org_id`` is the organization
    in org mode or the person's partition on a trusted-identity deployment,
    and NULL on the desktop; ``user_id`` is the person it is for.
    """

    __tablename__ = "notices"

    org_id: str | None = Field(default=None, max_length=36, description="Organization or partition; NULL on the desktop")
    user_id: str | None = Field(default=None, max_length=36, description="Person the notice is for; NULL on the desktop")
    kind: str = Field(max_length=64, description="What happened, e.g. scheduled_run.failed")
    subject: str = Field(default="", max_length=255, description="What it is about, e.g. the task's name")
    target: dict[str, Any] = Field(
        default_factory=dict,
        sa_type=sa.JSON,  # type: ignore
        description="What opening the notice shows; its keys depend on the kind",
    )
    source_id: str | None = Field(
        default=None, max_length=64,
        description="The record that posted it, such as a run id, so one event posts one notice",
    )
    read_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="When the person read it; NULL while unread",
    )

    __table_args__ = (
        Index("ix_notices_org_id_user_id_created_at", "org_id", "user_id", "created_at"),
        Index("ix_notices_kind_source_id", "kind", "source_id"),
        # One ending per run, even when two reapers post concurrently.
        Index(
            "uq_notices_scheduled_run_source", "source_id", unique=True,
            sqlite_where=sa.text("kind IN ('scheduled_run.failed', 'scheduled_run.stopped', 'scheduled_run.succeeded')"),
            postgresql_where=sa.text("kind IN ('scheduled_run.failed', 'scheduled_run.stopped', 'scheduled_run.succeeded')"),
        ),
    )
