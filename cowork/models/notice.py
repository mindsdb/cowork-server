from __future__ import annotations

from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy import Index
from sqlmodel import Field

from .base import BaseSQLModel


class Notice(BaseSQLModel, table=True):
    """One entry in a person's in-app notification inbox (cowork.services.notices).

    A feature posts a notice; the app lists it under the bell until the person
    reads it. ``kind`` says what happened and ``data`` carries what the app
    needs to word it and open its target, so a new kind needs no change here.
    ``title`` names the subject (a task's name), never an error: notices can
    show on a locked screen.

    Owner columns are NULL on the desktop, which has one person.
    """

    __tablename__ = "notices"

    org_id: str | None = Field(default=None, max_length=36, description="Owner's organization or partition; NULL on desktop")
    user_id: str | None = Field(default=None, max_length=36, description="Person the notice is for; NULL on desktop")
    kind: str = Field(max_length=64, description="What happened, e.g. scheduled_run.failed")
    title: str = Field(max_length=255, description="The subject's name, shown as the notice's heading")
    data: dict[str, Any] = Field(
        default_factory=dict,
        sa_type=sa.JSON,  # type: ignore
        description="Kind-specific details: ids of the target, an outcome",
    )
    read_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="When the person read it; NULL while unread",
    )

    __table_args__ = (
        Index("ix_notices_org_id_user_id_created_at", "org_id", "user_id", "created_at"),
    )
