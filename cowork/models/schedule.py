from __future__ import annotations

from datetime import datetime
from uuid import UUID

import sqlalchemy as sa
from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class Schedule(BaseSQLModel, table=True):
    __tablename__ = "schedules"

    title: str = Field(description="Display name for the schedule")
    prompt: str = Field(description="The prompt to run on each execution")
    cadence: str = Field(description="Execution cadence: once | hourly | daily | weekly")
    timezone: str = Field(default="UTC", description="IANA timezone name")
    next_run_at: datetime = Field(
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="UTC datetime of next scheduled execution",
    )
    enabled: bool = Field(default=True, description="Whether the schedule is active")
    project_id: UUID = Field(foreign_key="projects.id", description="Project context for execution")
    model: str = Field(description="Model identifier to use for execution")
    last_run_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="UTC datetime of last completed execution",
    )
    last_result_conversation_id: UUID | None = Field(
        default=None,
        foreign_key="conversations.id",
        description="Conversation created by the last execution",
    )
    last_error: str | None = Field(default=None, description="Error message from last failed run")
    missed_runs: int = Field(default=0, description="Count of runs missed while the scheduler was offline")


class ScheduleRun(BaseSQLModel, table=True):
    __tablename__ = "schedule_runs"

    schedule_id: UUID = Field(foreign_key="schedules.id", description="Parent schedule")
    started_at: datetime = Field(
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="UTC datetime when the run started",
    )
    finished_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="UTC datetime when the run finished",
    )
    duration_ms: int | None = Field(default=None, description="Wall-clock duration in milliseconds")
    status: str = Field(description="Run status: running | success | failed")
    error: str | None = Field(default=None, description="Error message if the run failed")
    conversation_id: UUID | None = Field(
        default=None,
        foreign_key="conversations.id",
        description="Conversation created during this run",
    )
    is_manual: bool = Field(default=False, description="True if triggered via run-now endpoint")
