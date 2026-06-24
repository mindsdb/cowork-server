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
    missed_run_policy: str = Field(
        default="skip",
        sa_column_kwargs={"server_default": "skip"},
        description="What to do with occurrences missed while offline: skip | run_once | catch_up",
    )
    consecutive_failures: int = Field(
        default=0,
        sa_column_kwargs={"server_default": sa.text("0")},
        description="Number of consecutive failed runs; reset to 0 on any success",
    )
    health: str = Field(
        default="ok",
        sa_column_kwargs={"server_default": "ok"},
        description="Rolled-up health: ok | failing | paused",
    )


class ScheduleRun(BaseSQLModel, table=True):
    __tablename__ = "schedule_runs"
    __table_args__ = (
        # A given occurrence (scheduled or manual) is keyed once per schedule;
        # this is the DB-level guard that stops a double-fire.
        sa.UniqueConstraint("schedule_id", "idempotency_key", name="uq_schedule_run_idempotency"),
    )

    schedule_id: UUID = Field(foreign_key="schedules.id", description="Parent schedule")
    idempotency_key: str | None = Field(
        default=None,
        description="Stable key for the occurrence this run satisfies; unique per schedule",
    )
    attempts: int = Field(
        default=1,
        sa_column_kwargs={"server_default": sa.text("1")},
        description="How many in-process attempts this run took (retries on transient failure)",
    )
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
