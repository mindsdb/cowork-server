from datetime import datetime
from uuid import UUID

import sqlalchemy as sa
from sqlmodel import Field, Relationship

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
    project_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid(),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        description="Project context for execution",
    )
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
    org_id: str | None = Field(default=None, index=True, max_length=36, description="Owning organization; NULL on local/desktop rows")
    created_by: str | None = Field(default=None, max_length=36, description="User who created the row; NULL on local/desktop rows")

    # Deleting a schedule takes its runs with it. Delete-orphan makes the ORM
    # issue the child DELETEs before the parent regardless of backend (SQLite
    # runs without FK enforcement); the schedule_id ON DELETE CASCADE below is
    # the backstop for a run the scheduler inserts concurrently under Postgres.
    runs: list["ScheduleRun"] = Relationship(
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class ScheduleRun(BaseSQLModel, table=True):
    __tablename__ = "schedule_runs"

    schedule_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid(),
            sa.ForeignKey("schedules.id", ondelete="CASCADE"),
            nullable=False,
        ),
        description="Parent schedule",
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
