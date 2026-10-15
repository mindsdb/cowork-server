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
    consecutive_failures: int = Field(
        default=0,
        sa_column_kwargs={"server_default": "0"},
        description="Cron occurrences in a row that settled failed; a success resets it",
    )
    auto_paused_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="When repeated failures paused the schedule; cleared on resume",
    )
    paused_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description=(
            "When the user paused the schedule; cleared on resume. A one-off is also "
            "disabled once it is claimed, so `enabled` alone does not mean paused"
        ),
    )
    missed_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="When a one-off was found too late to run on its own; cleared when it runs or is moved",
    )
    org_id: str | None = Field(default=None, index=True, max_length=36, description="Owning organization; NULL on local/desktop rows")
    created_by: str | None = Field(default=None, max_length=36, description="User who created the row; NULL on local/desktop rows")

    # Deleting a schedule takes its runs with it. Delete-orphan makes the ORM
    # issue the child DELETEs before the parent regardless of backend (SQLite
    # runs without FK enforcement); the schedule_id ON DELETE CASCADE below is
    # the backstop for a run the scheduler inserts concurrently under Postgres.
    runs: list["ScheduleRun"] = Relationship(
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    occurrences: list["ScheduleOccurrence"] = Relationship(
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    skipped_periods: list["ScheduleSkippedPeriod"] = Relationship(
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class ScheduleSkippedPeriod(BaseSQLModel, table=True):
    """Slots a schedule passed over without running, summarized: one row per
    time the scheduler found it behind, however many slots that covered."""

    __tablename__ = "schedule_skipped_periods"

    schedule_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid(),
            sa.ForeignKey("schedules.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        description="Parent schedule",
    )
    first_slot: datetime = Field(
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="UTC instant of the first slot passed over",
    )
    last_slot: datetime = Field(
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="UTC instant of the last slot passed over",
    )
    slot_count: int = Field(description="Slots passed over, first and last included")
    reason: str = Field(description="offline (the app was not running) | one_off_missed")


class ScheduleOccurrence(BaseSQLModel, table=True):
    """One logical firing of a schedule: a cron slot or a manual run-now.

    Each attempt at it is a ScheduleRun, so retries are attempts within one
    occurrence rather than new firings. The occurrence key makes a firing
    claimable once: a second poll, a second replica, or a manual press that
    races a cron slot loses the insert instead of running the task again.
    """

    __tablename__ = "schedule_occurrences"
    __table_args__ = (
        sa.UniqueConstraint("schedule_id", "occurrence_key", name="uq_schedule_occurrences_key"),
        # Single-flight: at most one unsettled occurrence per schedule. The
        # claim is an insert, so the database arbitrates between pollers.
        sa.Index(
            "uq_schedule_occurrences_one_active",
            "schedule_id",
            unique=True,
            sqlite_where=sa.text("status = 'active'"),
            postgresql_where=sa.text("status = 'active'"),
        ),
        sa.Index("ix_schedule_occurrences_status_next_attempt", "status", "next_attempt_at"),
    )

    schedule_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid(),
            sa.ForeignKey("schedules.id", ondelete="CASCADE"),
            nullable=False,
        ),
        description="Parent schedule",
    )
    occurrence_key: str = Field(max_length=128, description="Stable identity of this firing")
    trigger_source: str = Field(description="What fired it: cron | manual")
    scheduled_for: datetime = Field(
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="UTC instant the occurrence was due (the claim time for manual runs)",
    )
    status: str = Field(description="active | success | failed | cancelled")
    attempts: int = Field(default=0, description="Attempts started so far")
    max_attempts: int = Field(default=1, description="Attempts allowed before settling failed")
    next_attempt_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="When the next attempt may start; NULL while one is running",
    )
    lease_expires_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="When a running attempt is presumed dead and may be reaped",
    )
    lease_token: str | None = Field(
        default=None,
        max_length=36,
        description=(
            "Who holds the occurrence now. Every change of hands and every write by "
            "a holder is conditional on it, so a holder that lost the occurrence "
            "(reaped, or settled elsewhere) cannot overwrite the new holder's state"
        ),
    )
    settled_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="UTC datetime the occurrence reached its final status",
    )
    error: str | None = Field(default=None, description="Error from the last failed attempt")


class ScheduleRun(BaseSQLModel, table=True):
    __tablename__ = "schedule_runs"
    __table_args__ = (
        sa.UniqueConstraint("occurrence_id", "attempt", name="uq_schedule_runs_occurrence_attempt"),
    )

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
    occurrence_id: UUID | None = Field(
        default=None,
        sa_column=sa.Column(
            sa.Uuid(),
            sa.ForeignKey("schedule_occurrences.id", ondelete="CASCADE"),
            nullable=True,
            index=True,
        ),
        description="The occurrence this run is an attempt at; NULL on runs from before occurrences",
    )
    attempt: int | None = Field(default=None, description="1-based attempt number within the occurrence")
    execution_started_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description=(
            "When the attempt handed its turn to the agent. Written before the hand-off, "
            "so a run without it is known not to have acted"
        ),
    )
    stop_requested_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="When the scheduler asked the turn to stop (its duration bound)",
    )
    outcome: str | None = Field(
        default=None,
        description="How the attempt ended, by what is known of its execution (AttemptOutcome)",
    )
