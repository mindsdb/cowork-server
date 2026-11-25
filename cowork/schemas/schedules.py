from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID

from typing import Literal

from pydantic import Field, computed_field, field_validator, model_validator

from cowork.schemas.base import CamelRequest, CamelResponse


# The `model` value a schedule stores when it should run on the account's
# configured default models rather than a pinned id. The task form sends it
# for "Account default", and a create without a model stores it too.
#
# It is NOT a servable model id. Passing it into a turn makes the harness
# override every model role with the literal string "default"
# (`_apply_model_override` treats any truthy value as a real pick), and the
# gateway 404s it as `model_not_found` — surfaced to the user as "That model
# isn't available" (ENG-2353). `resolve_schedule_model` turns it back into None
# before the turn is built, so the account-wide defaults govern the run.
DEFAULT_MODEL_SENTINEL = "default"


def resolve_schedule_model(stored: str | None) -> str | None:
    """The model to run a schedule with, or None to follow the account defaults.

    A schedule stores ``DEFAULT_MODEL_SENTINEL`` when it should track the
    account's configured default model. That sentinel is not a real model id,
    so it must become None (a no-op override) before the turn is built, or the
    gateway rejects it. A schedule that pinned a concrete id is passed through
    unchanged; a run whose pin can't be served falls back to the default (see
    cowork.scheduler).
    """
    if not stored or stored == DEFAULT_MODEL_SENTINEL:
        return None
    return stored


class Cadence(str, Enum):
    once = "once"
    every_15_minutes = "every_15_minutes"
    every_30_minutes = "every_30_minutes"
    hourly = "hourly"
    daily = "daily"
    weekdays = "weekdays"
    weekly = "weekly"
    monthly = "monthly"


Weekday = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
_WEEK_ORDER = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


class Recurrence(CamelRequest):
    """Which days a weekly or monthly task runs on. The time of day is the
    task's next run's, in its zone.

    - Weekly: ``days``, such as ``["mon", "wed", "fri"]``.
    - Monthly: ``month_day`` (1-31; a shorter month runs on its last day), or
      ``week`` (1-4, or -1 for the last) with ``weekday``, such as the first
      Monday.
    """

    days: list[Weekday] | None = Field(default=None, min_length=1)
    month_day: int | None = Field(default=None, ge=1, le=31)
    week: Literal[1, 2, 3, 4, -1] | None = None
    weekday: Weekday | None = None

    @field_validator("days")
    @classmethod
    def _unique_in_week_order(cls, days: list[str] | None) -> list[str] | None:
        if days is None:
            return None
        return sorted(set(days), key=_WEEK_ORDER.index)

    @model_validator(mode="after")
    def _one_rule(self) -> "Recurrence":
        if (self.week is None) != (self.weekday is None):
            raise ValueError("`week` and `weekday` go together, as in the first Monday")
        rules = [self.days is not None, self.month_day is not None, self.week is not None]
        if sum(rules) != 1:
            raise ValueError("Give one rule: `days`, `monthDay`, or `week` with `weekday`")
        return self

    def stored(self) -> dict:
        """As the schedule's `recurrence` column holds it."""
        return self.model_dump(exclude_none=True)


class RecurrenceResponse(CamelResponse):
    days: list[str] | None = None
    month_day: int | None = None
    week: int | None = None
    weekday: str | None = None


def check_recurrence(cadence: str, recurrence: "Recurrence | None") -> None:
    """A rule fits its cadence: `days` for weekly, a day of the month for
    monthly, and none for any other cadence."""
    if recurrence is None:
        return
    if cadence == Cadence.weekly and recurrence.days is not None:
        return
    if cadence == Cadence.monthly and recurrence.days is None:
        return
    raise ValueError(f"A {Cadence(cadence).value} task can't take that `recurrence`")


class RunStatus(str, Enum):
    running = "running"
    success = "success"
    failed = "failed"
    cancelled = "cancelled"


class AttemptOutcome(str, Enum):
    """What is known about a run's execution when it ended. Only
    ``failed_before_execution`` shows the agent did nothing."""

    completed = "completed"
    # The turn ended on its own, but at one of its limits (steps, tokens or
    # automatic continuations) with nobody there to say "keep going".
    stopped_at_limit = "stopped_at_limit"
    # The turn ended on its own, but a change it asked approval for was
    # refused: a scheduled run has nobody to approve it, so it wasn't made.
    writes_refused = "writes_refused"
    # Failed before the agent could act: setup or permission, or a turn whose
    # stream shows nothing but its start, its memory read and its failure.
    failed_before_execution = "failed_before_execution"
    # Failed after the agent may have acted.
    failed_during_execution = "failed_during_execution"
    # Stopped by the run's time limit, or its process went away, after the
    # agent may have acted.
    interrupted = "interrupted"
    # Stopped by the user.
    cancelled = "cancelled"
    # No terminal record says how the turn ended, or a failure after handoff
    # prevents determining what the turn did.
    uncertain = "uncertain"


class ScheduleState(str, Enum):
    """What a schedule will do next, for display."""

    active = "active"
    # The user paused it: nothing starts on its own until it is resumed.
    paused = "paused"
    # Repeated failures paused it.
    auto_paused = "auto_paused"
    # A one-off whose time passed while the app could not run it, by too much
    # to run on its own. Run now still runs it.
    missed = "missed"
    # A one-off that has run.
    done = "done"


def schedule_state(schedule) -> ScheduleState:
    if schedule.auto_paused_at is not None:
        return ScheduleState.auto_paused
    if schedule.paused_at is not None:
        return ScheduleState.paused
    if schedule.missed_at is not None:
        return ScheduleState.missed
    if not schedule.enabled:
        return ScheduleState.done if schedule.cadence == Cadence.once else ScheduleState.paused
    return ScheduleState.active


def _utc_next_run(value: datetime | None) -> datetime | None:
    # SQLite stores a datetime's clock time without its offset and reads it
    # back as UTC, so 09:00-07:00 would come back as 09:00 UTC. A naive time
    # already means UTC.
    if value is None or value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc)


class ScheduleCreateRequest(CamelRequest):
    title: str
    prompt: str
    cadence: Cadence
    next_run_at: datetime
    # Weekly and monthly only; see Recurrence.
    recurrence: Recurrence | None = None
    model: str | None = None
    timezone: str = "UTC"
    project_id: UUID | None = None
    enabled: bool = True

    _next_run_utc = field_validator("next_run_at")(_utc_next_run)

    @model_validator(mode="after")
    def _recurrence_fits(self) -> "ScheduleCreateRequest":
        check_recurrence(self.cadence, self.recurrence)
        return self


class ScheduleUpdateRequest(CamelRequest):
    title: str | None = None
    prompt: str | None = None
    cadence: Cadence | None = None
    next_run_at: datetime | None = None
    # Checked against the cadence it ends up with, by ScheduleService.
    recurrence: Recurrence | None = None
    model: str | None = None
    timezone: str | None = None
    project_id: UUID | None = None
    enabled: bool | None = None

    _next_run_utc = field_validator("next_run_at")(_utc_next_run)


class ScheduleResponse(CamelResponse):
    id: UUID
    title: str
    prompt: str
    cadence: str
    # Which days a weekly or monthly task runs on; see Recurrence. Absent on
    # other cadences, and on a weekly task saved before rules existed, which
    # runs on its next run's weekday.
    recurrence: RecurrenceResponse | None = None
    timezone: str
    next_run_at: datetime
    enabled: bool
    project_id: UUID
    model: str
    last_run_at: datetime | None
    last_result_conversation_id: UUID | None
    last_error: str | None
    missed_runs: int
    consecutive_failures: int = 0
    auto_paused_at: datetime | None = None
    paused_at: datetime | None = None
    missed_at: datetime | None = None
    # Not a Schedule column — endpoints fill it from ScheduleRunService so the
    # UI can show an in-flight run (manual or cron).
    running: bool = False
    run_count: int = 0
    # Filled by the endpoints: the latest finished run (ScheduleRunResponse
    # shape), so a failure shows on the schedule before any auto-pause.
    last_run: ScheduleRunResponse | None = None
    created_at: datetime | None
    modified_at: datetime | None

    @computed_field
    @property
    def state(self) -> ScheduleState:
        return schedule_state(self)


class RunTrigger(str, Enum):
    trial = "trial"
    manual = "manual"
    cron = "cron"


class ScheduleRunResponse(CamelResponse):
    id: UUID
    schedule_id: UUID
    started_at: datetime
    finished_at: datetime | None
    duration_ms: int | None
    status: str
    error: str | None
    conversation_id: UUID | None
    is_manual: bool
    # A run-now that tried the task out (POST .../run-now?trial=true).
    trial: bool = False
    outcome: AttemptOutcome | None = None
    # The pinned model that could not be served before using the account default.
    unavailable_model: str | None = None
    # The cron slot the run was for; a run that started well after it was
    # missed or delayed (the app closed, the computer asleep). None on a
    # manual run.
    slot_at: datetime | None = None
    created_at: datetime | None

    @computed_field
    @property
    def trigger(self) -> RunTrigger:
        return RunTrigger.trial if self.trial else (RunTrigger.manual if self.is_manual else RunTrigger.cron)
