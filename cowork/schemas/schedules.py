from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel

from cowork.schemas.base import CamelRequest, CamelResponse


# The `model` value a schedule stores when it should run on the account's
# configured default models rather than a pinned id. The scheduler UI removed
# the per-task model picker (see ScheduleTaskModal), so the frontend always
# sends `model: null` and every schedule carries this sentinel.
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
    account's configured default model — the only mode the UI offers. That
    sentinel is not a real model id, so it must become None (a no-op override)
    before the turn is built, or the gateway rejects it (ENG-2353). A schedule
    that pinned a concrete id (legacy rows, or a future picker) is passed
    through unchanged.
    """
    if not stored or stored == DEFAULT_MODEL_SENTINEL:
        return None
    return stored


class Cadence(str, Enum):
    once = "once"
    hourly = "hourly"
    daily = "daily"
    weekdays = "weekdays"
    weekly = "weekly"


class RunStatus(str, Enum):
    running = "running"
    success = "success"
    failed = "failed"
    cancelled = "cancelled"


class OccurrenceStatus(str, Enum):
    active = "active"
    success = "success"
    failed = "failed"
    cancelled = "cancelled"


class AttemptOutcome(str, Enum):
    """What is known about an attempt's execution when it ended.

    Only ``failed_before_execution`` is proof the agent did nothing, so it is
    the only failure that may be tried again automatically. Every other
    failure keeps its output for review and is not replayed.
    """

    completed = "completed"
    # Failed before the agent could act: setup, permission, or a turn whose
    # stream shows nothing but its start and its failure.
    failed_before_execution = "failed_before_execution"
    # Failed after the agent may have acted.
    failed_during_execution = "failed_during_execution"
    # Stopped by the scheduler's duration bound, or its process went away,
    # after the agent may have acted.
    interrupted = "interrupted"
    # Stopped by the user.
    cancelled = "cancelled"
    # No terminal record says how the turn ended.
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
    # A one-off that has run, or is running.
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


class ScheduleCreateRequest(CamelRequest):
    title: str
    prompt: str
    cadence: Cadence
    next_run_at: datetime
    model: str | None = None
    timezone: str = "UTC"
    project_id: UUID | None = None
    enabled: bool = True


class ScheduleUpdateRequest(CamelRequest):
    title: str | None = None
    prompt: str | None = None
    cadence: Cadence | None = None
    next_run_at: datetime | None = None
    model: str | None = None
    timezone: str | None = None
    project_id: UUID | None = None
    enabled: bool | None = None


class SkippedPeriodResponse(CamelResponse):
    first_slot: datetime
    last_slot: datetime
    slot_count: int
    reason: str
    created_at: datetime | None


class ScheduleResponse(CamelResponse):
    id: UUID
    title: str
    prompt: str
    cadence: str
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
    # Filled by the endpoints: the derived ScheduleState, and the most recent
    # stretch of slots it passed over without running.
    state: ScheduleState = ScheduleState.active
    last_skipped: SkippedPeriodResponse | None = None
    # The number of the version the next occurrence will run.
    version: int | None = None
    # The latest finished run (ScheduleRunResponse shape), filled by the
    # endpoints.
    last_run: dict | None = None
    # Not a Schedule column — endpoints fill it from ScheduleRunService so the
    # UI can show an in-flight run (manual or cron).
    running: bool = False
    # Also filled by the endpoints: when a failed run will be tried again.
    retry_at: datetime | None = None
    created_at: datetime | None
    modified_at: datetime | None


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
    occurrence_id: UUID | None = None
    attempt: int | None = None
    # Read through the occurrence (ScheduleRunService.run_context): what fired
    # it, the slot it ran for, and the version it ran.
    trigger: str | None = None
    scheduled_for: datetime | None = None
    version: int | None = None
    outcome: AttemptOutcome | None = None
    execution_started_at: datetime | None = None
    stop_requested_at: datetime | None = None
    created_at: datetime | None


class ScheduleEventResponse(CamelResponse):
    id: UUID
    schedule_id: UUID
    kind: str
    actor: str | None
    detail: dict | None
    created_at: datetime | None
