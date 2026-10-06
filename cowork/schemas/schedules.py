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
    # Not a Schedule column — endpoints fill it from ScheduleRunService so the
    # UI can show an in-flight run (manual or cron).
    running: bool = False
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
    created_at: datetime | None
