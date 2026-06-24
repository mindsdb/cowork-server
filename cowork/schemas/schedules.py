from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel

from cowork.schemas.base import CamelRequest, CamelResponse


class Cadence(str, Enum):
    once = "once"
    hourly = "hourly"
    daily = "daily"
    weekly = "weekly"


class RunStatus(str, Enum):
    running = "running"
    success = "success"
    failed = "failed"


class MissedRunPolicy(str, Enum):
    """What to do with occurrences that came due while the app was offline.

    skip      — fast-forward to the next future occurrence, run nothing (the
                historical behaviour).
    run_once  — run a single catch-up immediately on resume, then fast-forward.
    catch_up  — run every missed occurrence (bounded by ``_MAX_CATCH_UP``),
                oldest first, then continue normally.
    """

    skip = "skip"
    run_once = "run_once"
    catch_up = "catch_up"


class ScheduleHealth(str, Enum):
    """Rolled-up health derived from recent run outcomes.

    ok       — last run succeeded (or never run yet).
    failing  — one or more recent consecutive failures, still enabled and
               retrying on the normal cadence.
    paused   — auto-paused after too many consecutive failures, or paused by
               the user. ``last_error`` carries the reason for the former.
    """

    ok = "ok"
    failing = "failing"
    paused = "paused"


class ScheduleCreateRequest(CamelRequest):
    title: str
    prompt: str
    cadence: Cadence
    next_run_at: datetime
    model: str | None = None
    timezone: str = "UTC"
    project_id: UUID | None = None
    enabled: bool = True
    missed_run_policy: MissedRunPolicy = MissedRunPolicy.skip


class ScheduleUpdateRequest(CamelRequest):
    title: str | None = None
    prompt: str | None = None
    cadence: Cadence | None = None
    next_run_at: datetime | None = None
    model: str | None = None
    timezone: str | None = None
    project_id: UUID | None = None
    enabled: bool | None = None
    missed_run_policy: MissedRunPolicy | None = None


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
    missed_run_policy: str
    consecutive_failures: int
    health: str
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
    attempts: int
    conversation_id: UUID | None
    is_manual: bool
    created_at: datetime | None
