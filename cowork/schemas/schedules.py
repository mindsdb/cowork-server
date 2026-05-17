from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel


class Cadence(str, Enum):
    once = "once"
    hourly = "hourly"
    daily = "daily"
    weekly = "weekly"


class RunStatus(str, Enum):
    running = "running"
    success = "success"
    failed = "failed"


class ScheduleCreateRequest(BaseModel):
    title: str
    prompt: str
    cadence: Cadence
    next_run_at: datetime
    model: str
    timezone: str = "UTC"
    project_id: UUID | None = None
    enabled: bool = True


class ScheduleUpdateRequest(BaseModel):
    title: str | None = None
    prompt: str | None = None
    cadence: Cadence | None = None
    next_run_at: datetime | None = None
    model: str | None = None
    timezone: str | None = None
    project_id: UUID | None = None
    enabled: bool | None = None


class ScheduleResponse(BaseModel):
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
    created_at: datetime | None
    modified_at: datetime | None

    model_config = {"from_attributes": True}


class ScheduleRunResponse(BaseModel):
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

    model_config = {"from_attributes": True}
