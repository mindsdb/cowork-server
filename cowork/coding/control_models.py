from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from cowork.coding.contracts import (
    DeliveryRecord,
    SourceContext,
    WorkspaceKind,
    utc_now,
)

CONTROL_SCHEMA_VERSION = 1
RUNTIME_PROTOCOL_VERSION = "1.0"


class ComputerStatus(str, Enum):
    online = "online"
    offline = "offline"
    draining = "draining"


class RunStatus(str, Enum):
    queued = "queued"
    preparing = "preparing"
    ready = "ready"
    running = "running"
    awaiting_approval = "awaiting_approval"
    completed = "completed"
    cancelled = "cancelled"
    interrupted = "interrupted"
    failed = "failed"
    recovering = "recovering"


TERMINAL_RUN_STATUSES = {
    RunStatus.completed,
    RunStatus.cancelled,
    RunStatus.failed,
}


class WorkspaceStatus(str, Enum):
    pending = "pending"
    preparing = "preparing"
    ready = "ready"
    unavailable = "unavailable"
    released = "released"


class ComputerCapabilities(BaseModel):
    platform: Literal["darwin", "windows", "linux"]
    architecture: str = Field(min_length=1, max_length=64)
    runtime_version: str = Field(min_length=1, max_length=64)
    protocol_versions: list[str] = Field(default_factory=lambda: [RUNTIME_PROTOCOL_VERSION], min_length=1, max_length=8)
    agent_engines: list[str] = Field(default_factory=list, max_length=32)
    shells: list[str] = Field(default_factory=list, max_length=16)
    has_git: bool = True
    has_terminal: bool = True
    supports_local_folders: bool = True
    max_concurrent_runs: int = Field(default=4, ge=1, le=64)

    @field_validator("protocol_versions", "agent_engines", "shells")
    @classmethod
    def unique_values(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item.strip() for item in value if item.strip()))


class Computer(BaseModel):
    schema_version: int = CONTROL_SCHEMA_VERSION
    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=120)
    status: ComputerStatus = ComputerStatus.online
    capabilities: ComputerCapabilities
    registration_epoch: int = Field(default=1, ge=1)
    active_run_count: int = Field(default=0, ge=0)
    last_seen_at: datetime = Field(default_factory=utc_now)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class RuntimeCredential(BaseModel):
    """Server-side runtime authentication material; never returned in lists."""

    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    computer_id: str = Field(min_length=1, max_length=128)
    token_hash: str = Field(min_length=64, max_length=64)
    registration_epoch: int = Field(ge=1)
    created_at: datetime = Field(default_factory=utc_now)


class TaskRunCredential(BaseModel):
    """Per-run agent credential; only its digest reaches durable storage."""

    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    run_id: str = Field(min_length=1, max_length=128)
    computer_id: str = Field(min_length=1, max_length=128)
    epoch: int = Field(ge=1)
    token_hash: str = Field(min_length=64, max_length=64)
    created_at: datetime = Field(default_factory=utc_now)


class TaskResourceScope(BaseModel):
    all_project_resources: bool = True
    resource_ids: list[str] = Field(default_factory=list, max_length=64)

    @field_validator("resource_ids")
    @classmethod
    def unique_resource_ids(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def validate_scope(self) -> TaskResourceScope:
        if self.all_project_resources and self.resource_ids:
            raise ValueError("all-resource scope cannot also name individual resources")
        if not self.all_project_resources and not self.resource_ids:
            raise ValueError("a narrowed task scope must include at least one resource")
        return self


class CodeTask(BaseModel):
    schema_version: int = CONTROL_SCHEMA_VERSION
    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    title: str = Field(min_length=1, max_length=200)
    prompt: str = Field(default="", max_length=200_000)
    project_id: str | None = Field(default=None, max_length=128)
    resource_scope: TaskResourceScope = Field(default_factory=TaskResourceScope)
    source_contexts: list[SourceContext] = Field(default_factory=list, max_length=24)
    deliveries: list[DeliveryRecord] = Field(default_factory=list, max_length=250)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class TaskRun(BaseModel):
    schema_version: int = CONTROL_SCHEMA_VERSION
    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    task_id: str = Field(min_length=1, max_length=128)
    computer_id: str = Field(min_length=1, max_length=128)
    status: RunStatus = RunStatus.queued
    epoch: int = Field(default=1, ge=1)
    lease_id: str | None = Field(default=None, max_length=128)
    lease_expires_at: datetime | None = None
    last_event_seq: int = Field(default=0, ge=0)
    checkpoint: dict[str, object] = Field(default_factory=dict)
    last_error: str | None = Field(default=None, max_length=4_000)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class ExecutionWorkspace(BaseModel):
    schema_version: int = CONTROL_SCHEMA_VERSION
    id: str = Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9_-]+$")
    run_id: str = Field(min_length=1, max_length=128)
    resource_id: str = Field(min_length=1, max_length=128)
    computer_id: str = Field(min_length=1, max_length=128)
    status: WorkspaceStatus = WorkspaceStatus.pending
    path: str = Field(default="", max_length=32_768)
    workspace_kind: WorkspaceKind | None = None
    base_revision: str | None = Field(default=None, max_length=512)
    task_branch: str | None = Field(default=None, max_length=512)
    detail: str = Field(default="", max_length=4_000)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ConnectorGrant(BaseModel):
    schema_version: int = CONTROL_SCHEMA_VERSION
    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    run_id: str = Field(min_length=1, max_length=128)
    epoch: int = Field(default=1, ge=1)
    provider: Literal["github", "linear"]
    connection_name: str = Field(min_length=1, max_length=512)
    actions: list[str] = Field(min_length=1, max_length=32)
    resource_constraints: dict[str, str] = Field(default_factory=dict, max_length=32)
    token_hash: str = Field(min_length=64, max_length=64)
    expires_at: datetime
    revoked_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("actions")
    @classmethod
    def unique_actions(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class RuntimeCommand(BaseModel):
    schema_version: int = CONTROL_SCHEMA_VERSION
    protocol_version: str = RUNTIME_PROTOCOL_VERSION
    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    run_id: str = Field(min_length=1, max_length=128)
    epoch: int = Field(ge=1)
    kind: Literal["prepare", "start", "steer", "approve", "cancel", "checkpoint", "operation", "release"]
    payload: dict[str, object] = Field(default_factory=dict)
    idempotency_key: str = Field(default="", max_length=256)
    created_at: datetime = Field(default_factory=utc_now)
    claimed_at: datetime | None = None
    claim_expires_at: datetime | None = None
    delivery_count: int = Field(default=0, ge=0)
    acked_at: datetime | None = None
    result: dict[str, object] | None = None
    error: str | None = Field(default=None, max_length=4_000)

    @field_validator("result")
    @classmethod
    def bound_result(cls, value: dict[str, object] | None) -> dict[str, object] | None:
        if value is not None and len(json.dumps(value, ensure_ascii=False, default=str)) > 2 * 1024 * 1024:
            raise ValueError("runtime command result exceeds 2 MiB")
        return value


class RuntimeEvent(BaseModel):
    protocol_version: str = RUNTIME_PROTOCOL_VERSION
    run_id: str = Field(min_length=1, max_length=128)
    computer_id: str = Field(min_length=1, max_length=128)
    lease_id: str = Field(min_length=1, max_length=128)
    epoch: int = Field(ge=1)
    seq: int = Field(ge=1)
    kind: Literal["status", "event", "approval", "checkpoint", "workspace", "turn_completed", "error"]
    payload: dict[str, object] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)


class ComputerPage(BaseModel):
    items: list[Computer]


class ResourceAvailability(BaseModel):
    resource_id: str
    status: Literal["available", "offline", "unavailable"]
    eligible_computer_ids: list[str] = Field(default_factory=list)
    required_computer_id: str | None = None
    detail: str = ""


class ResourceAvailabilityPage(BaseModel):
    items: list[ResourceAvailability]


class TaskRunPage(BaseModel):
    items: list[TaskRun]


class TaskControlSnapshot(BaseModel):
    task: CodeTask
    run: TaskRun
    computer: Computer
    workspaces: list[ExecutionWorkspace] = Field(default_factory=list)
