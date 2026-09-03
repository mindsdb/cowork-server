from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from cowork.coding.connector_capabilities import ConnectorCapability
from cowork.coding.contracts import PermissionMode, Personality, ReasoningEffort, ServiceTier
from cowork.coding.control_models import (
    RUNTIME_PROTOCOL_VERSION,
    CodeTask,
    Computer,
    ComputerCapabilities,
    ExecutionWorkspace,
    PendingComputer,
    RuntimeCommand,
    TaskRun,
)
from cowork.coding.project_models import CodeProject


class ComputerRegistrationRequest(BaseModel):
    protocol_version: str = RUNTIME_PROTOCOL_VERSION
    registration_token: str = Field(min_length=32, max_length=512)
    name: str = Field(min_length=1, max_length=120)
    capabilities: ComputerCapabilities


class ComputerRegistrationResponse(BaseModel):
    computer: Computer
    runtime_token: str
    heartbeat_interval_seconds: int = 10


class ComputerHeartbeatRequest(BaseModel):
    protocol_version: str = RUNTIME_PROTOCOL_VERSION
    active_run_count: int = Field(default=0, ge=0)


class RuntimeLeaseRequest(BaseModel):
    protocol_version: str = RUNTIME_PROTOCOL_VERSION
    wait_seconds: float = Field(default=0, ge=0, le=20)


class RuntimeExecutionConfig(BaseModel):
    engine_id: str
    model: str
    permission_mode: PermissionMode
    reasoning_effort: ReasoningEffort | None = None
    service_tier: ServiceTier = "standard"
    personality: Personality = "pragmatic"
    network_access: bool = False
    web_search: bool = False
    developer_instructions: str = ""
    environment: dict[str, str] = Field(default_factory=dict)


class RuntimeLease(BaseModel):
    task: CodeTask
    run: TaskRun
    lease_id: str
    agent_token: str = Field(min_length=32, max_length=512)
    project: CodeProject | None = None
    execution: RuntimeExecutionConfig
    connector_capabilities: list[ConnectorCapability] = Field(default_factory=list, max_length=64)
    workspaces: list[ExecutionWorkspace] = Field(default_factory=list, max_length=64)
    commands: list[RuntimeCommand] = Field(default_factory=list)


class RuntimeCommandPage(BaseModel):
    items: list[RuntimeCommand] = Field(default_factory=list)


class RuntimeFenceRequest(BaseModel):
    protocol_version: str = RUNTIME_PROTOCOL_VERSION
    lease_id: str = Field(min_length=1, max_length=128)
    epoch: int = Field(ge=1)


class RuntimeCommandAckRequest(RuntimeFenceRequest):
    command_id: str = Field(min_length=1, max_length=128)
    result: dict[str, object] | None = None
    error: str | None = Field(default=None, max_length=4_000)


class RegistrationTokenRequest(BaseModel):
    """Optional details from the desktop's "Connect a computer" form."""

    name: str = Field(min_length=1, max_length=120)
    platform: Literal["darwin", "windows", "linux"]
    replaces: str | None = Field(default=None, min_length=1, max_length=128)


class RegistrationTokenResponse(BaseModel):
    registration_token: str
    expires_in_seconds: int = 600
    pending: PendingComputer | None = None


class ComputerUpdateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
