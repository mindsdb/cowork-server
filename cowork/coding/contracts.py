from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from cowork.coding.redaction import redact_text, sanitize

SCHEMA_VERSION = 1


def utc_now() -> datetime:
    return datetime.now(UTC)


class SessionStatus(str, Enum):
    ready = "ready"
    running = "running"
    awaiting_approval = "awaiting_approval"
    completed = "completed"
    cancelled = "cancelled"
    interrupted = "interrupted"
    failed = "failed"


class WorkspaceKind(str, Enum):
    git_worktree = "git_worktree"
    local_copy = "local_copy"
    direct_folder = "direct_folder"


class PermissionMode(str, Enum):
    read_only = "read_only"
    supervised = "supervised"
    workspace = "workspace"
    full_access = "full_access"


class EventType(str, Enum):
    session = "session"
    user_message = "user_message"
    agent_message = "agent_message"
    reasoning = "reasoning"
    plan = "plan"
    tool = "tool"
    command = "command"
    file_change = "file_change"
    diff = "diff"
    approval = "approval"
    usage = "usage"
    error = "error"


class ApprovalDecision(str, Enum):
    approve_once = "approve_once"
    approve_session = "approve_session"
    deny = "deny"


class EngineCommand(BaseModel):
    name: str
    label: str
    description: str
    argument_hint: str | None = None
    action: Literal["turn", "goal", "compact", "status", "client"] = "turn"
    client_action: Literal["controls", "skills", "mcp", "fork", "terminal"] | None = None

    @model_validator(mode="after")
    def require_client_action(self) -> EngineCommand:
        if self.action == "client" and self.client_action is None:
            raise ValueError("client commands require a registered client action")
        if self.action != "client" and self.client_action is not None:
            raise ValueError("only client commands may register a client action")
        return self


class EngineCapabilities(BaseModel):
    id: str
    label: str
    adapter_version: str
    available: bool
    reason: str | None = None
    supports_steering: bool = False
    supports_approvals: bool = False
    supports_reasoning: bool = False
    supports_diff_events: bool = False
    supports_models: bool = False
    supports_terminal: bool = False
    commands: list[EngineCommand] = Field(default_factory=list)


class ExtensionEntry(BaseModel):
    id: str
    label: str
    description: str = ""
    status: str = "available"
    detail: str = ""
    path: str | None = None


class ExtensionInventory(BaseModel):
    skills: list[ExtensionEntry] = Field(default_factory=list)
    mcp_servers: list[ExtensionEntry] = Field(default_factory=list)
    hooks: list[ExtensionEntry] = Field(default_factory=list)
    apps: list[ExtensionEntry] = Field(default_factory=list)
    plugins: list[ExtensionEntry] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    config_path: str | None = None


class RuntimePlatformStatus(BaseModel):
    platform: str
    windows_sandbox: str | None = None
    setup_started: bool = False


class CodingEvent(BaseModel):
    schema_version: int = SCHEMA_VERSION
    seq: int = 0
    timestamp: datetime = Field(default_factory=utc_now)
    type: EventType
    title: str = ""
    text: str = ""
    phase: Literal["started", "progress", "completed", "failed", "pending"] | None = None
    item_id: str | None = None
    turn_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("title")
    @classmethod
    def bound_title(cls, value: str) -> str:
        return redact_text(value)[:512]

    @field_validator("text")
    @classmethod
    def bound_text(cls, value: str) -> str:
        # Agent output is untrusted and may be unbounded. Keep a generous event
        # ceiling while preventing one notification from exhausting memory/disk.
        return redact_text(value)[:32_768]

    @field_validator("data")
    @classmethod
    def bound_data(cls, value: dict[str, Any]) -> dict[str, Any]:
        bounded = sanitize(value)
        return bounded if isinstance(bounded, dict) else {}


class PendingApproval(BaseModel):
    id: str
    method: str
    kind: str
    title: str
    detail: str
    cwd: str | None = None
    risk: str
    scope: str
    allow_session: bool = False
    created_at: datetime = Field(default_factory=utc_now)


class InputReference(BaseModel):
    name: str = Field(min_length=1, max_length=512)
    path: str = Field(min_length=1, max_length=32_768)
    kind: Literal["mention", "local_image"] = "mention"


class QueuedInstruction(BaseModel):
    id: str
    prompt: str = Field(min_length=1, max_length=200_000)
    attachments: list[InputReference] = Field(default_factory=list, max_length=20)
    created_at: datetime = Field(default_factory=utc_now)


class TaskWorkspace(BaseModel):
    folder_id: str
    folder_name: str
    source_path: str
    workspace_path: str
    workspace_kind: WorkspaceKind
    repository_root: str | None = None
    base_revision: str | None = None
    base_branch: str | None = None
    task_branch: str | None = None
    source_dirty: bool = False


class SourceComment(BaseModel):
    id: str = Field(default="", max_length=512)
    author: str = Field(default="", max_length=512)
    body: str = Field(default="", max_length=20_000)
    url: str = Field(default="", max_length=8_192)
    created_at: str = Field(default="", max_length=120)


class SourceAttachment(BaseModel):
    id: str = Field(default="", max_length=512)
    title: str = Field(default="", max_length=512)
    url: str = Field(min_length=1, max_length=8_192)


class SourceContext(BaseModel):
    provider: Literal["github", "linear", "slack"]
    kind: Literal["issue", "pull_request", "conversation"]
    url: str = Field(min_length=1, max_length=8_192)
    title: str = Field(default="", max_length=512)
    external_id: str = Field(default="", max_length=512)
    connection_name: str | None = Field(default=None, max_length=512)
    body: str = Field(default="", max_length=100_000)
    state: str = Field(default="", max_length=120)
    author: str = Field(default="", max_length=512)
    comments: list[SourceComment] = Field(default_factory=list, max_length=100)
    attachments: list[SourceAttachment] = Field(default_factory=list, max_length=100)


class DeliveryRecord(BaseModel):
    provider: Literal["github", "linear", "slack"]
    action: Literal["progress", "result", "draft_pull_request", "complete_source"]
    target_url: str
    status: Literal["pending", "published", "failed"] = "pending"
    external_url: str | None = None
    detail: str = ""
    folder_id: str | None = None
    folder_name: str | None = None
    base_branch: str | None = None
    task_branch: str | None = None
    connection_name: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class DeliveryAutomationPolicy(BaseModel):
    fix_failing_checks: bool = False
    mark_ready_when_passing: bool = False
    merge_when_approved: bool = False
    complete_source_after_merge: bool = False
    archive_after_merge: bool = False
    max_fix_attempts: int = Field(default=2, ge=1, le=5)


class DeliveryAutomationState(BaseModel):
    fix_attempts: dict[str, int] = Field(default_factory=dict, max_length=128)


class DeliveryAutomationClaimRequest(BaseModel):
    fingerprint: str = Field(min_length=1, max_length=512)


class DeliveryAutomationClaim(BaseModel):
    claimed: bool
    attempts: int
    limit: int


class ResolvedSkill(BaseModel):
    id: str
    kind: Literal["skill", "instructions", "workflow"]
    name: str
    description: str = ""
    origin: Literal["team", "personal", "built_in"]
    source_id: str | None = None
    source_name: str
    source_path: str
    version: str | None = None
    content_hash: str


class CodingSession(BaseModel):
    schema_version: int = SCHEMA_VERSION
    id: str
    title: str
    engine_id: str
    engine_adapter_version: str
    model: str
    permission_mode: PermissionMode = PermissionMode.supervised
    reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] | None = None
    service_tier: Literal["standard", "priority"] = "standard"
    personality: Literal["none", "friendly", "pragmatic"] = "pragmatic"
    network_access: bool = False
    web_search: bool = False
    additional_dirs: list[str] = Field(default_factory=list)
    status: SessionStatus = SessionStatus.ready
    project_id: str | None = None
    project_name: str | None = None
    source_path: str
    workspace_path: str
    workspace_kind: WorkspaceKind
    workspaces: list[TaskWorkspace] = Field(default_factory=list)
    repository_root: str | None = None
    base_revision: str | None = None
    source_dirty: bool = False
    workspace_warning: str | None = None
    guidance_summary: str | None = None
    developer_instructions: str = ""
    resolved_skills: list[ResolvedSkill] = Field(default_factory=list)
    # ``None`` identifies legacy tasks created before task-scoped resolution.
    # An empty list is an intentional snapshot with no shared Code skills.
    skill_roots: list[str] | None = None
    skill_instructions: str = ""
    environment: dict[str, str] = Field(default_factory=dict)
    allocated_ports: dict[str, int] = Field(default_factory=dict)
    source_contexts: list[SourceContext] = Field(default_factory=list)
    deliveries: list[DeliveryRecord] = Field(default_factory=list)
    delivery_policy: DeliveryAutomationPolicy = Field(default_factory=DeliveryAutomationPolicy)
    delivery_automation: DeliveryAutomationState = Field(default_factory=DeliveryAutomationState)
    engine_session_id: str | None = None
    active_turn_id: str | None = None
    pending_approval: PendingApproval | None = None
    queued_instructions: list[QueuedInstruction] = Field(default_factory=list)
    archived: bool = False
    last_error: str | None = None
    event_count: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class WorkspaceInspection(BaseModel):
    path: str
    exists: bool
    is_directory: bool
    is_git: bool
    repository_root: str | None = None
    branch: str | None = None
    revision: str | None = None
    dirty: bool = False
    warning: str | None = None


class DiffFile(BaseModel):
    folder_id: str | None = None
    folder_name: str | None = None
    path: str
    status: str
    additions: int = 0
    deletions: int = 0
    patch: str = ""
    binary: bool = False

    @field_validator("patch")
    @classmethod
    def bound_patch(cls, value: str) -> str:
        limit = 256 * 1024
        if len(value) <= limit:
            return value
        return value[:limit] + "\n… diff truncated by Cowork …\n"


class GitState(BaseModel):
    folder_id: str | None = None
    folder_name: str | None = None
    is_git: bool
    branch: str | None = None
    revision: str | None = None
    detached: bool = False
    dirty: bool = False
    status_lines: list[str] = Field(default_factory=list)
    worktree_path: str
    source_path: str


class SessionCreateRequest(BaseModel):
    path: str | None = Field(default=None, min_length=1, max_length=32_768)
    project_id: str | None = Field(default=None, min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=200_000)
    engine_id: str | None = Field(default=None, min_length=1, max_length=128)
    model: str | None = Field(default=None, min_length=1, max_length=256)
    permission_mode: PermissionMode = PermissionMode.supervised
    reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] | None = None
    service_tier: Literal["standard", "priority"] = "standard"
    personality: Literal["none", "friendly", "pragmatic"] = "pragmatic"
    network_access: bool = False
    web_search: bool = False
    additional_dirs: list[str] = Field(default_factory=list, max_length=16)
    attachments: list[InputReference] = Field(default_factory=list, max_length=20)
    allow_direct_folder: bool = False
    source_contexts: list[SourceContext] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def require_workspace_source(self) -> SessionCreateRequest:
        if bool(self.path) == bool(self.project_id):
            raise ValueError("Choose exactly one Code Project or folder")
        return self


class SessionUpdateRequest(BaseModel):
    model: str | None = Field(default=None, min_length=1, max_length=256)
    permission_mode: PermissionMode | None = None
    reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] | None = None
    service_tier: Literal["standard", "priority"] | None = None
    personality: Literal["none", "friendly", "pragmatic"] | None = None
    network_access: bool | None = None
    web_search: bool | None = None
    additional_dirs: list[str] | None = Field(default=None, max_length=16)


class TurnRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=200_000)
    attachments: list[InputReference] = Field(default_factory=list, max_length=20)


class ApprovalRequest(BaseModel):
    decision: ApprovalDecision


class BranchRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class CommitRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)


class RenameSessionRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class SessionPage(BaseModel):
    items: list[CodingSession]


class EventPage(BaseModel):
    items: list[CodingEvent]
    next_seq: int


class TerminalStatus(str, Enum):
    stopped = "stopped"
    running = "running"
    exited = "exited"
    failed = "failed"


class TerminalChunk(BaseModel):
    seq: int
    data_base64: str
    stream: Literal["stdout", "stderr"] = "stdout"
    cap_reached: bool = False
    timestamp: datetime = Field(default_factory=utc_now)


class TerminalPage(BaseModel):
    process_id: str | None = None
    status: TerminalStatus = TerminalStatus.stopped
    items: list[TerminalChunk] = Field(default_factory=list)
    first_seq: int = 0
    next_seq: int = 0
    exit_code: int | None = None
    error: str | None = None


class TerminalStartRequest(BaseModel):
    cols: int = Field(default=100, ge=1, le=1_000)
    rows: int = Field(default=30, ge=1, le=1_000)


class TerminalInputRequest(BaseModel):
    data_base64: str = Field(min_length=1, max_length=1_500_000)


class TerminalResizeRequest(BaseModel):
    cols: int = Field(ge=1, le=1_000)
    rows: int = Field(ge=1, le=1_000)
