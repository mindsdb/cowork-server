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


ReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh"]
ServiceTier = Literal["standard", "priority"]
Personality = Literal["none", "friendly", "pragmatic"]


class TaskCapability(str, Enum):
    """Versioned user-visible operations an execution computer actually supports."""

    files = "files"
    review = "review"
    terminal = "terminal"
    project_actions = "project_actions"
    slash_commands = "slash_commands"
    task_controls = "task_controls"
    extensions = "extensions"
    platform_settings = "platform_settings"
    fork = "fork"
    open_workspace = "open_workspace"


class TaskCapabilities(BaseModel):
    files: bool = True
    review: bool = True
    terminal: bool = True
    project_actions: bool = True
    slash_commands: bool = True
    task_controls: bool = True
    extensions: bool = True
    platform_settings: bool = True
    fork: bool = True
    open_workspace: bool = True


class TerminalShellPreference(str, Enum):
    auto = "auto"
    bash = "bash"
    zsh = "zsh"
    fish = "fish"
    system = "system"
    pwsh = "pwsh"
    powershell = "powershell"
    cmd = "cmd"


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
    child_work = "child_work"
    approval = "approval"
    usage = "usage"
    error = "error"
    command_result = "command_result"


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
    manifest_version: int = 2
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
    features: dict[str, Literal["supported", "unsupported"]] = Field(default_factory=dict)
    commands: list[EngineCommand] = Field(default_factory=list)


class ExtensionEntry(BaseModel):
    id: str
    label: str
    description: str = ""
    status: str = "available"
    detail: str = ""
    path: str | None = None
    supersedes: list[ExtensionEntry] = Field(default_factory=list)


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
    resource_id: str | None = Field(default=None, min_length=1, max_length=128)
    relative_path: str | None = Field(default=None, min_length=1, max_length=32_768)
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    content_hash: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_precise_reference(self) -> InputReference:
        if bool(self.resource_id) != bool(self.relative_path):
            raise ValueError("resource_id and relative_path must be provided together")
        if bool(self.line_start) != bool(self.line_end):
            raise ValueError("line_start and line_end must be provided together")
        if self.line_start and self.line_end and self.line_end < self.line_start:
            raise ValueError("line_end must not be before line_start")
        return self


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


class TerminalTab(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=80)
    project_action_id: str | None = Field(default=None, min_length=1, max_length=160)
    project_resource_id: str | None = Field(default=None, min_length=1, max_length=120)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("label")
    @classmethod
    def clean_label(cls, value: str) -> str:
        label = value.strip()
        if not label:
            raise ValueError("terminal names cannot be empty")
        return label


class CodingSession(BaseModel):
    schema_version: int = SCHEMA_VERSION
    id: str
    title: str
    engine_id: str
    engine_adapter_version: str
    model: str
    permission_mode: PermissionMode = PermissionMode.supervised
    reasoning_effort: ReasoningEffort | None = None
    service_tier: ServiceTier = "standard"
    personality: Personality = "pragmatic"
    network_access: bool = False
    web_search: bool = False
    additional_dirs: list[str] = Field(default_factory=list)
    status: SessionStatus = SessionStatus.ready
    # Compatibility projection of the durable control-plane records. The
    # session remains the existing renderer/runtime contract while Task and
    # TaskRun become the canonical durable identities.
    task_id: str | None = None
    run_id: str | None = None
    computer_id: str | None = None
    run_status: Literal[
        "queued",
        "preparing",
        "ready",
        "running",
        "awaiting_approval",
        "completed",
        "cancelled",
        "interrupted",
        "failed",
        "recovering",
    ] | None = None
    computer_name: str | None = None
    computer_status: Literal["online", "offline", "draining"] | None = None
    computer_is_local: bool = True
    task_capabilities: TaskCapabilities = Field(default_factory=TaskCapabilities)
    resource_ids: list[str] = Field(default_factory=list, max_length=64)
    scope_all_project_resources: bool = True
    runtime_epoch: int = Field(default=1, ge=1)
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
    terminal_tabs: list[TerminalTab] = Field(default_factory=list, max_length=12)
    pinned: bool = False
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
    staged: bool = False
    unstaged: bool = False

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
    resource_ids: list[str] | None = Field(default=None, min_length=1, max_length=64)
    computer_id: str | None = Field(default=None, min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=200_000)
    engine_id: str | None = Field(default=None, min_length=1, max_length=128)
    model: str | None = Field(default=None, min_length=1, max_length=256)
    permission_mode: PermissionMode = PermissionMode.supervised
    reasoning_effort: ReasoningEffort | None = None
    service_tier: ServiceTier = "standard"
    personality: Personality = "pragmatic"
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

    @field_validator("resource_ids")
    @classmethod
    def unique_resources(cls, value: list[str] | None) -> list[str] | None:
        return list(dict.fromkeys(value)) if value is not None else None


class SessionUpdateRequest(BaseModel):
    model: str | None = Field(default=None, min_length=1, max_length=256)
    permission_mode: PermissionMode | None = None
    reasoning_effort: ReasoningEffort | None = None
    service_tier: ServiceTier | None = None
    personality: Personality | None = None
    network_access: bool | None = None
    web_search: bool | None = None
    additional_dirs: list[str] | None = Field(default=None, max_length=16)


class SessionRecoverRequest(BaseModel):
    computer_id: str | None = Field(default=None, min_length=1, max_length=128)
    allow_recreate: bool = False


class TurnRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=200_000)
    attachments: list[InputReference] = Field(default_factory=list, max_length=20)


class QueueRunRequest(BaseModel):
    instruction_id: str | None = Field(default=None, min_length=1, max_length=128)


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


class TerminalTabState(TerminalTab):
    status: TerminalStatus = TerminalStatus.stopped
    exit_code: int | None = None
    error: str | None = None


class TerminalTabPage(BaseModel):
    items: list[TerminalTabState] = Field(default_factory=list)


class TerminalCreateRequest(BaseModel):
    label: str | None = Field(default=None, max_length=80)

    @field_validator("label")
    @classmethod
    def clean_optional_label(cls, value: str | None) -> str | None:
        if value is None:
            return None
        label = value.strip()
        if not label:
            raise ValueError("terminal names cannot be empty")
        return label


class TerminalRenameRequest(BaseModel):
    label: str = Field(min_length=1, max_length=80)

    @field_validator("label")
    @classmethod
    def clean_label(cls, value: str) -> str:
        label = value.strip()
        if not label:
            raise ValueError("terminal names cannot be empty")
        return label


class TerminalStartRequest(BaseModel):
    cols: int = Field(default=100, ge=1, le=1_000)
    rows: int = Field(default=30, ge=1, le=1_000)
    shell: TerminalShellPreference = TerminalShellPreference.auto


class TerminalShellOption(BaseModel):
    id: TerminalShellPreference
    label: str


class TerminalShellInventory(BaseModel):
    platform: str
    resolved: TerminalShellPreference
    items: list[TerminalShellOption]


class TerminalInputRequest(BaseModel):
    data_base64: str = Field(min_length=1, max_length=1_500_000)


class TerminalResizeRequest(BaseModel):
    cols: int = Field(ge=1, le=1_000)
    rows: int = Field(ge=1, le=1_000)
