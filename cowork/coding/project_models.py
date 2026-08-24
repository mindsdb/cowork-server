from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# DeliveryRecord and SourceContext remain available from this long-standing
# public model module for compatibility with existing callers.
from cowork.coding.contracts import (  # noqa: F401
    DeliveryRecord,
    PermissionMode,
    SourceContext,
    utc_now,
)
from cowork.coding.skill_models import ProjectSkillSource


def _valid_environment_name(name: str) -> bool:
    return bool(name) and name.replace("_", "a").isalnum() and name[0].isalpha()


class ProjectCommand(BaseModel):
    """A shell-free command executed inside one project folder."""

    id: str = Field(min_length=1, max_length=160)
    label: str = Field(min_length=1, max_length=120)
    argv: list[str] = Field(min_length=1, max_length=64)
    phase: Literal["setup", "validate"]

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, value: list[str]) -> list[str]:
        if any(not item or "\x00" in item for item in value):
            raise ValueError("command arguments cannot be empty or contain NUL bytes")
        return value


class ProjectFolder(BaseModel):
    id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=120)
    path: str = Field(min_length=1, max_length=32_768)
    base_branch: str | None = Field(default=None, max_length=255)
    commands: list[ProjectCommand] = Field(default_factory=list, max_length=32)


class ProjectConnection(BaseModel):
    provider: Literal["github", "linear", "slack"]
    name: str = Field(min_length=1, max_length=512)
    label: str = Field(default="", max_length=512)


class PlaybookReference(BaseModel):
    repository: str = Field(min_length=1, max_length=32_768)
    branch: str = Field(default="main", min_length=1, max_length=255)
    applied_revision: str | None = None
    available_revision: str | None = None
    cache_path: str | None = None
    last_checked_at: datetime | None = None
    disabled_items: list[str] = Field(default_factory=list, max_length=250)


class ProjectEnvironment(BaseModel):
    variables: dict[str, str] = Field(default_factory=dict, max_length=128)
    port_names: list[str] = Field(default_factory=lambda: ["PORT"], max_length=16)

    @field_validator("variables")
    @classmethod
    def validate_variables(cls, value: dict[str, str]) -> dict[str, str]:
        for name, item in value.items():
            if not _valid_environment_name(name):
                raise ValueError(f"invalid environment variable name: {name}")
            if len(item) > 32_768:
                raise ValueError(f"environment variable {name} is too large")
        return value

    @field_validator("port_names")
    @classmethod
    def validate_port_names(cls, value: list[str]) -> list[str]:
        if len(value) != len({name.casefold() for name in value}):
            raise ValueError("development port names must be unique")
        for name in value:
            if not _valid_environment_name(name):
                raise ValueError(f"invalid development port name: {name}")
        return value

    @model_validator(mode="after")
    def reject_port_variable_collisions(self) -> ProjectEnvironment:
        variables_by_key = {name.casefold(): name for name in self.variables}
        ports_by_key = {name.casefold(): name for name in self.port_names}
        overlap = set(variables_by_key) & set(ports_by_key)
        if overlap:
            names = sorted(variables_by_key[key] for key in overlap)
            raise ValueError(f"development ports cannot overwrite project variables: {', '.join(names)}")
        return self


class CodeProject(BaseModel):
    schema_version: int = 1
    id: str
    name: str = Field(min_length=1, max_length=120)
    folders: list[ProjectFolder] = Field(min_length=1, max_length=24)
    playbook: PlaybookReference | None = None
    skill_sources: list[ProjectSkillSource] = Field(default_factory=list, max_length=24)
    connections: list[ProjectConnection] = Field(default_factory=list, max_length=24)
    environment: ProjectEnvironment = Field(default_factory=ProjectEnvironment)
    default_engine_id: str = "codex"
    default_model: str = "gpt"
    permission_mode: PermissionMode = PermissionMode.supervised
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_unique_folders(self) -> CodeProject:
        ids = [folder.id for folder in self.folders]
        paths = [folder.path.casefold() for folder in self.folders]
        if len(ids) != len(set(ids)):
            raise ValueError("project folder ids must be unique")
        if len(paths) != len(set(paths)):
            raise ValueError("the same folder cannot be added twice")
        connections = [(item.provider, item.name) for item in self.connections]
        if len(connections) != len(set(connections)):
            raise ValueError("the same connection cannot be added twice")
        source_ids = [item.source_id for item in self.skill_sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("the same skill source cannot be added twice")
        return self


class ProjectCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    folders: list[ProjectFolder] = Field(min_length=1, max_length=24)
    connections: list[ProjectConnection] = Field(default_factory=list, max_length=24)
    environment: ProjectEnvironment = Field(default_factory=ProjectEnvironment)
    default_engine_id: str = "codex"
    default_model: str = "gpt"
    permission_mode: PermissionMode = PermissionMode.supervised


class ProjectUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    folders: list[ProjectFolder] | None = Field(default=None, min_length=1, max_length=24)
    playbook: PlaybookReference | None = None
    connections: list[ProjectConnection] | None = Field(default=None, max_length=24)
    environment: ProjectEnvironment | None = None
    default_engine_id: str | None = None
    default_model: str | None = None
    permission_mode: PermissionMode | None = None


class ProjectPage(BaseModel):
    items: list[CodeProject]


class PlaybookItem(BaseModel):
    kind: Literal["skill", "instructions", "workflow"]
    name: str
    path: str
    description: str = ""
    enabled: bool = True


class PlaybookStatus(BaseModel):
    configured: bool
    current_revision: str | None = None
    available_revision: str | None = None
    update_available: bool = False
    items: list[PlaybookItem] = Field(default_factory=list)
    diff: str = ""
    error: str | None = None


class PlaybookConfigureRequest(BaseModel):
    repository: str = Field(min_length=1, max_length=32_768)
    branch: str = Field(default="main", min_length=1, max_length=255)


class PlaybookItemsRequest(BaseModel):
    enabled_paths: list[str] = Field(default_factory=list, max_length=250)


class SourceContextRequest(BaseModel):
    provider: Literal["github", "linear", "slack"]
    kind: Literal["issue", "pull_request", "conversation"]
    url: str = Field(min_length=1, max_length=8_192)
    connection_name: str | None = Field(default=None, max_length=512)


class PublishRequest(BaseModel):
    provider: Literal["github", "linear", "slack"]
    action: Literal["progress", "result"]
    target_url: str = Field(min_length=1, max_length=8_192)
    text: str = Field(min_length=1, max_length=100_000)
    connection_name: str | None = Field(default=None, max_length=512)
    confirmed: bool = False


class DraftPullRequestSpec(BaseModel):
    folder_id: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=256)
    body: str = Field(default="", max_length=100_000)


class DraftPullRequestRequest(BaseModel):
    title: str = Field(min_length=1, max_length=256)
    body: str = Field(default="", max_length=100_000)
    drafts: list[DraftPullRequestSpec] = Field(default_factory=list, max_length=24)
    connection_name: str | None = Field(default=None, max_length=512)
    confirmed: bool = False

    @field_validator("drafts")
    @classmethod
    def validate_unique_drafts(cls, value: list[DraftPullRequestSpec]) -> list[DraftPullRequestSpec]:
        folder_ids = [draft.folder_id for draft in value]
        if len(folder_ids) != len(set(folder_ids)):
            raise ValueError("each folder can have only one pull request specification")
        return value


class PullRequestActionRequest(BaseModel):
    action: Literal["ready", "merge"]
    target_url: str = Field(min_length=1, max_length=8_192)
    connection_name: str | None = Field(default=None, max_length=512)
    confirmed: bool = False


class PullRequestCheck(BaseModel):
    name: str = Field(max_length=512)
    state: Literal["passing", "failing", "pending", "neutral"]
    url: str = Field(default="", max_length=8_192)


class PullRequestFeedback(BaseModel):
    id: str = Field(default="", max_length=512)
    author: str = Field(default="", max_length=512)
    state: str = Field(default="commented", max_length=120)
    body: str = Field(default="", max_length=20_000)
    url: str = Field(default="", max_length=8_192)
    path: str = Field(default="", max_length=4_096)
    created_at: str = Field(default="", max_length=120)


class PullRequestStatus(BaseModel):
    state: Literal["draft", "open", "merged", "closed"]
    review_state: Literal["approved", "changes_requested", "pending", "none"]
    ci_state: Literal["passing", "failing", "pending", "none"]
    number: int | None = None
    title: str = Field(default="", max_length=512)
    url: str = Field(default="", max_length=8_192)
    updated_at: str = Field(default="", max_length=120)
    checks: list[PullRequestCheck] = Field(default_factory=list, max_length=200)
    feedback: list[PullRequestFeedback] = Field(default_factory=list, max_length=200)
    detail: str = ""


class DeliveryPlanItem(BaseModel):
    folder_id: str
    folder_name: str
    workspace_path: str
    remote_url: str | None = None
    base_branch: str | None = None
    task_branch: str | None = None
    status: Literal["ready", "needs_commit", "no_changes", "unavailable", "published"]
    detail: str = ""
    external_url: str | None = None
    connection_name: str | None = None
    pull_request_status: PullRequestStatus | None = None
    status_error: str | None = None


class IntegrationStatus(BaseModel):
    provider: Literal["github", "linear", "slack"]
    connection_name: str
    label: str
    status: Literal["connected", "reconnect", "missing"]
    detail: str = ""


class DeliveryPlan(BaseModel):
    items: list[DeliveryPlanItem] = Field(default_factory=list)
    integrations: list[IntegrationStatus] = Field(default_factory=list)
