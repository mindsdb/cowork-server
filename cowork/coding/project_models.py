from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

# DeliveryRecord and SourceContext remain available from this long-standing
# public model module for compatibility with existing callers.
from cowork.coding.contracts import (  # noqa: F401
    DeliveryRecord,
    PermissionMode,
    SourceContext,
    utc_now,
)
from cowork.coding.git_transport import validate_git_source
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


class RepositoryResource(BaseModel):
    kind: Literal["repository"] = "repository"
    id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=120)
    source_url: str | None = Field(default=None, max_length=8_192)
    provider: Literal["github", "gitlab", "bitbucket", "git"] = "git"
    repository: str | None = Field(default=None, max_length=512)
    connector_name: str | None = Field(default=None, max_length=512)
    local_path: str | None = Field(default=None, max_length=32_768)
    computer_id: str | None = Field(default=None, max_length=128)
    default_branch: str | None = Field(default=None, max_length=255)
    checkout_strategy: Literal["worktree", "clone"] = "worktree"
    commands: list[ProjectCommand] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def require_source(self) -> RepositoryResource:
        if not self.source_url and not self.local_path:
            raise ValueError("repository resources require a remote URL or local checkout")
        if not self.source_url and not self.computer_id:
            raise ValueError("a local-only repository must identify its computer")
        if self.source_url:
            self.source_url = validate_git_source(self.source_url)
        if self.source_url and not self.repository:
            provider, identity = repository_identity(self.source_url)
            self.provider = provider
            self.repository = identity
        return self


class LocalFolderResource(BaseModel):
    kind: Literal["local_folder"] = "local_folder"
    id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=120)
    path: str = Field(min_length=1, max_length=32_768)
    computer_id: str = Field(min_length=1, max_length=128)
    commands: list[ProjectCommand] = Field(default_factory=list, max_length=32)


ProjectResource = Annotated[
    RepositoryResource | LocalFolderResource,
    Field(discriminator="kind"),
]


def repository_identity(source_url: str) -> tuple[Literal["github", "gitlab", "bitbucket", "git"], str]:
    """Return a stable provider and repository identity without credentials."""

    normalized = source_url.strip().removesuffix(".git").rstrip("/")
    if normalized.startswith("git@") and ":" in normalized:
        host, path = normalized[4:].split(":", 1)
    else:
        parsed = urlparse(normalized)
        host = parsed.hostname or ""
        path = parsed.path.lstrip("/")
    provider: Literal["github", "gitlab", "bitbucket", "git"] = "git"
    if host.casefold() in {"github.com", "www.github.com"}:
        provider = "github"
    elif host.casefold() in {"gitlab.com", "www.gitlab.com"}:
        provider = "gitlab"
    elif host.casefold() in {"bitbucket.org", "www.bitbucket.org"}:
        provider = "bitbucket"
    return provider, path or normalized


def resource_folder(resource: ProjectResource) -> ProjectFolder:
    if isinstance(resource, RepositoryResource):
        return ProjectFolder(
            id=resource.id,
            name=resource.name,
            # Remote-only repositories deliberately have no durable machine
            # path.  The legacy projection still requires a non-empty display
            # value; execution code resolves a real checkout before using it.
            path=resource.local_path or resource.source_url or resource.name,
            base_branch=resource.default_branch,
            commands=resource.commands,
        )
    return ProjectFolder(
        id=resource.id,
        name=resource.name,
        path=resource.path,
        commands=resource.commands,
    )


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
    schema_version: int = 2
    id: str
    name: str = Field(min_length=1, max_length=120)
    resources: list[ProjectResource] = Field(min_length=1, max_length=24)
    # Compatibility projection used by the existing workspace/review layer.
    # New durable code must use ``resources`` as its source of truth.
    folders: list[ProjectFolder] = Field(default_factory=list, max_length=24)
    playbook: PlaybookReference | None = None
    skill_sources: list[ProjectSkillSource] = Field(default_factory=list, max_length=24)
    connections: list[ProjectConnection] = Field(default_factory=list, max_length=24)
    environment: ProjectEnvironment = Field(default_factory=ProjectEnvironment)
    default_engine_id: str = "codex"
    default_model: str = "gpt"
    permission_mode: PermissionMode = PermissionMode.supervised
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_folders(cls, value):
        if not isinstance(value, dict) or value.get("resources") or not value.get("folders"):
            return value
        value = dict(value)
        computer_id = value.pop("_migration_computer_id", None) or "local"
        folders = [
            folder.model_dump(mode="python") if isinstance(folder, ProjectFolder) else folder
            for folder in value["folders"]
        ]
        # A legacy base branch is repository metadata. Preserve that signal in
        # the first typed representation instead of projecting every folder to
        # ``LocalFolderResource`` and irreversibly dropping it before the
        # service can inspect the checkout.
        value["resources"] = [
            {
                **(
                    {
                        "kind": "repository",
                        "local_path": folder["path"],
                        "default_branch": folder.get("base_branch"),
                    }
                    if folder.get("base_branch")
                    else {"kind": "local_folder", "path": folder["path"]}
                ),
                "id": folder["id"],
                "name": folder["name"],
                "computer_id": computer_id,
                "commands": folder.get("commands", []),
            }
            for folder in folders
        ]
        value["schema_version"] = 2
        return value

    @model_validator(mode="after")
    def validate_unique_resources(self) -> CodeProject:
        ids = [resource.id for resource in self.resources]
        locations = [
            (resource.source_url or resource.local_path or "").casefold()
            if isinstance(resource, RepositoryResource)
            else resource.path.casefold()
            for resource in self.resources
        ]
        if len(ids) != len(set(ids)):
            raise ValueError("project resource ids must be unique")
        if len(locations) != len(set(locations)):
            raise ValueError("the same folder cannot be added twice")
        github_connections = [item.name for item in self.connections if item.provider == "github"]
        if len(github_connections) == 1:
            for resource in self.resources:
                if (
                    isinstance(resource, RepositoryResource)
                    and resource.provider == "github"
                    and not resource.connector_name
                ):
                    resource.connector_name = github_connections[0]
        self.folders = [resource_folder(resource) for resource in self.resources]
        connections = [(item.provider, item.name) for item in self.connections]
        if len(connections) != len(set(connections)):
            raise ValueError("the same connection cannot be added twice")
        source_ids = [item.source_id for item in self.skill_sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("the same skill source cannot be added twice")
        return self


class ProjectCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    resources: list[ProjectResource] | None = Field(default=None, min_length=1, max_length=24)
    folders: list[ProjectFolder] | None = Field(default=None, min_length=1, max_length=24)
    connections: list[ProjectConnection] = Field(default_factory=list, max_length=24)
    environment: ProjectEnvironment = Field(default_factory=ProjectEnvironment)
    skill_sources: list[ProjectSkillSource] = Field(default_factory=list, max_length=24)
    default_engine_id: str = "codex"
    default_model: str = "gpt"
    permission_mode: PermissionMode = PermissionMode.supervised

    @model_validator(mode="after")
    def require_resources(self) -> ProjectCreateRequest:
        if bool(self.resources) == bool(self.folders):
            raise ValueError("provide exactly one project resource list")
        return self


class ProjectUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    folders: list[ProjectFolder] | None = Field(default=None, min_length=1, max_length=24)
    resources: list[ProjectResource] | None = Field(default=None, min_length=1, max_length=24)
    playbook: PlaybookReference | None = None
    connections: list[ProjectConnection] | None = Field(default=None, max_length=24)
    environment: ProjectEnvironment | None = None
    skill_sources: list[ProjectSkillSource] | None = Field(default=None, max_length=24)
    default_engine_id: str | None = None
    default_model: str | None = None
    permission_mode: PermissionMode | None = None

    @model_validator(mode="after")
    def reject_duplicate_resource_inputs(self) -> ProjectUpdateRequest:
        if self.resources is not None and self.folders is not None:
            raise ValueError("update resources or legacy folders, not both")
        return self


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


class WorkItemSearchRequest(BaseModel):
    provider: Literal["github", "linear"]
    query: str = Field(default="", max_length=256)
    connection_name: str | None = Field(default=None, max_length=512)
    limit: int = Field(default=20, ge=1, le=50)


class WorkItemSummary(BaseModel):
    provider: Literal["github", "linear"]
    kind: Literal["issue", "pull_request"]
    url: str = Field(min_length=1, max_length=8_192)
    title: str = Field(default="", max_length=512)
    external_id: str = Field(default="", max_length=512)
    state: str = Field(default="", max_length=120)
    scope: str = Field(default="", max_length=512)
    assignee: str = Field(default="", max_length=512)
    updated_at: str = Field(default="", max_length=120)
    connection_name: str = Field(min_length=1, max_length=512)


class WorkItemPage(BaseModel):
    items: list[WorkItemSummary] = Field(default_factory=list, max_length=50)
    incomplete: bool = False


class PublishRequest(BaseModel):
    provider: Literal["github", "linear", "slack"]
    action: Literal["progress", "result"]
    target_url: str = Field(min_length=1, max_length=8_192)
    text: str = Field(min_length=1, max_length=100_000)
    connection_name: str | None = Field(default=None, max_length=512)
    confirmed: bool = False


class SourceActionRequest(BaseModel):
    provider: Literal["github", "linear"]
    action: Literal["complete"]
    target_url: str = Field(min_length=1, max_length=8_192)
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
    action: Literal["ready", "merge", "resolve_thread"]
    target_url: str = Field(min_length=1, max_length=8_192)
    connection_name: str | None = Field(default=None, max_length=512)
    thread_id: str | None = Field(default=None, max_length=512)
    confirmed: bool = False

    @model_validator(mode="after")
    def require_thread_identity(self) -> PullRequestActionRequest:
        if (self.action == "resolve_thread") != bool(self.thread_id):
            raise ValueError("resolve_thread actions require exactly one review thread identity")
        return self


class PullRequestAnnotation(BaseModel):
    path: str = Field(default="", max_length=4_096)
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    level: Literal["notice", "warning", "failure"] = "notice"
    title: str = Field(default="", max_length=512)
    message: str = Field(default="", max_length=20_000)


class PullRequestCheck(BaseModel):
    id: str = Field(default="", max_length=512)
    name: str = Field(max_length=512)
    state: Literal["passing", "failing", "pending", "neutral"]
    url: str = Field(default="", max_length=8_192)
    detail: str = Field(default="", max_length=20_000)
    annotations: list[PullRequestAnnotation] = Field(default_factory=list, max_length=50)


class PullRequestFeedback(BaseModel):
    id: str = Field(default="", max_length=512)
    author: str = Field(default="", max_length=512)
    state: str = Field(default="commented", max_length=120)
    body: str = Field(default="", max_length=20_000)
    url: str = Field(default="", max_length=8_192)
    path: str = Field(default="", max_length=4_096)
    line: int | None = Field(default=None, ge=1)
    created_at: str = Field(default="", max_length=120)
    thread_id: str = Field(default="", max_length=512)
    resolved: bool = False
    outdated: bool = False


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
