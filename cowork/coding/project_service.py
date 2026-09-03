from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path

from cowork.coding.project_models import (
    CodeProject,
    LocalFolderResource,
    ProjectCreateRequest,
    ProjectFolder,
    ProjectPage,
    ProjectResource,
    ProjectUpdateRequest,
    RepositoryResource,
)
from cowork.coding.project_store import CodeProjectStore
from cowork.coding.workspace import WorkspaceError, WorkspaceManager


class CodeProjectService:
    """Own project lifecycle and folder validity independently of coding agents."""

    def __init__(
        self,
        root: Path,
        store: CodeProjectStore | None = None,
        workspaces: WorkspaceManager | None = None,
        validator: Callable[[CodeProject], None] | None = None,
        computer_id: str = "local",
    ) -> None:
        self.store = store or CodeProjectStore(root, computer_id)
        self.workspaces = workspaces or WorkspaceManager(root)
        self.validator = validator
        self.computer_id = computer_id
        self._upgrade_legacy_resources()

    def list(self) -> ProjectPage:
        return ProjectPage(items=self.store.list())

    def get(self, project_id: str) -> CodeProject:
        return self.store.get(project_id)

    def create(self, request: ProjectCreateRequest) -> CodeProject:
        resources = request.resources or self._resources_from_folders(request.folders or [])
        project = CodeProject(
            id=str(uuid.uuid4()),
            name=self._name(request.name),
            resources=resources,
            connections=request.connections,
            environment=request.environment,
            skill_sources=request.skill_sources,
            default_engine_id=request.default_engine_id,
            default_model=request.default_model,
            permission_mode=request.permission_mode,
        )
        return self.store.create(self._validated(project))

    def update(self, project_id: str, request: ProjectUpdateRequest) -> CodeProject:
        project = self.get(project_id)
        values = {name: getattr(request, name) for name in request.model_fields_set}
        if "folders" in values and values["folders"] is not None:
            values["resources"] = self._resources_from_folders(values.pop("folders"), project.resources)
        if "name" in values and values["name"] is not None:
            values["name"] = self._name(values["name"])
        return self.store.save(self._validated(CodeProject.model_validate({
            **project.model_dump(mode="python"),
            **values,
        })))

    def delete(self, project_id: str, active_session_count: int) -> None:
        if active_session_count:
            raise WorkspaceError("Delete this project's coding tasks before deleting the Code Project")
        self.store.delete(project_id)

    def inspect_folders(self, project_id: str):
        project = self.get(project_id)
        result = []
        resources = {resource.id: resource for resource in project.resources}
        for folder in project.folders:
            resource = resources[folder.id]
            if isinstance(resource, RepositoryResource) and not resource.local_path:
                result.append({
                    "folder": folder,
                    "inspection": {
                        "path": resource.source_url or resource.name,
                        "exists": True,
                        "is_directory": True,
                        "is_git": True,
                        "dirty": False,
                    },
                    "base_branch_available": True,
                })
                continue
            inspection = self.workspaces.inspect(folder.path)
            base_branch_available = True
            if inspection.is_git and folder.base_branch:
                root = Path(inspection.repository_root or inspection.path)
                base_branch_available = self.workspaces.branch_revision(root, folder.base_branch) is not None
            result.append({
                "folder": folder,
                "inspection": inspection,
                "base_branch_available": base_branch_available,
            })
        return result

    def resolve_local_resource(self, folder: ProjectFolder) -> ProjectResource:
        """Classify a user-selected path without making the renderer Git-aware."""
        return self._resources_from_folders([folder])[0]

    def _normalize(self, project: CodeProject) -> CodeProject:
        normalized: list[ProjectResource] = []
        seen: set[str] = set()
        seen_repositories: set[str] = set()
        for resource in project.resources:
            raw_path = resource.local_path if isinstance(resource, RepositoryResource) else resource.path
            if not raw_path:
                normalized.append(resource)
                continue
            inspection = self.workspaces.inspect(raw_path)
            if not inspection.exists or not inspection.is_directory:
                raise WorkspaceError(f"Folder is unavailable: {raw_path}")
            path = str(Path(inspection.path).resolve())
            key = (resource.source_url or path).casefold() if isinstance(resource, RepositoryResource) else path.casefold()
            if key in seen:
                raise WorkspaceError("The same project resource cannot be added twice")
            seen.add(key)
            if isinstance(resource, RepositoryResource):
                if inspection.is_git:
                    repository = str(
                        Path(inspection.repository_root or inspection.path).resolve()
                    ).casefold()
                    if repository in seen_repositories:
                        raise WorkspaceError(
                            "A Git repository can only be added once; choose the repository root"
                        )
                    seen_repositories.add(repository)
                normalized.append(resource.model_copy(update={"local_path": path}))
            else:
                normalized.append(resource.model_copy(update={"path": path}))
        return CodeProject.model_validate({
            **project.model_dump(mode="python"),
            "resources": normalized,
        })

    def _validated(self, project: CodeProject) -> CodeProject:
        normalized = self._normalize(project)
        if self.validator:
            self.validator(normalized)
        return normalized

    @staticmethod
    def _name(value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Code Project name cannot be empty")
        return normalized[:120]

    def _resources_from_folders(
        self,
        folders: list[ProjectFolder],
        existing: list[ProjectResource] | None = None,
    ) -> list[ProjectResource]:
        existing_by_id = {resource.id: resource for resource in existing or []}
        resources: list[ProjectResource] = []
        seen_repositories: set[str] = set()
        for folder in folders:
            current = existing_by_id.get(folder.id)
            inspection = self.workspaces.inspect(folder.path)
            if inspection.is_git:
                root = Path(inspection.repository_root or inspection.path)
                repository = str(root.resolve()).casefold()
                if repository in seen_repositories:
                    raise WorkspaceError(
                        "A Git repository can only be added once; choose the repository root"
                    )
                seen_repositories.add(repository)
                remote = self.workspaces.git.run(root, "config", "--get", "remote.origin.url", check=False).stdout.strip() or None
                resources.append(RepositoryResource(
                    id=folder.id,
                    name=folder.name,
                    source_url=remote,
                    local_path=str(root.resolve()),
                    computer_id=None if remote else self.computer_id,
                    default_branch=folder.base_branch,
                    checkout_strategy=current.checkout_strategy if isinstance(current, RepositoryResource) else "worktree",
                    commands=folder.commands,
                ))
            else:
                resources.append(LocalFolderResource(
                    id=folder.id,
                    name=folder.name,
                    path=folder.path,
                    computer_id=current.computer_id if isinstance(current, LocalFolderResource) else self.computer_id,
                    commands=folder.commands,
                ))
        return resources

    def _upgrade_legacy_resources(self) -> None:
        """Idempotently classify schema-1 folders without losing local paths."""
        for project in self.store.list():
            if project.schema_version == 2 and any(isinstance(item, RepositoryResource) for item in project.resources):
                continue
            resources = self._resources_from_folders(project.folders, project.resources)
            if resources != project.resources or project.schema_version != 2:
                self.store.save(CodeProject.model_validate({
                    **project.model_dump(mode="python"),
                    "schema_version": 2,
                    "resources": resources,
                }))
