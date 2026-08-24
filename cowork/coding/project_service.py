from __future__ import annotations

import uuid
from pathlib import Path

from cowork.coding.project_models import (
    CodeProject,
    ProjectCreateRequest,
    ProjectPage,
    ProjectUpdateRequest,
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
    ) -> None:
        self.store = store or CodeProjectStore(root)
        self.workspaces = workspaces or WorkspaceManager(root)

    def list(self) -> ProjectPage:
        return ProjectPage(items=self.store.list())

    def get(self, project_id: str) -> CodeProject:
        return self.store.get(project_id)

    def create(self, request: ProjectCreateRequest) -> CodeProject:
        project = CodeProject(
            id=str(uuid.uuid4()),
            name=self._name(request.name),
            folders=request.folders,
            connections=request.connections,
            environment=request.environment,
            default_engine_id=request.default_engine_id,
            default_model=request.default_model,
            permission_mode=request.permission_mode,
        )
        return self.store.create(self._normalize(project))

    def update(self, project_id: str, request: ProjectUpdateRequest) -> CodeProject:
        project = self.get(project_id)
        # Keep nested ProjectFolder/ProjectEnvironment instances typed. A
        # model_dump here turns them into dictionaries; model_copy deliberately
        # does not revalidate updates, so normalization would later crash on
        # ``folder.path`` during an ordinary settings save.
        values = {name: getattr(request, name) for name in request.model_fields_set}
        if "name" in values:
            values["name"] = self._name(values["name"])
        return self.store.save(self._normalize(project.model_copy(update=values)))

    def delete(self, project_id: str, active_session_count: int) -> None:
        if active_session_count:
            raise WorkspaceError("Delete this project's coding tasks before deleting the Code Project")
        self.store.delete(project_id)

    def inspect_folders(self, project_id: str):
        project = self.get(project_id)
        result = []
        for folder in project.folders:
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

    def _normalize(self, project: CodeProject) -> CodeProject:
        normalized = []
        seen: set[str] = set()
        for folder in project.folders:
            inspection = self.workspaces.inspect(folder.path)
            if not inspection.exists or not inspection.is_directory:
                raise WorkspaceError(f"Folder is unavailable: {folder.path}")
            path = str(Path(inspection.path).resolve())
            key = path.casefold()
            if key in seen:
                raise WorkspaceError("The same folder cannot be added twice")
            seen.add(key)
            normalized.append(folder.model_copy(update={"path": path}))
        return project.model_copy(update={"folders": normalized})

    @staticmethod
    def _name(value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Code Project name cannot be empty")
        return normalized[:120]
