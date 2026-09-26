from __future__ import annotations

from pathlib import Path

from cowork.coding.contracts import DiffFile
from cowork.coding.git_transport import validate_git_source
from cowork.coding.project_models import CodeProject, RepositoryResource
from cowork.coding.repository_setup_models import RepositoryStatus, TaskRepositorySetup
from cowork.coding.workspace import WorkspaceError, WorkspaceManager


def task_project(
    project: CodeProject, setup: TaskRepositorySetup, resource_ids: list[str] | None,
    *, local_computer_id: str | None = None,
) -> CodeProject:
    """Validate scope before any task records or worktrees are created."""
    selected = (
        set(resource_ids)
        if resource_ids is not None
        else {item.id for item in project.resources}
    )
    repositories = {
        item.id
        for item in project.resources
        if isinstance(item, RepositoryResource) and item.id in selected
    }
    if set(setup.base_branches) - repositories:
        raise WorkspaceError(
            "Branch choices must refer to repositories included in this task"
        )
    if not repositories and (setup.branch or setup.include_local_changes):
        raise WorkspaceError(
            "Include a repository to choose a task branch or local changes"
        )
    resources = []
    for item in project.resources:
        updates = {}
        if item.id in setup.base_branches:
            updates["default_branch"] = setup.base_branches[item.id]
        if (
            local_computer_id is not None
            and item.id in selected
            and isinstance(item, RepositoryResource)
            and item.local_path
            and item.computer_id is None
        ):
            # A locally added Git repo with an origin remains portable in the
            # project. Bind only this local task's snapshot so runtime routing
            # preserves the checkout whose branches/changes the user inspected.
            updates["computer_id"] = local_computer_id
        resources.append(item.model_copy(update=updates))
    return CodeProject.model_validate(
        {
            **project.model_dump(),
            "resources": resources,
        }
    )


class RepositorySetupService:
    """Read local checkout state without fetching, switching branches or touching its index."""

    def __init__(self, workspaces: WorkspaceManager, computer_id: str):
        self.workspaces = workspaces
        self.computer_id = computer_id

    def _path(self, resource: RepositoryResource) -> Path | None:
        if not resource.local_path or resource.computer_id not in {
            None,
            self.computer_id,
        }:
            return None
        return Path(resource.local_path).expanduser()

    def status(self, project: CodeProject) -> list[RepositoryStatus]:
        result = []
        for resource in project.resources:
            if not isinstance(resource, RepositoryResource):
                continue
            item = RepositoryStatus(resource_id=resource.id)
            path = self._path(resource)
            if path is None or (not path.is_dir() and resource.source_url):
                item.branches = (
                    [resource.default_branch] if resource.default_branch else []
                )
                item.detail = (
                    "Downloaded when the task starts"
                    if resource.source_url
                    else "On another computer"
                )
            else:
                try:
                    inspection = self.workspaces.inspect(str(path))
                    if not inspection.is_git:
                        raise WorkspaceError(
                            "This checkout is unavailable or has no Git repository"
                        )
                    item.local = True
                    item.branch = inspection.branch
                    root = Path(inspection.repository_root or path)
                    item.branches = self.workspaces.git.run(
                        root,
                        "for-each-ref",
                        "--format=%(refname:short)",
                        "refs/heads",
                        "refs/remotes",
                    ).stdout.splitlines()
                    item.branches = [
                        name for name in item.branches if not name.endswith("/HEAD")
                    ]
                    changes = self.workspaces.source_change_paths(root)
                    item.changes = changes[:250]
                    item.change_count = len(changes)
                    if not inspection.revision:
                        item.detail = "No commits yet — this folder will be copied"
                except WorkspaceError as exc:
                    item.available = False
                    item.detail = str(exc)
            result.append(item)
        return result

    def diff(self, project: CodeProject, resource_id: str) -> list[DiffFile]:
        resource = next(
            (item for item in project.resources if item.id == resource_id), None
        )
        if not isinstance(resource, RepositoryResource):
            raise WorkspaceError("Choose a repository in this project")
        path = self._path(resource)
        if path is None:
            raise WorkspaceError(
                "Local changes can only be reviewed on their original computer"
            )
        inspection = self.workspaces.inspect(str(path))
        if not inspection.is_git:
            raise WorkspaceError("This checkout is unavailable")
        if not inspection.revision:
            raise WorkspaceError(
                "This repository has no commits yet. Review its files in the original folder"
            )
        return self.workspaces.diff(str(path), inspection.revision)

    def branches(self, project: CodeProject, resource_id: str) -> list[str]:
        resource = next(
            (item for item in project.resources if item.id == resource_id), None
        )
        if not isinstance(resource, RepositoryResource):
            raise WorkspaceError("Choose a repository in this project")
        path = self._path(resource)
        if path is not None and path.is_dir():
            return self.workspaces.git.run(
                path,
                "for-each-ref",
                "--format=%(refname:short)",
                "refs/heads",
                "refs/remotes",
            ).stdout.splitlines()
        if not resource.source_url:
            raise WorkspaceError("This repository is unavailable on this computer")
        result = self.workspaces.git.run(
            self.workspaces.root,
            "ls-remote",
            "--heads",
            validate_git_source(resource.source_url),
            timeout=20,
        )
        return sorted(
            {
                line.split("\t", 1)[1].removeprefix("refs/heads/")
                for line in result.stdout.splitlines()
                if "\trefs/heads/" in line
            }
        )
