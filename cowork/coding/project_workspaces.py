from __future__ import annotations

import os
import socket
import subprocess
import threading
from dataclasses import dataclass

from cowork.coding.contracts import DiffFile, GitState, TaskWorkspace, WorkspaceKind
from cowork.coding.project_models import CodeProject, ProjectCommand, ProjectFolder
from cowork.coding.workspace import (
    PreparedWorkspace,
    WorkspaceError,
    WorkspaceManager,
    _org_mode,
)


@dataclass(frozen=True)
class CommandResult:
    command_id: str
    label: str
    folder_id: str
    phase: str
    return_code: int
    output: str


@dataclass(frozen=True)
class PreparedProjectWorkspace:
    primary: TaskWorkspace
    workspaces: tuple[TaskWorkspace, ...]
    ports: dict[str, int]


class PortAllocator:
    """Allocate collision-free task ports and retain ownership across restarts."""

    def __init__(self, first: int = 41_000, last: int = 49_999) -> None:
        self.first = first
        self.last = last
        self._lock = threading.RLock()
        self._owned: dict[int, str] = {}

    def restore(self, session_id: str, ports: dict[str, int]) -> None:
        with self._lock:
            for port in ports.values():
                self._owned.setdefault(port, session_id)

    def allocate(self, session_id: str, names: list[str]) -> dict[str, int]:
        with self._lock:
            allocated: dict[str, int] = {}
            for name in names:
                allocated[name] = self._next_available(session_id)
            return allocated

    def release(self, session_id: str) -> None:
        with self._lock:
            self._owned = {port: owner for port, owner in self._owned.items() if owner != session_id}

    def _next_available(self, session_id: str) -> int:
        for port in range(self.first, self.last + 1):
            if port in self._owned or not self._bindable(port):
                continue
            self._owned[port] = session_id
            return port
        raise WorkspaceError("No task development ports are currently available")

    @staticmethod
    def _bindable(port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


class ProjectCommandRunner:
    """Execute structured project commands without platform shell parsing."""

    def run(
        self,
        project: CodeProject,
        workspaces: tuple[TaskWorkspace, ...],
        phase: str,
        ports: dict[str, int],
    ) -> list[CommandResult]:
        if _org_mode():
            raise WorkspaceError("Local Code Project commands are not available on this deployment")
        by_id = {workspace.folder_id: workspace for workspace in workspaces}
        environment = os.environ.copy()
        environment.update(project.environment.variables)
        environment.update({name: str(value) for name, value in ports.items()})
        results: list[CommandResult] = []
        for folder in project.folders:
            workspace = by_id[folder.id]
            for command in (item for item in folder.commands if item.phase == phase):
                results.append(self._run_one(command, workspace, environment))
        return results

    @staticmethod
    def _run_one(command: ProjectCommand, workspace: TaskWorkspace, environment: dict[str, str]) -> CommandResult:
        try:
            completed = subprocess.run(
                command.argv,
                cwd=workspace.workspace_path,
                env=environment,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
                shell=False,
                timeout=600,
            )
            output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
            return CommandResult(
                command_id=command.id,
                label=command.label,
                folder_id=workspace.folder_id,
                phase=command.phase,
                return_code=completed.returncode,
                output=output[:32_000],
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return CommandResult(
                command_id=command.id,
                label=command.label,
                folder_id=workspace.folder_id,
                phase=command.phase,
                return_code=-1,
                output=str(exc)[:32_000],
            )


class ProjectWorkspaceManager:
    """Coordinate one independent task workspace across all project folders."""

    def __init__(
        self,
        workspaces: WorkspaceManager,
        ports: PortAllocator | None = None,
        commands: ProjectCommandRunner | None = None,
    ) -> None:
        self.workspaces = workspaces
        self.ports = ports or PortAllocator()
        self.commands = commands or ProjectCommandRunner()
        self._lock = threading.RLock()

    def prepare(self, session_id: str, project: CodeProject) -> PreparedProjectWorkspace:
        with self._lock:
            prepared: list[TaskWorkspace] = []
            try:
                for folder in project.folders:
                    key = self._key(session_id, folder.id)
                    item = self.workspaces.prepare(key, folder.path, True, folder.base_branch)
                    prepared.append(self._task_workspace(session_id, project, folder, item))
                ports = self.ports.allocate(session_id, project.environment.port_names)
                return PreparedProjectWorkspace(primary=prepared[0], workspaces=tuple(prepared), ports=ports)
            except Exception:
                for workspace in reversed(prepared):
                    self._cleanup_one(session_id, workspace)
                self.ports.release(session_id)
                raise

    def fork(
        self,
        session_id: str,
        project: CodeProject,
        parent_workspaces: list[TaskWorkspace],
    ) -> PreparedProjectWorkspace:
        """Copy every project workspace while preserving each original baseline."""
        with self._lock:
            parent_by_id = {workspace.folder_id: workspace for workspace in parent_workspaces}
            prepared: list[TaskWorkspace] = []
            try:
                for folder in project.folders:
                    parent = parent_by_id.get(folder.id)
                    if parent is None:
                        raise WorkspaceError(f"The task is missing its {folder.name} workspace")
                    key = self._key(session_id, folder.id)
                    item = self.workspaces.fork(
                        key,
                        parent.source_path,
                        parent.workspace_path,
                        parent.workspace_kind,
                        parent.base_revision,
                    )
                    prepared.append(self._task_workspace(session_id, project, folder, item, parent.base_branch))
                ports = self.ports.allocate(session_id, project.environment.port_names)
                return PreparedProjectWorkspace(primary=prepared[0], workspaces=tuple(prepared), ports=ports)
            except Exception:
                for workspace in reversed(prepared):
                    self._cleanup_one(session_id, workspace)
                self.ports.release(session_id)
                raise

    def restore_ports(self, session_id: str, ports: dict[str, int]) -> None:
        self.ports.restore(session_id, ports)

    def run_commands(
        self,
        project: CodeProject,
        workspaces: list[TaskWorkspace],
        phase: str,
        ports: dict[str, int],
    ) -> list[CommandResult]:
        return self.commands.run(project, tuple(workspaces), phase, ports)

    def diff(self, workspaces: list[TaskWorkspace]) -> list[DiffFile]:
        files: list[DiffFile] = []
        for workspace in workspaces:
            for file in self.workspaces.diff(workspace.workspace_path, workspace.base_revision):
                files.append(file.model_copy(update={"folder_id": workspace.folder_id, "folder_name": workspace.folder_name}))
        return files

    def git_states(self, workspaces: list[TaskWorkspace]) -> list[GitState]:
        return [
            self.workspaces.git_state(workspace.source_path, workspace.workspace_path).model_copy(
                update={"folder_id": workspace.folder_id, "folder_name": workspace.folder_name}
            )
            for workspace in workspaces
        ]

    def commit(self, workspaces: list[TaskWorkspace], message: str) -> list[GitState]:
        states: list[GitState] = []
        for workspace in workspaces:
            if workspace.workspace_kind != WorkspaceKind.git_worktree:
                continue
            state = self.workspaces.git_state(workspace.source_path, workspace.workspace_path)
            if state.dirty:
                state = self.workspaces.commit(workspace.workspace_path, message)
            states.append(state.model_copy(update={"folder_id": workspace.folder_id, "folder_name": workspace.folder_name}))
        return states

    def apply(self, session_id: str, workspaces: list[TaskWorkspace]) -> list[str]:
        """Preflight every folder before mutating any source folder."""
        with self._lock:
            plans = [
                self.workspaces.preflight_apply(
                    self._key(session_id, workspace.folder_id),
                    workspace.source_path,
                    workspace.workspace_path,
                    workspace.base_revision,
                )
                for workspace in workspaces
            ]
            applied: list[str] = []
            for workspace, plan in zip(workspaces, plans, strict=True):
                self.workspaces.apply_checked(
                    workspace.source_path,
                    workspace.workspace_path,
                    workspace.base_revision,
                    plan,
                )
                if plan:
                    applied.append(workspace.folder_name)
            return applied

    def cleanup(self, session_id: str, workspaces: list[TaskWorkspace]) -> None:
        with self._lock:
            failures: list[str] = []
            for workspace in reversed(workspaces):
                try:
                    self._cleanup_one(session_id, workspace)
                except WorkspaceError as exc:
                    failures.append(f"{workspace.folder_name}: {exc}")
            self.ports.release(session_id)
            if failures:
                raise WorkspaceError("; ".join(failures))
            self.workspaces.prune_task_root(session_id)

    def _cleanup_one(self, session_id: str, workspace: TaskWorkspace) -> None:
        self.workspaces.cleanup(
            self._key(session_id, workspace.folder_id),
            workspace.source_path,
            workspace.workspace_path,
            workspace.workspace_kind,
            workspace.base_revision,
        )

    def _task_workspace(
        self,
        session_id: str,
        project: CodeProject,
        folder: ProjectFolder,
        prepared: PreparedWorkspace,
        base_branch: str | None = None,
    ) -> TaskWorkspace:
        branch = None
        try:
            if prepared.kind == WorkspaceKind.git_worktree:
                branch = self._task_branch(project.name, session_id)
                self.workspaces.create_branch(str(prepared.workspace_path), branch)
            resolved_base_branch = base_branch or folder.base_branch
            if prepared.kind == WorkspaceKind.git_worktree and not resolved_base_branch:
                resolved_base_branch = self.workspaces.inspect(str(prepared.source_path)).branch
            return TaskWorkspace(
                folder_id=folder.id,
                folder_name=folder.name,
                source_path=str(prepared.source_path),
                workspace_path=str(prepared.workspace_path),
                workspace_kind=prepared.kind,
                repository_root=str(prepared.repository_root) if prepared.repository_root else None,
                base_revision=prepared.base_revision,
                base_branch=resolved_base_branch,
                task_branch=branch,
                source_dirty=prepared.source_dirty,
            )
        except Exception:
            self.workspaces.cleanup(
                self._key(session_id, folder.id),
                str(prepared.source_path),
                str(prepared.workspace_path),
                prepared.kind,
                prepared.base_revision,
            )
            raise

    @staticmethod
    def _key(session_id: str, folder_id: str) -> str:
        return f"{session_id}/{folder_id}"

    @staticmethod
    def _task_branch(project_name: str, session_id: str) -> str:
        slug = "-".join("".join(character.lower() if character.isalnum() else " " for character in project_name).split())
        return f"cowork/{slug[:48] or 'project'}/{session_id[:12]}"
