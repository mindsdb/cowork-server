from __future__ import annotations

import threading
import uuid
from pathlib import Path

from cowork.coding.contracts import (
    CodingEvent,
    CodingSession,
    EventType,
    SessionStatus,
    TaskWorkspace,
    utc_now,
)
from cowork.coding.engines.base import EngineCredentials
from cowork.coding.control_models import RunStatus
from cowork.coding.control_service import ControlPlaneService
from cowork.coding.operation_types import EventEmitter, GetExecutionProject, MaintenanceSession
from cowork.coding.project_workspaces import ProjectWorkspaceManager
from cowork.coding.runtime import RuntimeManager, engine_workspace_path
from cowork.coding.skill_runtime import SkillRuntimeResolver
from cowork.coding.store import CodingStore
from cowork.coding.turns import RunningTurn
from cowork.coding.workspace import WorkspaceManager


class SessionLifecycleOperations:
    """Exclusive lifecycle changes for persisted coding sessions."""

    def __init__(
        self,
        *,
        maintenance_session: MaintenanceSession,
        emit: EventEmitter,
        store: CodingStore,
        workspaces: WorkspaceManager,
        project_workspaces: ProjectWorkspaceManager,
        execution_project: GetExecutionProject,
        control: ControlPlaneService,
        skill_runtime: SkillRuntimeResolver,
        runtimes: RuntimeManager,
        running: dict[str, RunningTurn],
        lock: threading.RLock,
    ) -> None:
        self.maintenance_session = maintenance_session
        self.emit = emit
        self.store = store
        self.workspaces = workspaces
        self.project_workspaces = project_workspaces
        self.execution_project = execution_project
        self.control = control
        self.skill_runtime = skill_runtime
        self.runtimes = runtimes
        self.running = running
        self.lock = lock

    def delete_session(self, session_id: str) -> None:
        runtime_lock = self.runtimes.session_lock(session_id)
        with runtime_lock, self.maintenance_session(
            session_id,
            "Stop the active turn before deleting this coding task",
        ) as session:
            self.runtimes.close_locked(session_id)
            if session.workspaces:
                self.project_workspaces.cleanup(session.id, session.workspaces)
            else:
                self.workspaces.cleanup(
                    session.id,
                    session.source_path,
                    session.workspace_path,
                    session.workspace_kind,
                    session.base_revision,
                )
            self.store.delete_session(session.id)
            self.skill_runtime.cleanup(session.id)

    def rename_session(self, session_id: str, title: str) -> CodingSession:
        normalized = " ".join(title.split())
        if not normalized:
            raise ValueError("Task name cannot be empty")
        return self.store.update_session(
            session_id,
            lambda current: setattr(current, "title", normalized[:200]),
        )

    def set_archived(self, session_id: str, archived: bool) -> CodingSession:
        with self.lock:
            if session_id in self.running:
                raise RuntimeError("Stop the active turn before archiving this coding task")
            return self.store.update_session(
                session_id,
                lambda current: setattr(current, "archived", archived),
            )

    def set_pinned(self, session_id: str, pinned: bool) -> CodingSession:
        """Persist navigation metadata without making the task look newly active."""
        return self.store.update_session(
            session_id,
            lambda current: setattr(current, "pinned", pinned),
            touch_updated_at=False,
        )

    def fork_session(self, session_id: str, credentials: EngineCredentials) -> CodingSession:
        parent_lock = self.runtimes.session_lock(session_id)
        with parent_lock:  # noqa: SIM117 - the maintenance reservation must be acquired second.
            with self.maintenance_session(
                session_id,
                "Wait for the active turn to finish before forking this coding task",
            ) as parent:
                return self._fork_reserved(parent, credentials)

    def _fork_reserved(self, parent: CodingSession, credentials: EngineCredentials) -> CodingSession:
        new_id = str(uuid.uuid4())
        project = self.execution_project(parent)
        parent_task = self.control.store.get_task(parent.task_id) if parent.task_id else None
        control_snapshot = self.control.create_task_run(
            task_id=new_id,
            title=f"{parent.title} (fork)"[:200],
            prompt=parent_task.prompt if parent_task else "",
            project=project,
            requested_resource_ids=None,
            computer_id=parent.computer_id,
            engine_id=parent.engine_id,
            standalone_computer_id=self.control.local_computer.id if project is None else None,
        )
        try:
            self.control.set_run_status(control_snapshot.run.id, RunStatus.preparing)
        except Exception:
            self.control.delete_task(control_snapshot.run.id)
            raise
        if project and parent.workspaces:
            try:
                prepared_project = self.project_workspaces.fork(
                    new_id,
                    project.name,
                    parent.workspaces,
                    list(parent.allocated_ports),
                )
            except Exception:
                self.control.delete_task(control_snapshot.run.id)
                raise
            prepared = prepared_project.primary
            child_workspaces = list(prepared_project.workspaces)
            child_ports = prepared_project.ports
            parent_project_paths = {item.workspace_path for item in parent.workspaces[1:]}
            external_dirs = [path for path in parent.additional_dirs if path not in parent_project_paths]
            child_dirs = [item.workspace_path for item in child_workspaces[1:]] + external_dirs
            inherited_environment = {
                name: value
                for name, value in parent.environment.items()
                if name not in parent.allocated_ports
            }
            child_environment = {
                **inherited_environment,
                **{name: str(port) for name, port in child_ports.items()},
            }
            instructions = self._forked_instructions(parent, child_workspaces)
        else:
            try:
                prepared = self.workspaces.fork(
                    new_id,
                    parent.source_path,
                    parent.workspace_path,
                    parent.workspace_kind,
                    parent.base_revision,
                )
            except Exception:
                self.control.delete_task(control_snapshot.run.id)
                raise
            child_workspaces = []
            child_ports = {}
            child_dirs = parent.additional_dirs
            child_environment = parent.environment
            instructions = parent.developer_instructions

        prepared_kind = prepared.workspace_kind if isinstance(prepared, TaskWorkspace) else prepared.kind
        prepared_warning = None if isinstance(prepared, TaskWorkspace) else prepared.warning
        control_workspaces = child_workspaces or [
            TaskWorkspace(
                folder_id=parent.resource_ids[0] if parent.resource_ids else "folder",
                folder_name=Path(parent.source_path).name or "Folder",
                source_path=str(prepared.source_path),
                workspace_path=str(prepared.workspace_path),
                workspace_kind=prepared_kind,
                repository_root=str(prepared.repository_root) if prepared.repository_root else None,
                base_revision=prepared.base_revision,
                source_dirty=prepared.source_dirty,
            )
        ]
        try:
            child_skill_roots = self.skill_runtime.clone(parent.id, new_id)
            parent_runtime = self.runtimes.open_locked(parent, credentials)
            child = self.store.load_session(parent.id).model_copy(
                update={
                    "id": new_id,
                    "title": f"{parent.title} (fork)"[:200],
                    "task_id": control_snapshot.task.id,
                    "run_id": control_snapshot.run.id,
                    "computer_id": control_snapshot.computer.id,
                    "runtime_epoch": control_snapshot.run.epoch,
                    "resource_ids": (
                        [item.id for item in project.resources]
                        if project
                        else [control_workspaces[0].folder_id]
                    ),
                    "scope_all_project_resources": True,
                    "workspace_path": str(prepared.workspace_path),
                    "workspace_kind": prepared_kind,
                    "repository_root": str(prepared.repository_root) if prepared.repository_root else None,
                    "base_revision": prepared.base_revision,
                    "source_dirty": prepared.source_dirty,
                    "workspace_warning": prepared_warning,
                    "workspaces": child_workspaces,
                    "additional_dirs": child_dirs,
                    "allocated_ports": child_ports,
                    "environment": child_environment,
                    "developer_instructions": instructions,
                    "skill_roots": child_skill_roots,
                    "engine_session_id": None,
                    "active_turn_id": None,
                    "pending_approval": None,
                    "queued_instructions": [],
                    "terminal_tabs": [],
                    "pinned": False,
                    "archived": False,
                    "status": SessionStatus.completed,
                    "last_error": None,
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                }
            )
            child.engine_session_id = parent_runtime.fork(
                engine_workspace_path(child),
                tuple(child.additional_dirs),
            )
            self.store.save_session(child)
            self.control.attach_prepared_workspaces(
                control_snapshot.run.id,
                control_workspaces,
            )
            self.control.set_run_status(control_snapshot.run.id, RunStatus.ready)
            self.control.set_run_status(control_snapshot.run.id, RunStatus.completed)
            self.store.copy_event_history(parent.id, child)
            self.emit(
                child.id,
                CodingEvent(
                    type=EventType.session,
                    title="Task forked",
                    text=f"Forked from {parent.title} with its conversation and working changes.",
                    phase="completed",
                    data={"parentSessionId": parent.id},
                ),
            )
            return self.store.load_session(child.id)
        except Exception:
            self.skill_runtime.cleanup(new_id)
            if child_workspaces:
                self.project_workspaces.cleanup(new_id, child_workspaces)
            else:
                self.workspaces.cleanup(
                    new_id,
                    str(prepared.source_path),
                    str(prepared.workspace_path),
                    prepared_kind,
                    prepared.base_revision,
                )
            try:
                self.store.delete_session(new_id)
            except FileNotFoundError:
                pass
            try:
                self.control.delete_task(control_snapshot.run.id)
            except KeyError:
                pass
            raise

    @staticmethod
    def _forked_instructions(
        parent: CodingSession,
        child_workspaces: list[TaskWorkspace],
    ) -> str:
        """Retarget the parent's frozen instructions without rereading live project state."""

        instructions = parent.developer_instructions
        child_by_id = {workspace.folder_id: workspace for workspace in child_workspaces}
        for workspace in parent.workspaces:
            child = child_by_id.get(workspace.folder_id)
            if child is not None:
                instructions = instructions.replace(workspace.workspace_path, child.workspace_path)
        return instructions
