from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from cowork.coding.approvals import ApprovalBroker
from cowork.coding.commands import CodingCommandHandler
from cowork.coding.context import (
    validate_directories,
)
from cowork.coding.contracts import (
    ApprovalDecision,
    CodingEvent,
    CodingSession,
    EngineCapabilities,
    EventPage,
    EventType,
    PendingApproval,
    SessionPage,
    SessionStatus,
    SessionUpdateRequest,
    TaskCapabilities,
    TaskCapability,
    WorkspaceInspection,
)
from cowork.coding.control_models import RunStatus
from cowork.coding.control_service import ControlPlaneService
from cowork.coding.control_store import ControlPlaneStore
from cowork.coding.delivery import ProjectDeliveryService
from cowork.coding.engines.base import EngineCredentials
from cowork.coding.engines.registry import CodingEngineRegistry, engine_registry
from cowork.coding.playbooks import PlaybookService
from cowork.coding.project_service import CodeProjectService
from cowork.coding.project_actions import ProjectActionService
from cowork.coding.project_models import (
    CodeProject,
    ProjectActionPage,
    ProjectActionRunRequest,
    ProjectActionRunResponse,
)
from cowork.coding.project_store import CodeProjectStore
from cowork.coding.project_tasks import ProjectTaskOperations
from cowork.coding.project_workspaces import ProjectWorkspaceManager
from cowork.coding.remote_execution import RemoteExecutionCoordinator
from cowork.coding.runtime import RuntimeManager
from cowork.coding.service_delivery import CodingDeliveryOperations
from cowork.coding.service_terminals import CodingTerminalOperations
from cowork.coding.service_turns import CodingTurnOperations
from cowork.coding.service_workspace_files import CodingWorkspaceFilesOperations
from cowork.coding.session_factory import CodingSessionFactory
from cowork.coding.session_lifecycle import SessionLifecycleOperations
from cowork.coding.shells import shell_inventory
from cowork.coding.skill_library import SkillLibraryService
from cowork.coding.skill_runtime import SkillRuntimeResolver
from cowork.coding.store import CodingStore
from cowork.coding.task_delivery import TaskDeliveryService
from cowork.coding.terminal_service import TaskTerminalService
from cowork.coding.turns import RunningTurn, TurnExecutor
from cowork.coding.workspace import WorkspaceManager
from cowork.common.paths import cowork_home

logger = logging.getLogger(__name__)


class CodingService(
    CodingWorkspaceFilesOperations,
    CodingDeliveryOperations,
    CodingTerminalOperations,
    CodingTurnOperations,
):
    def __init__(
        self,
        root: Path,
        registry: CodingEngineRegistry | None = None,
        store: CodingStore | None = None,
        workspaces: WorkspaceManager | None = None,
        control_store: ControlPlaneStore | None = None,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry = registry or engine_registry
        self.store = store or CodingStore(root)
        self.workspaces = workspaces or WorkspaceManager(root)
        shell_options = [item.id.value for item in shell_inventory().items]
        available_engines = [item.id for item in self.registry.capabilities() if item.available]
        self.control = ControlPlaneService(
            root,
            ControlPlaneService.default_capabilities(available_engines, shell_options),
            control_store,
        )
        self.project_store = CodeProjectStore(root, self.control.local_computer.id)
        self.skill_library = SkillLibraryService(root, self.project_store, self.workspaces.git)
        self.projects = CodeProjectService(
            root,
            self.project_store,
            self.workspaces,
            self.skill_library.validate_project,
            self.control.local_computer.id,
        )
        self.playbooks = PlaybookService(root, self.project_store, self.workspaces.git)
        self.skill_runtime = SkillRuntimeResolver(self.skill_library)
        self.project_workspaces = ProjectWorkspaceManager(self.workspaces)
        self.delivery = ProjectDeliveryService(self.workspaces.git)
        self._lock = threading.RLock()
        self._running: dict[str, RunningTurn] = {}
        self._maintenance: set[str] = set()
        self.approvals = ApprovalBroker(self._approval_opened, self._approval_closed)
        self.runtimes = RuntimeManager(root, self.registry, self.approvals.request)
        self.lifecycle = SessionLifecycleOperations(
            maintenance_session=self._maintenance_session,
            emit=self._emit,
            store=self.store,
            workspaces=self.workspaces,
            project_workspaces=self.project_workspaces,
            execution_project=self._execution_project,
            control=self.control,
            skill_runtime=self.skill_runtime,
            runtimes=self.runtimes,
            running=self._running,
            lock=self._lock,
        )
        self.project_tasks = ProjectTaskOperations(
            get_session=self.get_session,
            maintenance_session=self._maintenance_session,
            emit=self._emit,
            store=self.store,
            workspaces=self.workspaces,
            project_workspaces=self.project_workspaces,
            execution_project=self._execution_project,
            delivery=self.delivery,
        )
        self.session_factory = CodingSessionFactory(
            self.registry,
            self.store,
            self.workspaces,
            self.projects,
            self.playbooks,
            self.skill_runtime,
            self.project_workspaces,
            self._emit,
            self.control,
        )
        self.commands = CodingCommandHandler(
            self.registry,
            self.runtimes,
            self.get_session,
            self._emit,
            self._lock,
            lambda session_id: session_id in self._running,
        )
        self.remote = RemoteExecutionCoordinator(
            self.control,
            self.store,
            self.projects,
            self.get_session,
        )
        self.task_delivery = TaskDeliveryService(
            self.store,
            self.control,
            self._execution_project,
            self.project_tasks,
            self.remote,
            self.get_session,
        )
        self.task_terminals = TaskTerminalService(
            self.store,
            self.runtimes,
            self.remote,
            self.get_session,
        )
        self.project_actions = ProjectActionService(
            get_session=self.get_session,
            projects=self.projects,
            terminals=self.task_terminals,
            get_computer=self.control.store.get_computer,
            local_computer_id=self.control.local_computer.id,
            execution_project=self._execution_project,
        )
        self.turns = TurnExecutor(
            self.runtimes,
            self.store,
            self._running,
            self._lock,
            self.get_session,
            self._emit,
            self.run_next_queued,
        )
        self.store.reconcile_interrupted()
        for session in self.store.list_sessions():
            try:
                project = self.projects.get(session.project_id) if session.project_id else None
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "Coding task %s references an unavailable project %s: %s",
                    session.id,
                    session.project_id,
                    exc,
                )
                project = None
            snapshot = self.control.migrate_session(session, project)
            if not session.task_id or not session.run_id or not session.computer_id:
                session.task_id = snapshot.task.id
                session.run_id = snapshot.run.id
                session.computer_id = snapshot.computer.id
                session.runtime_epoch = snapshot.run.epoch
                session.resource_ids = [item.resource_id for item in snapshot.workspaces]
                self.store.save_session(session, touch_updated_at=False)
            self.project_workspaces.restore_ports(session.id, session.allocated_ports)

    def capabilities(self) -> list[EngineCapabilities]:
        return self.registry.capabilities()

    def inspect_workspace(self, path: str) -> WorkspaceInspection:
        return self.workspaces.inspect(path)

    def list_sessions(self, include_archived: bool = False) -> SessionPage:
        sessions = self.store.list_sessions()
        visible = sessions if include_archived else [item for item in sessions if not item.archived]
        return SessionPage(items=[self._control_view(item) for item in visible])

    def get_session(self, session_id: str) -> CodingSession:
        try:
            return self._control_view(self.store.load_session(session_id))
        except (FileNotFoundError, ValueError) as exc:
            raise KeyError("coding session not found") from exc

    def _control_view(self, session: CodingSession) -> CodingSession:
        """Project canonical Task Run state onto the compatibility session."""

        if not session.run_id:
            return session
        self.control.expire_stale_computers()
        self.control.expire_leases()
        try:
            run = self.control.store.get_run(session.run_id)
            computer = self.control.store.get_computer(run.computer_id)
        except KeyError:
            return session
        is_local = computer.id == self.control.local_computer.id
        available = set(computer.capabilities.task_capabilities)
        return session.model_copy(update={
            "computer_id": computer.id,
            "run_status": run.status.value,
            "computer_name": computer.name,
            "computer_status": computer.status.value,
            "computer_is_local": is_local,
            "task_capabilities": TaskCapabilities(**{
                capability.value: is_local or capability in available
                for capability in TaskCapability
            }),
            "runtime_epoch": run.epoch,
            "last_error": run.last_error or session.last_error,
        })

    def delete_project(self, project_id: str) -> None:
        task_count = sum(1 for item in self.store.list_sessions() if item.project_id == project_id)
        self.projects.delete(project_id, task_count)
        self.playbooks.cleanup(project_id)

    def delete_session(self, session_id: str) -> None:
        session = self.get_session(session_id)
        if self._is_remote(session):
            self.remote.require_idle(session, "deleting this coding task")
            self.remote.release_workspace(session)
            self.store.delete_session(session.id)
            self.skill_runtime.cleanup(session.id)
            if session.run_id:
                self.control.delete_task(session.run_id)
            return
        self.lifecycle.delete_session(session_id)
        if session.run_id:
            self.control.delete_task(session.run_id)

    def rename_session(self, session_id: str, title: str) -> CodingSession:
        return self.lifecycle.rename_session(session_id, title)

    def set_archived(self, session_id: str, archived: bool) -> CodingSession:
        session = self.get_session(session_id)
        if archived and self._is_remote(session):
            self.remote.release_workspace(session)
        return self.lifecycle.set_archived(session_id, archived)

    def set_pinned(self, session_id: str, pinned: bool) -> CodingSession:
        return self.lifecycle.set_pinned(session_id, pinned)

    def fork_session(self, session_id: str, credentials: EngineCredentials) -> CodingSession:
        session = self.get_session(session_id)
        self._require_task_capability(session, TaskCapability.fork)
        return self.lifecycle.fork_session(session_id, credentials)

    def events(self, session_id: str, after: int = 0) -> EventPage:
        self.get_session(session_id)
        items = self.store.events_after(session_id, max(0, after))
        return EventPage(items=items, next_seq=items[-1].seq if items else max(0, after))

    def wait_for_events(self, session_id: str, after: int, timeout: float = 15.0) -> EventPage:
        self.get_session(session_id)
        items = self.store.wait_for_events(session_id, max(0, after), timeout)
        return EventPage(items=items, next_seq=items[-1].seq if items else max(0, after))

    def recovery_plan(self, session_id: str):
        return self.remote.recovery_plan(session_id)

    def recover(
        self,
        session_id: str,
        computer_id: str | None = None,
        *,
        allow_recreate: bool = False,
    ) -> CodingSession:
        return self.remote.recover(
            session_id,
            computer_id,
            allow_recreate=allow_recreate,
        )

    def resolve_approval(self, session_id: str, approval_id: str, decision: ApprovalDecision) -> CodingSession:
        session = self.get_session(session_id)
        if self._is_remote(session):
            if not session.run_id or not session.pending_approval or session.pending_approval.id != approval_id:
                raise KeyError("approval is no longer pending")
            self.control.queue_command(
                session.run_id,
                "approve",
                {"approvalId": approval_id, "decision": decision.value},
                f"approval-{approval_id}",
            )
            return session
        self.approvals.resolve(session_id, approval_id, decision)
        return self.get_session(session_id)

    def git_state(self, session_id: str):
        session = self.get_session(session_id)
        if self._is_remote(session):
            return self.remote.operation(session, "git_state")
        return self.project_tasks.git_state(session_id)

    def git_states(self, session_id: str):
        session = self.get_session(session_id)
        if self._is_remote(session):
            return list(self.remote.operation(session, "git_states").get("items") or [])
        return self.project_tasks.git_states(session_id)

    def diff(self, session_id: str):
        session = self.get_session(session_id)
        if self._is_remote(session):
            return list(self.remote.operation(session, "diff").get("files") or [])
        return self.project_tasks.diff(session_id)

    def review_file_action(self, session_id: str, folder_id: str | None, path: str, action: str):
        session = self.get_session(session_id)
        if self._is_remote(session):
            self.remote.require_idle(session, "changing review state")
            return list(self.remote.operation(session, "review_file", {
                "folder_id": folder_id,
                "path": path,
                "action": action,
            }).get("files") or [])
        return self.project_tasks.review_file_action(session_id, folder_id, path, action)

    def create_branch(self, session_id: str, name: str):
        session = self.get_session(session_id)
        if self._is_remote(session):
            self.remote.require_idle(session, "changing task branches")
            return self.remote.operation(session, "branch", {"name": name})
        return self.project_tasks.create_branch(session_id, name)

    def commit(self, session_id: str, message: str):
        session = self.get_session(session_id)
        if self._is_remote(session):
            self.remote.require_idle(session, "committing task changes")
            return self.remote.operation(session, "commit", {"message": message})
        return self.project_tasks.commit(session_id, message)

    def apply_to_source(self, session_id: str) -> dict[str, str | None]:
        session = self.get_session(session_id)
        if self._is_remote(session):
            raise RuntimeError("Publish remote task changes with a branch or pull request")
        return self.project_tasks.apply_to_source(session_id)

    def validate_project(self, session_id: str) -> list[dict]:
        session = self.get_session(session_id)
        if self._is_remote(session):
            self.remote.require_idle(session, "running project validation")
            return list(self.remote.operation(session, "validate", timeout=610).get("items") or [])
        return self.project_tasks.validate_project(session_id)

    def discover_models(self, engine_id: str, credentials: EngineCredentials) -> list[str]:
        return self.registry.get(engine_id).discover_models(credentials)

    def extension_inventory(self, session_id: str, credentials: EngineCredentials):
        session = self.get_session(session_id)
        self._require_task_capability(session, TaskCapability.extensions)
        runtime = self.runtimes.open(session, credentials)
        return runtime.extension_inventory()

    def platform_status(self, session_id: str, credentials: EngineCredentials):
        session = self.get_session(session_id)
        self._require_task_capability(session, TaskCapability.platform_settings)
        return self.runtimes.open(session, credentials).platform_status()

    def setup_windows_sandbox(self, session_id: str, credentials: EngineCredentials):
        session = self.get_session(session_id)
        self._require_task_capability(session, TaskCapability.platform_settings)
        return self.runtimes.open(session, credentials).setup_windows_sandbox()

    def update_session_config(self, session_id: str, updates: SessionUpdateRequest) -> CodingSession:
        self._require_task_capability(self.get_session(session_id), TaskCapability.task_controls)
        with self.runtimes.session_lock(session_id):  # noqa: SIM117 - preserve lock ordering.
            with self._maintenance_session(
                session_id,
                "Wait for the active turn to finish before changing task controls",
            ) as session:
                if self.runtimes.terminal_is_running(session_id):
                    raise RuntimeError("Stop the task terminal before changing task controls")
                values = updates.model_dump(exclude_none=True)
                if "additional_dirs" in values:
                    values["additional_dirs"] = validate_directories(values["additional_dirs"])
                next_permission = values.get("permission_mode", session.permission_mode)
                if getattr(next_permission, "value", next_permission) == "full_access":
                    values["network_access"] = True
                self.runtimes.close_locked(session_id)
                return self.store.update_session(
                    session_id,
                    lambda current: self._apply_config_update(current, values),
                )

    def run_project_action(
        self,
        session_id: str,
        request: ProjectActionRunRequest,
        credentials: EngineCredentials,
    ) -> ProjectActionRunResponse:
        return self.project_actions.run(session_id, request, credentials)

    def project_action_page(self, session_id: str) -> ProjectActionPage:
        return self.project_actions.list(session_id)

    def _execution_project(self, session: CodingSession) -> CodeProject | None:
        if not session.task_id:
            return None
        try:
            return self.control.store.get_task(session.task_id).execution_project
        except KeyError:
            return None

    @staticmethod
    def _require_task_capability(session: CodingSession, capability: TaskCapability) -> None:
        if not getattr(session.task_capabilities, capability.value):
            label = capability.value.replace("_", " ")
            computer = session.computer_name or "this computer"
            raise RuntimeError(f"{label.capitalize()} isn't available for tasks running on {computer}")

    def close_all(self) -> None:
        """Stop every task-owned engine runtime during application shutdown."""
        self.prepare_shutdown()
        self.runtimes.close_all()

    def prepare_shutdown(self) -> int:
        """Checkpoint active turns before the desktop terminates the sidecar tree."""
        with self._lock:
            active_sessions = list(self._running)
        for session_id in active_sessions:
            try:
                self.approvals.cancel_session(session_id)
            except Exception:
                # Shutdown must continue releasing the remaining tasks and
                # runtimes even if one persisted approval cannot be updated.
                logger.exception("Could not cancel approval while shutting down task %s", session_id)
        return self.turns.interrupt(active_sessions)

    def _approval_opened(self, session_id: str, pending: PendingApproval) -> None:
        self._emit(
            session_id,
            CodingEvent(
                type=EventType.approval,
                title=pending.title,
                text=pending.detail,
                phase="pending",
                data={
                    "approvalId": pending.id,
                    "kind": pending.kind,
                    "cwd": pending.cwd,
                    "risk": pending.risk,
                    "scope": pending.scope,
                    "allowSession": pending.allow_session,
                },
            ),
            lambda current: self._open_approval(current, pending),
        )

    def _approval_closed(self, session_id: str, pending: PendingApproval, decision: ApprovalDecision) -> None:
        self._emit(
            session_id,
            CodingEvent(
                type=EventType.approval,
                title="Approval resolved",
                text=decision.value.replace("_", " ").capitalize(),
                phase="completed",
                data={"approvalId": pending.id, "decision": decision.value},
            ),
            self._close_approval,
        )

    def _emit(
        self,
        session_id: str,
        event: CodingEvent,
        update: Callable[[CodingSession], None] | None = None,
    ) -> CodingEvent:
        stored = self.store.append_event(session_id, event, update)
        try:
            self.control.sync_session(self.store.load_session(session_id))
        except (KeyError, ValueError):
            logger.exception("Could not synchronize Task Run state for coding task %s", session_id)
        return stored

    @staticmethod
    def _open_approval(session: CodingSession, pending: PendingApproval) -> None:
        session.status = SessionStatus.awaiting_approval
        session.pending_approval = pending

    @staticmethod
    def _close_approval(session: CodingSession) -> None:
        session.pending_approval = None
        if session.status == SessionStatus.awaiting_approval:
            session.status = SessionStatus.running

    @staticmethod
    def _apply_config_update(session: CodingSession, values: dict) -> None:
        for name, value in values.items():
            setattr(session, name, value)

    @staticmethod
    def _remove_queued_instruction(session: CodingSession, instruction_id: str) -> None:
        session.queued_instructions = [
            item for item in session.queued_instructions if item.id != instruction_id
        ]

    @staticmethod
    def _restore_queued_instruction(
        session: CodingSession,
        instruction: QueuedInstruction,
        index: int,
    ) -> None:
        if any(item.id == instruction.id for item in session.queued_instructions):
            return
        session.queued_instructions.insert(min(index, len(session.queued_instructions)), instruction)

    @contextmanager
    def _maintenance_session(
        self,
        session_id: str,
        active_turn_error: str,
    ) -> Iterator[CodingSession]:
        """Reserve an idle task for a potentially slow lifecycle operation."""
        with self._lock:
            session = self.get_session(session_id)
            if session_id in self._running:
                raise RuntimeError(active_turn_error)
            if session_id in self._maintenance:
                raise RuntimeError("This coding task is already being updated")
            self._maintenance.add(session_id)
        try:
            yield session
        finally:
            with self._lock:
                self._maintenance.discard(session_id)

@lru_cache(maxsize=1)
def get_coding_service() -> CodingService:
    return CodingService(cowork_home() / "coding")
