from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from cowork.coding.approvals import ApprovalBroker
from cowork.coding.commands import CodingCommandHandler, CommandIntent
from cowork.coding.connector_capabilities import ConnectorCapability
from cowork.coding.context import (
    validate_directories,
    validate_references,
    workspace_files,
)
from cowork.coding.contracts import (
    ApprovalDecision,
    CodingEvent,
    CodingSession,
    EngineCapabilities,
    EventPage,
    EventType,
    InputReference,
    PendingApproval,
    QueuedInstruction,
    SessionCreateRequest,
    SessionPage,
    SessionStatus,
    SessionUpdateRequest,
    TerminalPage,
    TerminalShellPreference,
    TerminalTabPage,
    TerminalTabState,
    WorkspaceInspection,
)
from cowork.coding.control_models import RunStatus, RuntimeEvent, TaskRun
from cowork.coding.control_service import ControlPlaneService
from cowork.coding.control_store import ControlPlaneStore
from cowork.coding.delivery import ProjectDeliveryService
from cowork.coding.engines.base import EngineCredentials, EngineSession
from cowork.coding.engines.registry import CodingEngineRegistry, engine_registry
from cowork.coding.playbooks import PlaybookService
from cowork.coding.project_service import CodeProjectService
from cowork.coding.project_store import CodeProjectStore
from cowork.coding.project_tasks import ProjectTaskOperations
from cowork.coding.project_workspaces import ProjectWorkspaceManager
from cowork.coding.remote_execution import RemoteExecutionCoordinator
from cowork.coding.runtime import RuntimeManager
from cowork.coding.service_delivery import CodingDeliveryOperations
from cowork.coding.session_factory import CodingSessionFactory
from cowork.coding.session_lifecycle import SessionLifecycleOperations
from cowork.coding.shells import shell_inventory
from cowork.coding.skill_library import SkillLibraryService
from cowork.coding.skill_runtime import SkillRuntimeResolver
from cowork.coding.store import CodingStore
from cowork.coding.task_delivery import TaskDeliveryService
from cowork.coding.terminal_service import TaskTerminalService
from cowork.coding.turns import RunningTurn, TurnExecutor, fail_turn, mark_running
from cowork.coding.workspace import WorkspaceManager
from cowork.common.paths import cowork_home
from cowork.services.skills import CodeSkillService

logger = logging.getLogger(__name__)


class CodingService(CodingDeliveryOperations):
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
            projects=self.projects,
            playbooks=self.playbooks,
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
            projects=self.projects,
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
            self.projects,
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
            except KeyError:
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
        return session.model_copy(update={
            "computer_id": computer.id,
            "run_status": run.status.value,
            "computer_name": computer.name,
            "computer_status": computer.status.value,
            "computer_is_local": computer.id == self.control.local_computer.id,
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
        return self.lifecycle.fork_session(session_id, credentials)

    def events(self, session_id: str, after: int = 0) -> EventPage:
        self.get_session(session_id)
        items = self.store.events_after(session_id, max(0, after))
        return EventPage(items=items, next_seq=items[-1].seq if items else max(0, after))

    def wait_for_events(self, session_id: str, after: int, timeout: float = 15.0) -> EventPage:
        self.get_session(session_id)
        items = self.store.wait_for_events(session_id, max(0, after), timeout)
        return EventPage(items=items, next_seq=items[-1].seq if items else max(0, after))

    def create_session(
        self,
        request: SessionCreateRequest,
        credentials: EngineCredentials,
        default_engine: str,
        default_model: str,
        code_skills: CodeSkillService | None = None,
    ) -> CodingSession:
        session = self.session_factory.create(
            request,
            credentials,
            default_engine,
            default_model,
            code_skills,
        )
        if self._is_remote(session):
            self.remote.queue_turn(session, request.prompt, request.attachments)
            return self.get_session(session.id)
        try:
            self.submit_turn(session.id, request.prompt, credentials, request.attachments)
        except Exception:
            self.session_factory.discard(session)
            raise
        return self.get_session(session.id)

    def submit_turn(
        self,
        session_id: str,
        prompt: str,
        credentials: EngineCredentials,
        attachments: list[InputReference] | tuple[InputReference, ...] = (),
    ) -> CodingSession:
        session = self._continue_completed_task(self.get_session(session_id))
        if self._is_remote(session):
            return self.remote.queue_turn(session, prompt, attachments)
        return self._submit_turn(session_id, prompt, credentials, attachments)

    def _continue_completed_task(self, session: CodingSession) -> CodingSession:
        if not session.run_id:
            return session
        run = self.control.store.get_run(session.run_id)
        if run.status not in {RunStatus.completed, RunStatus.cancelled, RunStatus.failed}:
            return session
        continued = self.control.continue_task(session.task_id or session.id, run.id)
        session.run_id = continued.id
        session.runtime_epoch = continued.epoch
        session.status = SessionStatus.ready
        session.last_error = None
        self.store.save_session(session)
        return self.get_session(session.id)

    def accept_runtime_event(self, event: RuntimeEvent) -> TaskRun:
        return self.remote.accept_event(event)

    def _is_remote(self, session: CodingSession) -> bool:
        return self.remote.is_remote(session)

    def runtime_connector_capabilities(self, session: CodingSession) -> list[ConnectorCapability]:
        return self.remote.connector_capabilities(session)

    def _submit_turn(
        self,
        session_id: str,
        prompt: str,
        credentials: EngineCredentials,
        attachments: list[InputReference] | tuple[InputReference, ...],
        *,
        maintenance_reserved: bool = False,
    ) -> CodingSession:
        """Validate and launch one turn, optionally inside a service reservation."""
        intent = CommandIntent.parse(prompt)
        self.commands.validate(self.get_session(session_id), intent.name)
        intent.validate_arguments()
        intent.validate_attachments(bool(attachments))
        if intent.runs_immediately:
            if maintenance_reserved:
                return self.commands.run_immediate(session_id, intent, prompt, credentials)
            with self._maintenance_session(
                session_id,
                "Wait for the active turn to finish before running this command",
            ):
                return self.commands.run_immediate(session_id, intent, prompt, credentials)
        with self._lock:
            session = self.get_session(session_id)
            if session_id in self._maintenance and not maintenance_reserved:
                raise RuntimeError("This coding task is being updated. Try again in a moment")
            if session_id in self._running or session.status in {SessionStatus.running, SessionStatus.awaiting_approval}:
                raise RuntimeError("This coding task already has a running turn")
            engine_attachments = validate_references(session, attachments)
            self._emit(
                session_id,
                CodingEvent(type=EventType.user_message, title="You", text=prompt, phase="completed"),
                mark_running,
            )
            thread = threading.Thread(
                target=self.turns.execute,
                args=(
                    session_id,
                    intent.engine_prompt(prompt),
                    credentials,
                    intent.goal_objective,
                    engine_attachments,
                    intent.is_review,
                    intent.resumes_goal,
                ),
                name=f"coding-turn-{session_id[:8]}",
                daemon=True,
            )
            # Reserve the slot before starting so two simultaneous HTTP calls
            # cannot both launch a turn. The engine is filled by the worker.
            self._running[session_id] = RunningTurn(engine=None, turn_id="", thread=thread)
            response = self.get_session(session_id)
            try:
                thread.start()
            except Exception as exc:
                self._running.pop(session_id, None)
                message = f"The coding worker could not start: {exc}"
                self._emit(
                    session_id,
                    CodingEvent(type=EventType.error, title="Coding agent failed", text=message, phase="failed"),
                    lambda current: fail_turn(current, False, message),
                )
                raise RuntimeError(message) from exc
        return response

    def steer(
        self,
        session_id: str,
        prompt: str,
        attachments: list[InputReference] | tuple[InputReference, ...] = (),
    ) -> CodingSession:
        session = self.get_session(session_id)
        if self._is_remote(session):
            if not session.run_id:
                raise RuntimeError("Remote task is missing its Task Run")
            if attachments:
                raise RuntimeError("File attachments cannot steer a task on another computer yet")
            run = self.control.store.get_run(session.run_id)
            if run.status not in {RunStatus.running, RunStatus.awaiting_approval}:
                raise RuntimeError("There is no active turn to steer")
            command = self.control.queue_command(
                session.run_id,
                "steer",
                {"prompt": prompt},
                f"steer-{uuid.uuid4().hex}",
            )
            self.store.append_event(
                session.id,
                CodingEvent(
                    type=EventType.user_message,
                    title="Steering now",
                    text=prompt,
                    phase="pending",
                    data={"commandId": command.id},
                ),
            )
            return self.get_session(session.id)
        intent = CommandIntent.parse(prompt)
        intent.validate_arguments()
        intent.validate_attachments(bool(attachments))
        intent.validate_while_running()
        engine: EngineSession | None = None
        turn_id = ""
        with self._lock:
            session = self.get_session(session_id)
            self.commands.validate(session, intent.name)
            engine_attachments = validate_references(session, attachments)
            running = self._running.get(session_id)
            if running is None:
                raise RuntimeError("There is no active turn to steer")
            engine, turn_id = running.route_steer(
                prompt,
                engine_attachments,
                require_ready=intent.runs_immediately,
            )
        if intent.runs_immediately:
            if engine is None:
                raise RuntimeError("Coding agent state is inconsistent")
            self._emit(session.id, CodingEvent(type=EventType.user_message, title="You", text=prompt, phase="completed"))
            if intent.name == "status":
                self.commands.emit_status(session, engine)
            else:
                self.commands.emit_goal_result(
                    session.id,
                    engine,
                    intent.goal_action,
                    intent.goal_objective,
                )
            return self.get_session(session_id)
        if engine is not None:
            engine.steer(turn_id, prompt, engine_attachments)
        self._emit(session.id, CodingEvent(type=EventType.user_message, title="Follow-up", text=prompt, phase="completed"))
        return self.get_session(session_id)

    def queue_turn(
        self,
        session_id: str,
        prompt: str,
        attachments: list[InputReference] | tuple[InputReference, ...] = (),
    ) -> CodingSession:
        intent = CommandIntent.parse(prompt)
        intent.validate_arguments()
        intent.validate_attachments(bool(attachments))
        with self._lock:
            session = self.get_session(session_id)
            self.commands.validate(session, intent.name)
            validate_references(session, attachments)
            instruction = QueuedInstruction(
                id=str(uuid.uuid4()),
                prompt=prompt,
                attachments=list(attachments),
            )
            if self._is_remote(session):
                if attachments:
                    raise RuntimeError("File attachments cannot be queued to another computer yet")
                if not session.run_id or self.control.store.get_run(session.run_id).status not in {
                    RunStatus.running,
                    RunStatus.awaiting_approval,
                }:
                    raise RuntimeError("There is no active turn. Send this as a new turn instead")
            elif session_id not in self._running:
                raise RuntimeError("There is no active turn. Send this as a new turn instead")
            if len(session.queued_instructions) >= 20:
                raise RuntimeError("This coding task already has 20 queued instructions")
            self._emit(
                session_id,
                CodingEvent(
                    type=EventType.user_message,
                    title="Queued next",
                    text=prompt,
                    phase="pending",
                    data={"queueId": instruction.id},
                ),
                lambda current: current.queued_instructions.append(instruction),
            )
        return self.get_session(session_id)

    def remove_queued_turn(self, session_id: str, instruction_id: str) -> CodingSession:
        with self._lock:
            session = self.get_session(session_id)
            if not any(item.id == instruction_id for item in session.queued_instructions):
                raise KeyError("queued instruction not found")
            self._emit(
                session_id,
                CodingEvent(
                    type=EventType.session,
                    title="Queued instruction removed",
                    text="The instruction will not run after the active turn.",
                    phase="completed",
                    data={"queueId": instruction_id},
                ),
                lambda current: self._remove_queued_instruction(current, instruction_id),
            )
        return self.get_session(session_id)

    def steer_queued_turn(self, session_id: str, instruction_id: str) -> CodingSession:
        """Promote one persisted queued instruction into the active turn.

        The instruction is removed before crossing the engine boundary so a
        turn completing concurrently cannot also start it as the next turn. If
        the engine rejects the steer, the queue entry is restored in place for
        an explicit retry or normal queued execution.
        """
        queue_index = 0
        with self._lock:
            session = self.get_session(session_id)
            try:
                queue_index, instruction = next(
                    (index, item)
                    for index, item in enumerate(session.queued_instructions)
                    if item.id == instruction_id
                )
            except StopIteration as exc:
                raise KeyError("queued instruction not found") from exc
            remote = self._is_remote(session)
            if not remote and session_id not in self._running:
                raise RuntimeError("There is no active turn to steer")
            self.store.update_session(
                session_id,
                lambda current: self._remove_queued_instruction(current, instruction_id),
            )
        try:
            return self.steer(session_id, instruction.prompt, instruction.attachments)
        except Exception:
            with self._lock:
                self.store.update_session(
                    session_id,
                    lambda current: self._restore_queued_instruction(current, instruction, queue_index),
                )
            raise

    def run_next_queued(self, session_id: str, credentials: EngineCredentials) -> CodingSession:
        """Start the oldest queued instruction once the task is idle.

        The queue is persisted with the task. This method is also exposed to the
        renderer so an interrupted desktop restart can resume a pending queue.
        """
        # Completing an ordinary turn should not briefly reserve the task just
        # to discover that its queue is empty. The fast path keeps a freshly
        # completed task immediately available for the user's next message.
        session = self.get_session(session_id)
        if not session.queued_instructions:
            return session
        if self._is_remote(session):
            return self.remote.start_next_queued(session)
        with self._maintenance_session(
            session_id,
            "Wait for the active turn to finish before resuming queued work",
        ):
            while True:
                session = self.get_session(session_id)
                if not session.queued_instructions:
                    return session
                instruction = session.queued_instructions[0]
                instruction_id = instruction.id
                self.store.update_session(
                    session_id,
                    lambda current, target_id=instruction_id: self._remove_queued_instruction(current, target_id),
                )
                try:
                    result = self._submit_turn(
                        session_id,
                        instruction.prompt,
                        credentials,
                        instruction.attachments,
                        maintenance_reserved=True,
                    )
                except Exception:
                    queued_instruction = instruction
                    self.store.update_session(
                        session_id,
                        lambda current, queued=queued_instruction: current.queued_instructions.insert(0, queued),
                    )
                    raise
                if result.status in {SessionStatus.running, SessionStatus.awaiting_approval}:
                    return result

    def cancel(self, session_id: str) -> CodingSession:
        session = self.get_session(session_id)
        if self._is_remote(session):
            if not session.run_id:
                raise RuntimeError("Remote task is missing its Task Run")
            run = self.control.store.get_run(session.run_id)
            if run.status not in {RunStatus.running, RunStatus.awaiting_approval}:
                raise RuntimeError("There is no active turn to stop")
            self.control.queue_command(
                session.run_id,
                "cancel",
                {},
                # One cancellation per canonical remote turn. Repeated clicks
                # before the runtime advances are deduplicated, while a later
                # turn in the same epoch receives a new sequence-backed key.
                f"cancel-{run.epoch}-{run.last_event_seq}",
            )
            return session
        engine: EngineSession | None = None
        turn_id = ""
        with self._lock:
            running = self._running.get(session_id)
            if running is None:
                raise RuntimeError("There is no active turn to stop")
            running.cancel_requested = True
            if running.turn_id and running.engine is not None:
                engine = running.engine
                turn_id = running.turn_id
        self._emit(
            session_id,
            CodingEvent(
                type=EventType.session,
                title="Stopping task",
                text="Cancellation requested. The agent is cleaning up the active turn.",
                phase="pending",
            ),
        )
        try:
            self.approvals.cancel_session(session_id)
        finally:
            if engine is not None:
                engine.cancel(turn_id)
        return self.get_session(session_id)

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
        runtime = self.runtimes.open(session, credentials)
        return runtime.extension_inventory()

    def platform_status(self, session_id: str, credentials: EngineCredentials):
        session = self.get_session(session_id)
        return self.runtimes.open(session, credentials).platform_status()

    def setup_windows_sandbox(self, session_id: str, credentials: EngineCredentials):
        session = self.get_session(session_id)
        return self.runtimes.open(session, credentials).setup_windows_sandbox()

    def workspace_files(self, session_id: str, query: str = "", limit: int = 40) -> list[dict[str, str]]:
        session = self.get_session(session_id)
        return workspace_files(session, query, limit)

    def update_session_config(self, session_id: str, updates: SessionUpdateRequest) -> CodingSession:
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

    def terminals(self, session_id: str) -> TerminalTabPage:
        return self.task_terminals.list(session_id)

    def create_terminal_tab(self, session_id: str, label: str | None = None) -> TerminalTabState:
        return self.task_terminals.create(session_id, label)

    def rename_terminal_tab(self, session_id: str, terminal_id: str, label: str) -> TerminalTabState:
        return self.task_terminals.rename(session_id, terminal_id, label)

    def delete_terminal_tab(self, session_id: str, terminal_id: str) -> None:
        self.task_terminals.delete(session_id, terminal_id)

    def terminal_tab(self, session_id: str, terminal_id: str, after: int = 0) -> TerminalPage:
        return self.task_terminals.page(session_id, terminal_id, after)

    def wait_for_terminal_tab(
        self,
        session_id: str,
        terminal_id: str,
        after: int,
        timeout: float = 15.0,
    ) -> TerminalPage:
        return self.task_terminals.wait(session_id, terminal_id, after, timeout)

    def start_terminal_tab(
        self,
        session_id: str,
        terminal_id: str,
        credentials: EngineCredentials,
        cols: int,
        rows: int,
        shell: TerminalShellPreference = TerminalShellPreference.auto,
    ) -> TerminalPage:
        return self.task_terminals.start(session_id, terminal_id, credentials, cols, rows, shell)

    def write_terminal_tab(self, session_id: str, terminal_id: str, data_base64: str) -> TerminalPage:
        return self.task_terminals.write(session_id, terminal_id, data_base64)

    def resize_terminal_tab(self, session_id: str, terminal_id: str, cols: int, rows: int) -> TerminalPage:
        return self.task_terminals.resize(session_id, terminal_id, cols, rows)

    def stop_terminal_tab(self, session_id: str, terminal_id: str) -> TerminalPage:
        return self.task_terminals.stop(session_id, terminal_id)

    # Compatibility seam for the original one-terminal desktop. Deployments
    # can roll the server and renderer independently without breaking a task
    # that still speaks the singular endpoint contract.
    def terminal(self, session_id: str, after: int = 0) -> TerminalPage:
        return self.task_terminals.legacy_page(session_id, after)

    def wait_for_terminal(self, session_id: str, after: int, timeout: float = 15.0) -> TerminalPage:
        return self.task_terminals.legacy_wait(session_id, after, timeout)

    def start_terminal(
        self,
        session_id: str,
        credentials: EngineCredentials,
        cols: int,
        rows: int,
        shell: TerminalShellPreference = TerminalShellPreference.auto,
    ) -> TerminalPage:
        return self.task_terminals.legacy_start(session_id, credentials, cols, rows, shell)

    def write_terminal(self, session_id: str, data_base64: str) -> TerminalPage:
        return self.task_terminals.legacy_write(session_id, data_base64)

    def resize_terminal(self, session_id: str, cols: int, rows: int) -> TerminalPage:
        return self.task_terminals.legacy_resize(session_id, cols, rows)

    def stop_terminal(self, session_id: str) -> TerminalPage:
        return self.task_terminals.legacy_stop(session_id)

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
