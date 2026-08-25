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
from cowork.coding.context import (
    safe_engine_error,
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
    TaskWorkspace,
    TerminalPage,
    WorkspaceInspection,
    WorkspaceKind,
    utc_now,
)
from cowork.coding.delivery import ProjectDeliveryService
from cowork.coding.engines.base import EngineCredentials, EngineSession
from cowork.coding.engines.registry import CodingEngineRegistry, engine_registry
from cowork.coding.integrations import DeveloperIntegrationService
from cowork.coding.playbooks import PlaybookService
from cowork.coding.project_models import (
    DraftPullRequestRequest,
    PullRequestActionRequest,
    PullRequestStatus,
)
from cowork.coding.project_service import CodeProjectService
from cowork.coding.project_store import CodeProjectStore
from cowork.coding.project_workspaces import ProjectWorkspaceManager
from cowork.coding.runtime import RuntimeManager, engine_workspace_path
from cowork.coding.session_factory import CodingSessionFactory, project_instructions
from cowork.coding.skill_library import SkillLibraryService
from cowork.coding.skill_runtime import SkillRuntimeResolver
from cowork.coding.store import CodingStore
from cowork.coding.turns import RunningTurn, TurnExecutor, fail_turn, mark_running
from cowork.coding.workspace import WorkspaceError, WorkspaceManager
from cowork.common.paths import cowork_home
from cowork.services.skills import CodeSkillService

logger = logging.getLogger(__name__)


class CodingService:
    def __init__(
        self,
        root: Path,
        registry: CodingEngineRegistry | None = None,
        store: CodingStore | None = None,
        workspaces: WorkspaceManager | None = None,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry = registry or engine_registry
        self.store = store or CodingStore(root)
        self.workspaces = workspaces or WorkspaceManager(root)
        self.project_store = CodeProjectStore(root)
        self.projects = CodeProjectService(root, self.project_store, self.workspaces)
        self.playbooks = PlaybookService(root, self.project_store, self.workspaces.git)
        self.skill_library = SkillLibraryService(root, self.project_store, self.workspaces.git)
        self.skill_runtime = SkillRuntimeResolver(self.skill_library)
        self.project_workspaces = ProjectWorkspaceManager(self.workspaces)
        self.delivery = ProjectDeliveryService(self.workspaces.git)
        self._lock = threading.RLock()
        self._running: dict[str, RunningTurn] = {}
        self._maintenance: set[str] = set()
        self.approvals = ApprovalBroker(self._approval_opened, self._approval_closed)
        self.runtimes = RuntimeManager(root, self.registry, self.approvals.request)
        self.session_factory = CodingSessionFactory(
            self.registry,
            self.store,
            self.workspaces,
            self.projects,
            self.playbooks,
            self.skill_runtime,
            self.project_workspaces,
            self._emit,
        )
        self.commands = CodingCommandHandler(
            self.registry,
            self.runtimes,
            self.get_session,
            self._emit,
            self._lock,
            lambda session_id: session_id in self._running,
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
            self.project_workspaces.restore_ports(session.id, session.allocated_ports)

    def capabilities(self) -> list[EngineCapabilities]:
        return self.registry.capabilities()

    def inspect_workspace(self, path: str) -> WorkspaceInspection:
        return self.workspaces.inspect(path)

    def list_sessions(self, include_archived: bool = False) -> SessionPage:
        sessions = self.store.list_sessions()
        return SessionPage(items=sessions if include_archived else [item for item in sessions if not item.archived])

    def get_session(self, session_id: str) -> CodingSession:
        try:
            return self.store.load_session(session_id)
        except (FileNotFoundError, ValueError) as exc:
            raise KeyError("coding session not found") from exc

    def delete_project(self, project_id: str) -> None:
        task_count = sum(1 for item in self.store.list_sessions() if item.project_id == project_id)
        self.projects.delete(project_id, task_count)
        self.playbooks.cleanup(project_id)

    def delete_session(self, session_id: str) -> None:
        runtime_lock = self.runtimes.session_lock(session_id)
        with runtime_lock, self._maintenance_session(
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
        return self.store.update_session(session_id, lambda current: setattr(current, "title", normalized[:200]))

    def set_archived(self, session_id: str, archived: bool) -> CodingSession:
        with self._lock:
            if session_id in self._running:
                raise RuntimeError("Stop the active turn before archiving this coding task")
            return self.store.update_session(session_id, lambda current: setattr(current, "archived", archived))

    def fork_session(self, session_id: str, credentials: EngineCredentials) -> CodingSession:
        parent_lock = self.runtimes.session_lock(session_id)
        with parent_lock:  # noqa: SIM117 - the maintenance reservation must be acquired second.
            with self._maintenance_session(
                session_id,
                "Wait for the active turn to finish before forking this coding task",
            ) as parent:
                new_id = str(uuid.uuid4())
                project = self.projects.get(parent.project_id) if parent.project_id else None
                if project and parent.workspaces:
                    prepared_project = self.project_workspaces.fork(new_id, project, parent.workspaces)
                    prepared = prepared_project.primary
                    child_workspaces = list(prepared_project.workspaces)
                    child_ports = prepared_project.ports
                    parent_project_paths = {item.workspace_path for item in parent.workspaces[1:]}
                    external_dirs = [path for path in parent.additional_dirs if path not in parent_project_paths]
                    child_dirs = [item.workspace_path for item in child_workspaces[1:]] + external_dirs
                    child_environment = {
                        **project.environment.variables,
                        **{name: str(port) for name, port in child_ports.items()},
                    }
                    try:
                        guidance, _ = self.playbooks.guidance(project.id) if project.playbook else ("", None)
                    except Exception:
                        self.project_workspaces.cleanup(new_id, child_workspaces)
                        raise
                    instructions = project_instructions(
                        project,
                        child_workspaces,
                        parent.source_contexts,
                        guidance,
                    )
                    if parent.skill_instructions:
                        instructions = f"{instructions}\n\n{parent.skill_instructions}".strip()
                else:
                    prepared = self.workspaces.fork(
                        new_id,
                        parent.source_path,
                        parent.workspace_path,
                        parent.workspace_kind,
                        parent.base_revision,
                    )
                    child_workspaces = []
                    child_ports = {}
                    child_dirs = parent.additional_dirs
                    child_environment = parent.environment
                    instructions = parent.developer_instructions
                prepared_kind = prepared.workspace_kind if isinstance(prepared, TaskWorkspace) else prepared.kind
                prepared_warning = None if isinstance(prepared, TaskWorkspace) else prepared.warning
                try:
                    child_skill_roots = self.skill_runtime.clone(parent.id, new_id)
                    parent_runtime = self.runtimes.open_locked(parent, credentials)
                    child = parent.model_copy(
                        update={
                            "id": new_id,
                            "title": f"{parent.title} (fork)"[:200],
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
                            "archived": False,
                            "status": SessionStatus.completed,
                            "last_error": None,
                            "created_at": utc_now(),
                            "updated_at": utc_now(),
                        }
                    )
                    engine_session_id = parent_runtime.fork(
                        engine_workspace_path(child),
                        tuple(child.additional_dirs),
                    )
                    child.engine_session_id = engine_session_id
                    self.store.save_session(child)
                    self.store.copy_event_history(parent.id, child)
                    self._emit(
                        child.id,
                        CodingEvent(
                            type=EventType.session,
                            title="Task forked",
                            text=f"Forked from {parent.title} with its conversation and working changes.",
                            phase="completed",
                            data={"parentSessionId": parent.id},
                        ),
                    )
                    return self.get_session(child.id)
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
                    raise

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
        return self._submit_turn(session_id, prompt, credentials, attachments)

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
            if session_id not in self._running:
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
            if session_id not in self._running:
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

    def resolve_approval(self, session_id: str, approval_id: str, decision: ApprovalDecision) -> CodingSession:
        self.approvals.resolve(session_id, approval_id, decision)
        return self.get_session(session_id)

    def git_state(self, session_id: str):
        session = self.get_session(session_id)
        if session.workspaces:
            return self.project_workspaces.git_states(session.workspaces)[0]
        return self.workspaces.git_state(session.source_path, session.workspace_path)

    def git_states(self, session_id: str):
        session = self.get_session(session_id)
        if session.workspaces:
            return self.project_workspaces.git_states(session.workspaces)
        return [self.workspaces.git_state(session.source_path, session.workspace_path)]

    def diff(self, session_id: str):
        session = self.get_session(session_id)
        if session.workspaces:
            return self.project_workspaces.diff(session.workspaces)
        return self.workspaces.diff(session.workspace_path, session.base_revision)

    def create_branch(self, session_id: str, name: str):
        with self._maintenance_session(
            session_id,
            "Wait for the active turn to finish before changing task branches",
        ) as session:
            if session.workspaces:
                states = []
                for workspace in session.workspaces:
                    if workspace.workspace_kind != WorkspaceKind.git_worktree:
                        continue
                    state = self.workspaces.create_branch(workspace.workspace_path, name)
                    states.append(state.model_copy(update={"folder_id": workspace.folder_id, "folder_name": workspace.folder_name}))
                if not states:
                    raise WorkspaceError("This project does not contain a Git folder")
                self._emit(session.id, CodingEvent(type=EventType.session, title="Branches created", text=name, phase="completed"))
                return states[0]
            if session.base_revision is None:
                raise WorkspaceError("This task is not using a Git worktree")
            state = self.workspaces.create_branch(session.workspace_path, name)
            self._emit(session.id, CodingEvent(type=EventType.session, title="Branch created", text=name, phase="completed"))
        return state

    def commit(self, session_id: str, message: str):
        with self._maintenance_session(
            session_id,
            "Wait for the active turn to finish before committing task changes",
        ) as session:
            if session.workspaces:
                states = self.project_workspaces.commit(session.workspaces, message)
                if not states:
                    raise WorkspaceError("This project does not contain a Git folder")
                self._emit(session.id, CodingEvent(type=EventType.session, title="Changes committed", text=message.strip(), phase="completed"))
                return states[0]
            if session.base_revision is None:
                raise WorkspaceError("This task is not using a Git worktree")
            state = self.workspaces.commit(session.workspace_path, message)
            self._emit(session.id, CodingEvent(type=EventType.session, title="Changes committed", text=message.strip(), phase="completed"))
        return state

    def apply_to_source(self, session_id: str) -> dict[str, str | None]:
        with self._maintenance_session(
            session_id,
            "Wait for the active turn to finish before applying task changes",
        ) as session:
            if session.workspaces:
                applied = self.project_workspaces.apply(session.id, session.workspaces)
                text = "No changes to apply." if not applied else f"Applied reviewed changes across {len(applied)} project folders."
                self._emit(session.id, CodingEvent(type=EventType.session, title="Handoff complete", text=text, phase="completed"))
                return {"status": "applied" if applied else "no_changes", "snapshot": None}
            if session.workspace_kind == WorkspaceKind.direct_folder:
                raise WorkspaceError("This legacy task uses its source folder directly and has no isolated handoff")
            snapshot = self.workspaces.apply_to_source(
                session.id,
                session.source_path,
                session.workspace_path,
                session.base_revision,
            )
            text = "No changes to apply." if snapshot is None else "Applied the task diff to the source working tree. A recovery patch was saved first."
            self._emit(session.id, CodingEvent(type=EventType.session, title="Handoff complete", text=text, phase="completed"))
        return {"status": "no_changes" if snapshot is None else "applied", "snapshot": str(snapshot) if snapshot else None}

    def validate_project(self, session_id: str) -> list[dict]:
        with self._maintenance_session(
            session_id,
            "Wait for the active turn to finish before running project checks",
        ) as session:
            if not session.project_id or not session.workspaces:
                return []
            project = self.projects.get(session.project_id)
            results = self.project_workspaces.run_commands(
                project,
                session.workspaces,
                "validate",
                session.allocated_ports,
            )
            for result in results:
                self._emit(
                    session.id,
                    CodingEvent(
                        type=EventType.command,
                        title=result.label,
                        text=result.output,
                        phase="completed" if result.return_code == 0 else "failed",
                        data={"folderId": result.folder_id, "returnCode": result.return_code, "phase": "validate"},
                    ),
                )
            return [result.__dict__ for result in results]

    def delivery_plan(
        self,
        session_id: str,
        integrations: DeveloperIntegrationService | None = None,
    ):
        session = self.get_session(session_id)
        if not session.project_id or not session.workspaces:
            raise WorkspaceError("This task is not linked to a Code Project")
        project = self.projects.get(session.project_id)
        return self.delivery.plan(session, project, integrations)

    def create_draft_pull_requests(
        self,
        session_id: str,
        request: DraftPullRequestRequest,
        integrations: DeveloperIntegrationService,
    ):
        with self._maintenance_session(
            session_id,
            "Wait for the active turn to finish before publishing draft pull requests",
        ) as session:
            if not session.project_id or not session.workspaces:
                raise WorkspaceError("This task is not linked to a Code Project")
            project = self.projects.get(session.project_id)
            records = self.delivery.create_draft_pull_requests(session, project, request, integrations)
            self.store.update_session(
                session.id,
                lambda current: current.deliveries.extend(records),
            )
            published = sum(item.status == "published" for item in records)
            failed = sum(item.status == "failed" for item in records)
            self._emit(
                session.id,
                CodingEvent(
                    type=EventType.session,
                    title="Draft pull requests prepared",
                    text=f"Created {published} draft pull request(s)." + (f" {failed} folder(s) need attention." if failed else ""),
                    phase="failed" if failed and not published else "completed",
                ),
            )
            return records

    def pull_request_action(
        self,
        session_id: str,
        request: PullRequestActionRequest,
        integrations: DeveloperIntegrationService,
    ) -> PullRequestStatus:
        with self._maintenance_session(
            session_id,
            "Wait for the active turn to finish before updating a pull request",
        ) as session:
            if not session.project_id:
                raise WorkspaceError("This task is not linked to a Code Project")
            project = self.projects.get(session.project_id)
            status = integrations.pull_request_action(project, request)
            self._emit(
                session.id,
                CodingEvent(
                    type=EventType.session,
                    title="Pull request updated",
                    text="Marked ready for review." if request.action == "ready" else "Merged on GitHub.",
                    phase="completed",
                ),
            )
            return status

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

    def terminal(self, session_id: str, after: int = 0) -> TerminalPage:
        self.get_session(session_id)
        return self.runtimes.terminal_page(session_id, after)

    def wait_for_terminal(self, session_id: str, after: int, timeout: float = 15.0) -> TerminalPage:
        self.get_session(session_id)
        return self.runtimes.wait_for_terminal(session_id, after, timeout)

    def start_terminal(
        self,
        session_id: str,
        credentials: EngineCredentials,
        cols: int,
        rows: int,
    ) -> TerminalPage:
        session = self.get_session(session_id)
        try:
            return self.runtimes.start_terminal(session, credentials, cols, rows)
        except Exception as exc:
            message = safe_engine_error(str(exc), credentials)
            raise RuntimeError(message) from exc

    def write_terminal(self, session_id: str, data_base64: str) -> TerminalPage:
        self.get_session(session_id)
        return self.runtimes.write_terminal(session_id, data_base64)

    def resize_terminal(self, session_id: str, cols: int, rows: int) -> TerminalPage:
        self.get_session(session_id)
        return self.runtimes.resize_terminal(session_id, cols, rows)

    def stop_terminal(self, session_id: str) -> TerminalPage:
        self.get_session(session_id)
        return self.runtimes.stop_terminal(session_id)

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
        return self.store.append_event(session_id, event, update)

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
