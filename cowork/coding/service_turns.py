from __future__ import annotations

import threading
import uuid

from cowork.coding.commands import CommandIntent
from cowork.coding.connector_capabilities import ConnectorCapability
from cowork.coding.context import validate_references
from cowork.coding.contracts import (
    CodingEvent,
    CodingSession,
    EventType,
    InputReference,
    QueuedInstruction,
    SessionCreateRequest,
    SessionStatus,
    TaskCapability,
)
from cowork.coding.control_errors import StateConflict
from cowork.coding.control_models import TERMINAL_RUN_STATUSES, RunStatus, RuntimeCommand, RuntimeEvent, TaskRun
from cowork.coding.engines.base import EngineCredentials, EngineSession
from cowork.coding.turns import RunningTurn, fail_turn, mark_running
from cowork.services.skills import CodeSkillService


class CodingTurnOperations:
    """Task-turn lifecycle shared by local and connected-computer runs."""

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
            intent = self._validated_command_intent(session, request.prompt, request.attachments)
            self.remote.queue_turn(session, request.prompt, request.attachments, intent)
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
            intent = self._validated_command_intent(session, prompt, attachments)
            return self.remote.queue_turn(session, prompt, attachments, intent)
        return self._submit_turn(session_id, prompt, credentials, attachments)

    def _continue_completed_task(self, session: CodingSession) -> CodingSession:
        if not session.run_id:
            return session
        run = self.control.store.get_run(session.run_id)
        finished = run.status in TERMINAL_RUN_STATUSES or (
            run.status == RunStatus.interrupted and run.computer_id == self.control.local_computer.id
        )
        if not finished:
            return session
        continued = self.control.continue_task(session.task_id or session.id, run.id)

        def link(current: CodingSession) -> None:
            current.run_id = continued.id
            current.runtime_epoch = continued.epoch
            current.status = SessionStatus.ready
            current.last_error = None

        self.store.update_session(session.id, link)
        return self.get_session(session.id)

    def accept_runtime_event(self, event: RuntimeEvent) -> TaskRun:
        return self.remote.accept_event(event)

    def acknowledge_runtime_command(
        self,
        run_id: str,
        command_id: str,
        computer_id: str,
        lease_id: str,
        epoch: int,
        result: dict[str, object] | None = None,
        error: str | None = None,
    ) -> RuntimeCommand:
        return self.remote.acknowledge_command(run_id, command_id, computer_id, lease_id, epoch, result, error)

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
        intent = self._validated_command_intent(self.get_session(session_id), prompt, attachments)
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
            if session_id in self._running or session.status in {
                SessionStatus.running,
                SessionStatus.awaiting_approval,
            }:
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

    def _validated_command_intent(
        self,
        session: CodingSession,
        prompt: str,
        attachments: list[InputReference] | tuple[InputReference, ...],
    ) -> CommandIntent:
        intent = CommandIntent.parse(prompt)
        self.commands.validate(session, intent.name)
        intent.validate_arguments()
        intent.validate_attachments(bool(attachments))
        if intent.name and self._is_remote(session):
            self._require_task_capability(session, TaskCapability.slash_commands)
        return intent

    def steer(
        self,
        session_id: str,
        prompt: str,
        attachments: list[InputReference] | tuple[InputReference, ...] = (),
    ) -> CodingSession:
        session = self.get_session(session_id)
        if session.pending_approval is not None:
            # Codex does not accept turn/steer while the turn waits on an
            # approval, so the request would hang until the decision arrives.
            raise StateConflict("Resolve the pending approval first; the instruction can be queued meanwhile")
        if self._is_remote(session):
            if not session.run_id:
                raise RuntimeError("Remote task is missing its Task Run")
            if attachments:
                raise RuntimeError("File attachments cannot steer a task on another computer yet")
            run = self.control.store.get_run(session.run_id)
            if run.status not in {RunStatus.running, RunStatus.awaiting_approval}:
                raise RuntimeError("There is no active turn to steer")
            intent = self._validated_command_intent(session, prompt, attachments)
            intent.validate_while_running()
            command_kind = "agent_command" if intent.runs_immediately else "steer"
            command = self.control.queue_command(
                session.run_id,
                command_kind,
                intent.runtime_payload(prompt),
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
            self._emit(
                session.id,
                CodingEvent(type=EventType.user_message, title="You", text=prompt, phase="completed"),
            )
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
        self._emit(
            session.id,
            CodingEvent(type=EventType.user_message, title="Follow-up", text=prompt, phase="completed"),
        )
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
        """Promote one persisted queued instruction into the active turn."""
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

    def run_next_queued(
        self,
        session_id: str,
        credentials: EngineCredentials,
        instruction_id: str | None = None,
    ) -> CodingSession:
        """Start the oldest persisted instruction once the task is idle.

        When the caller names the instruction it expects to start, a queue
        whose head has moved on (typically because the same request already
        succeeded) is a conflict and nothing runs.
        """
        session = self.get_session(session_id)
        self._require_queue_head(session, instruction_id)
        if not session.queued_instructions:
            return session
        if self._is_remote(session):
            return self.remote.start_next_queued(session)
        with self._maintenance_session(
            session_id,
            "Wait for the active turn to finish before resuming queued work",
        ):
            expected_instruction_id = instruction_id
            while True:
                session = self._continue_completed_task(self.get_session(session_id))
                self._require_queue_head(session, expected_instruction_id)
                expected_instruction_id = None
                if not session.queued_instructions:
                    return session
                instruction = session.queued_instructions[0]
                target_id = instruction.id
                self.store.update_session(
                    session_id,
                    lambda current, target_id=target_id: self._remove_queued_instruction(current, target_id),
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

    @staticmethod
    def _require_queue_head(session: CodingSession, instruction_id: str | None) -> None:
        if instruction_id is None:
            return
        head = session.queued_instructions[0].id if session.queued_instructions else None
        if head != instruction_id:
            raise StateConflict("That queued instruction is no longer next in the queue")

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
