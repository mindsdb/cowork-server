from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from cowork.coding.context import EngineFailure, classify_engine_failure, safe_engine_error
from cowork.coding.contracts import CodingEvent, CodingSession, EventType, SessionStatus
from cowork.coding.engines.base import (
    EngineCredentials,
    EngineInputReference,
    EngineSession,
)
from cowork.coding.runtime import RuntimeManager
from cowork.coding.store import CodingStore

_TERMINAL_STATUSES = {
    "completed": SessionStatus.completed,
    "interrupted": SessionStatus.interrupted,
    "cancelled": SessionStatus.cancelled,
    "failed": SessionStatus.failed,
}


def terminal_status(reported: str, cancel_requested: bool) -> SessionStatus:
    if cancel_requested:
        return SessionStatus.cancelled
    return _TERMINAL_STATUSES.get(reported, SessionStatus.completed)


@dataclass
class RunningTurn:
    engine: EngineSession | None
    turn_id: str
    thread: threading.Thread
    cancel_requested: bool = False
    interruption_requested: bool = False
    pending_steers: list[tuple[str, tuple[EngineInputReference, ...]]] = field(default_factory=list)

    def route_steer(
        self,
        prompt: str,
        attachments: tuple[EngineInputReference, ...],
        *,
        require_ready: bool,
    ) -> tuple[EngineSession | None, str]:
        """Resolve a steer against the runtime state without leaking its startup race."""
        if require_ready and (self.engine is None or not self.turn_id):
            raise RuntimeError("The coding runtime is still starting. Try this command again in a moment")
        if self.turn_id:
            if self.engine is None:
                raise RuntimeError("Coding agent state is inconsistent")
            return self.engine, self.turn_id
        self.pending_steers.append((prompt, attachments))
        return None, ""


def mark_running(session: CodingSession) -> None:
    session.status = SessionStatus.running
    session.last_error = None


def record_engine_session(session: CodingSession, engine_session_id: str) -> None:
    session.engine_session_id = engine_session_id


def record_active_turn(session: CodingSession, turn_id: str) -> None:
    session.active_turn_id = turn_id


def finish_turn(session: CodingSession, status: SessionStatus) -> None:
    session.status = status
    session.active_turn_id = None
    session.pending_approval = None


def fail_turn(session: CodingSession, cancelled: bool, message: str) -> None:
    session.status = SessionStatus.cancelled if cancelled else SessionStatus.failed
    session.active_turn_id = None
    session.pending_approval = None
    session.last_error = None if cancelled else message


def interrupt_turn(session: CodingSession) -> None:
    session.status = SessionStatus.interrupted
    session.active_turn_id = None
    session.pending_approval = None
    session.last_error = None


class EventBuffer:
    """Coalesce wire-level token deltas into useful, still-live UI events."""

    def __init__(self, emit: Callable[[CodingEvent], None]) -> None:
        self._emit = emit
        self._pending: CodingEvent | None = None
        self._since = time.monotonic()

    def add(self, event: CodingEvent) -> None:
        pending = self._pending
        mergeable = event.type in {
            EventType.agent_message,
            EventType.reasoning,
            EventType.command,
            EventType.file_change,
        }
        if (
            mergeable
            and pending is not None
            and pending.type == event.type
            and pending.item_id == event.item_id
            and pending.turn_id == event.turn_id
            and len(pending.text) + len(event.text) <= 4_000
            and time.monotonic() - self._since < 0.15
        ):
            pending.text += event.text
            pending.title = event.title or pending.title
            pending.phase = event.phase or pending.phase
            pending.timestamp = event.timestamp
            if event.data:
                pending.data = event.data
            return
        self.flush()
        if mergeable:
            self._pending = event
            self._since = time.monotonic()
        else:
            self._emit(event)

    def flush(self) -> None:
        if self._pending is not None:
            self._emit(self._pending)
            self._pending = None


class TurnExecutor:
    """Run one reserved engine operation and settle its persisted task state."""

    def __init__(
        self,
        runtimes: RuntimeManager,
        store: CodingStore,
        running: dict[str, RunningTurn],
        state_lock: Any,
        get_session: Callable[[str], CodingSession],
        emit: Callable[..., CodingSession],
        run_next_queued: Callable[[str, EngineCredentials], CodingSession],
    ) -> None:
        self._runtimes = runtimes
        self._store = store
        self._running = running
        self._state_lock = state_lock
        self._get_session = get_session
        self._emit = emit
        self._run_next_queued = run_next_queued

    def execute(
        self,
        session_id: str,
        prompt: str,
        credentials: EngineCredentials,
        goal_objective: str | None = None,
        attachments: tuple[EngineInputReference, ...] = (),
        review: bool = False,
        goal_resume: bool = False,
    ) -> None:
        engine_session: EngineSession | None = None
        turn_id = ""
        finished_status: SessionStatus | None = None
        reservation_released = False
        model = ""
        try:
            session = self._get_session(session_id)
            model = session.model
            engine_session = self._runtimes.open(session, credentials)
            self._store.update_session(
                session_id,
                lambda current: record_engine_session(current, engine_session.session_id),
            )
            self._attach_runtime(session_id, engine_session)
            turn_id = self._start_operation(
                engine_session,
                prompt,
                attachments,
                goal_objective,
                review,
                goal_resume,
            )
            self._store.update_session(
                session_id,
                lambda current: record_active_turn(current, turn_id),
            )
            cancel_requested, pending_steers = self._activate_turn(session_id, turn_id)
            if cancel_requested:
                engine_session.cancel(turn_id)
            else:
                for pending_prompt, pending_attachments in pending_steers:
                    engine_session.steer(turn_id, pending_prompt, pending_attachments)

            status, terminal_event, error_text = self._collect_events(session_id, engine_session, turn_id)
            if self._interruption_requested(session_id):
                status = SessionStatus.interrupted
                terminal_event = None
            finished_status = status
            finish: Callable[[CodingSession], None] = lambda current: finish_turn(current, status)
            if status == SessionStatus.failed and terminal_event is not None:
                failure = classify_engine_failure(error_text or terminal_event.text, credentials, model)
                terminal_event = terminal_event.model_copy(
                    update={"text": failure.message, "data": {**terminal_event.data, **failure.event_data()}},
                )
                finish = lambda current: fail_turn(current, False, failure.message)
            # Make the terminal status and turn-slot release one observable
            # transition. Otherwise the UI can see "Completed" and offer
            # fork/archive immediately while the reservation still rejects it.
            with self._state_lock:
                if terminal_event is not None:
                    self._emit(session_id, terminal_event, finish)
                else:
                    self._store.update_session(session_id, lambda current: finish_turn(current, status))
                self._running.pop(session_id, None)
                reservation_released = True
        except Exception as exc:  # noqa: BLE001 - engine adapters may surface arbitrary SDK failures.
            try:
                if engine_session is not None:
                    try:
                        # Once an adapter operation has failed, its protocol
                        # state is no longer safe to reuse. Closing the owned
                        # runtime also terminates commands that could otherwise
                        # continue after Cowork has marked the turn failed.
                        self._runtimes.close_session(session_id)
                    except Exception:
                        # Preserve the original, user-visible turn failure. A
                        # Codex adapter has its own process watchdog as the
                        # final teardown fallback.
                        pass
            finally:
                with self._state_lock:
                    self._record_failure(session_id, classify_engine_failure(str(exc), credentials, model))
                    self._running.pop(session_id, None)
                    reservation_released = True
        finally:
            if not reservation_released:
                with self._state_lock:
                    self._running.pop(session_id, None)
            if engine_session is not None:
                self._runtimes.discard_if_closed(session_id, engine_session)
            if finished_status == SessionStatus.completed:
                self._continue_queue(session_id, credentials)

    def interrupt(self, session_ids: list[str]) -> int:
        """Persist app-shutdown interruptions before closing owned runtimes."""
        interrupted: list[str] = []
        with self._state_lock:
            for session_id in session_ids:
                running = self._running.get(session_id)
                if running is None or running.interruption_requested:
                    continue
                running.interruption_requested = True
                interrupted.append(session_id)
        for session_id in interrupted:
            self._emit(
                session_id,
                CodingEvent(
                    type=EventType.session,
                    title="Task interrupted",
                    text="Cowork closed while this task was running. Send a follow-up to resume from the saved working copy.",
                    phase="failed",
                ),
                interrupt_turn,
            )
        return len(interrupted)

    def _attach_runtime(self, session_id: str, engine_session: EngineSession) -> None:
        with self._state_lock:
            reserved = self._running.get(session_id)
            if reserved is None:
                raise RuntimeError("coding turn reservation disappeared")
            reserved.engine = engine_session

    def _activate_turn(
        self,
        session_id: str,
        turn_id: str,
    ) -> tuple[bool, list[tuple[str, tuple[EngineInputReference, ...]]]]:
        with self._state_lock:
            reserved = self._running.get(session_id)
            if reserved is None:
                raise RuntimeError("coding turn was cancelled before it started")
            reserved.turn_id = turn_id
            return reserved.cancel_requested, list(reserved.pending_steers)

    @staticmethod
    def _start_operation(
        engine_session: EngineSession,
        prompt: str,
        attachments: tuple[EngineInputReference, ...],
        goal_objective: str | None,
        review: bool,
        goal_resume: bool,
    ) -> str:
        if goal_resume:
            return engine_session.resume_goal()
        if goal_objective:
            return engine_session.start_goal(goal_objective)
        if review:
            return engine_session.start_review()
        return engine_session.start_turn(prompt, attachments)

    def _collect_events(
        self,
        session_id: str,
        engine_session: EngineSession,
        turn_id: str,
    ) -> tuple[SessionStatus, CodingEvent | None, str]:
        """Stream the turn's events; return its terminal status, the terminal event and the last error text."""
        buffer = EventBuffer(lambda event: self._emit(session_id, event))
        final_status = "completed"
        terminal_event: CodingEvent | None = None
        error_text = ""
        for event in engine_session.events(turn_id):
            if event.type == EventType.session and event.data.get("status"):
                final_status = str(event.data["status"])
                terminal_event = event
            else:
                if event.type == EventType.error and event.text:
                    error_text = event.text
                buffer.add(event)
        buffer.flush()
        with self._state_lock:
            cancelled = bool(self._running.get(session_id) and self._running[session_id].cancel_requested)
        return terminal_status(final_status, cancelled), terminal_event, error_text

    def _record_failure(self, session_id: str, failure: EngineFailure) -> None:
        with self._state_lock:
            running = self._running.get(session_id)
            cancelled = bool(running and running.cancel_requested)
            interrupted = bool(running and running.interruption_requested)
        if interrupted and not cancelled:
            self._store.update_session(session_id, interrupt_turn)
            return
        self._emit(
            session_id,
            CodingEvent(
                type=EventType.session if cancelled else EventType.error,
                title="Task stopped" if cancelled else "Coding agent failed",
                text="The active coding turn was cancelled." if cancelled else failure.message,
                phase="completed" if cancelled else "failed",
                data={} if cancelled else failure.event_data(),
            ),
            lambda current: fail_turn(current, cancelled, failure.message),
        )

    def _interruption_requested(self, session_id: str) -> bool:
        with self._state_lock:
            running = self._running.get(session_id)
            return bool(running and running.interruption_requested)

    def _continue_queue(self, session_id: str, credentials: EngineCredentials) -> None:
        try:
            self._run_next_queued(session_id, credentials)
        except Exception as exc:  # noqa: BLE001 - queued work must fail visibly without killing the worker.
            self._emit(
                session_id,
                CodingEvent(
                    type=EventType.error,
                    title="Queued instruction could not start",
                    text=safe_engine_error(str(exc), credentials),
                    phase="failed",
                ),
            )
