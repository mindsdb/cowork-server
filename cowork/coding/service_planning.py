from __future__ import annotations

from cowork.coding.contracts import CodingEvent, CodingSession, EventType, ModeTurnRequest, SessionStatus
from cowork.coding.control_errors import StateConflict
from cowork.coding.engines.base import EngineCredentials


class CodingPlanningOperations:
    def submit_mode_turn(self, session_id: str, request: ModeTurnRequest, credentials: EngineCredentials) -> CodingSession:
        """Change mode and start its turn under the same lifecycle reservation.

        A failed start restores the previous mode. The event-count precondition
        prevents two windows from approving a superseded plan.
        """
        with self.runtimes.session_lock(session_id):
            with self._maintenance_session(session_id, "Wait for this turn to finish before changing mode") as session:
                if self._is_remote(session):
                    raise ValueError("Plan mode is available on this computer only")
                if session.event_count != request.expected_event_count:
                    raise StateConflict("This task changed. Review the latest plan and try again")
                if self.runtimes.terminal_is_running(session_id):
                    raise StateConflict("Stop the task terminal before changing mode")
                target = session.model_copy(update={"task_mode": request.task_mode})
                intent = self._validated_command_intent(target, request.prompt, request.attachments)
                if intent.runs_immediately:
                    raise ValueError("Send an instruction to change modes; run this command without changing mode")
                previous = session.task_mode
                self.runtimes.close_locked(session_id)
                self.store.update_session(session_id, lambda current: setattr(current, "task_mode", request.task_mode))
                try:
                    self._continue_completed_task(self.get_session(session_id))
                    return self._submit_turn(session_id, request.prompt, credentials, request.attachments, maintenance_reserved=True)
                except Exception as exc:
                    def restore(current: CodingSession) -> None:
                        current.task_mode = previous
                        # A rejected preflight may have reserved a fresh run,
                        # but it must not consume the completed plan's decision.
                        # Worker-start failures already have a failed run and
                        # retain the normal recovery flow instead.
                        if current.status == SessionStatus.ready:
                            current.status = session.status
                            current.last_error = session.last_error

                    self._emit(session_id, CodingEvent(
                        type=EventType.session, title="Could not change task mode",
                        text=str(exc), phase="failed",
                    ), restore)
                    raise
