from __future__ import annotations

from cowork.coding.contracts import CodingEvent, CodingSession, EventType, SessionStatus
from cowork.coding.questions import PendingQuestion, QuestionResponse

USER_INPUT_METHOD = "item/tool/requestUserInput"


class CodingQuestionOperations:
    """Route input requests separately from security approvals."""

    def _engine_request(self, session_id: str, method: str, params: dict | None) -> dict:
        if method == USER_INPUT_METHOD:
            return self.questions.request(session_id, params or {})
        return self.approvals.request(session_id, method, params)

    def answer_question(self, session_id: str, question_id: str, response: QuestionResponse) -> CodingSession:
        session = self.get_session(session_id)
        if self._is_remote(session):
            raise ValueError("Answering structured questions requires a local task in this version")
        self.questions.resolve(session_id, question_id, response)
        return self.get_session(session_id)

    def _question_opened(self, session_id: str, pending: PendingQuestion) -> None:
        def update(current: CodingSession) -> None:
            current.pending_question = pending
            # Preserve the existing run-state protocol: awaiting_approval is
            # the durable waiting-for-user state; the typed payload identifies
            # whether the UI presents a question or a permission decision.
            current.status = SessionStatus.awaiting_approval

        with self._lock:
            running = self._running.get(session_id)
            if running is None or running.cancel_requested:
                raise RuntimeError("This turn is no longer accepting questions")
            self._emit(session_id, CodingEvent(
                type=EventType.session, title="Answer needed", phase="pending",
                data={"questionId": pending.id},
            ), update)

    def _question_closed(self, session_id: str, pending: PendingQuestion, response: QuestionResponse | None) -> None:
        def update(current: CodingSession) -> None:
            if current.pending_question and current.pending_question.id == pending.id:
                current.pending_question = None
                if current.status == SessionStatus.awaiting_approval:
                    current.status = SessionStatus.running

        # Secret values go only to the engine. They never enter task history,
        # events, OS notifications, or control-plane snapshots.
        answer_text = "\n\n".join(
            f"{question.header}: {'[private answer]' if question.isSecret else response.answers[question.id][0]}"
            for question in pending.questions
        ) if response else "The question was cancelled."
        self._emit(session_id, CodingEvent(
            type=EventType.user_message if response else EventType.session,
            title="Your answers" if response else "Question cancelled",
            text=answer_text, phase="completed", data={"questionId": pending.id},
        ), update)
