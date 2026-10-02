"""Structured agent questions. Answers never grant command permissions."""
from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, model_validator

AnswerText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8_000)]


class QuestionOption(BaseModel):
    label: str = Field(min_length=1, max_length=512)
    description: str = Field(default="", max_length=4_000)


class AgentQuestion(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    header: str = Field(default="Question", max_length=256)
    question: str = Field(min_length=1, max_length=8_000)
    options: list[QuestionOption] | None = Field(default=None, max_length=12)
    isOther: bool = True
    isSecret: bool = False


class PendingQuestion(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    questions: list[AgentQuestion] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def unique_ids(self) -> PendingQuestion:
        if len({question.id for question in self.questions}) != len(self.questions):
            raise ValueError("Question identifiers must be unique")
        return self


class QuestionResponse(BaseModel):
    answers: dict[str, list[AnswerText]] = Field(max_length=3)

    def validate_for(self, pending: PendingQuestion) -> None:
        if set(self.answers) != {question.id for question in pending.questions}:
            raise ValueError("Answer each question before continuing")
        for question in pending.questions:
            values = self.answers[question.id]
            # Codex questions are single-choice, with an optional free-text answer.
            if len(values) != 1:
                raise ValueError("Choose one answer for each question")
            if question.options and not question.isOther and not question.isSecret and values[0] not in {option.label for option in question.options}:
                raise ValueError("Choose one of the offered answers")

    def engine_payload(self) -> dict:
        return {"answers": {key: {"answers": values} for key, values in self.answers.items()}}


@dataclass
class _Waiter:
    session_id: str
    pending: PendingQuestion
    event: threading.Event = field(default_factory=threading.Event)
    response: QuestionResponse | None = None
    closed: bool = False


class QuestionBroker:
    def __init__(
        self,
        on_open: Callable[[str, PendingQuestion], None],
        on_close: Callable[[str, PendingQuestion, QuestionResponse | None], None],
        timeout: float = 3600,
    ) -> None:
        self._on_open, self._on_close = on_open, on_close
        self._timeout = timeout
        self._lock = threading.RLock()
        self._waiters: dict[str, _Waiter] = {}

    def request(self, session_id: str, params: dict) -> dict:
        pending = PendingQuestion.model_validate({"questions": params.get("questions")})
        waiter = _Waiter(session_id, pending)
        with self._lock:
            if any(item.session_id == session_id for item in self._waiters.values()):
                return {"answers": {}}
            self._waiters[pending.id] = waiter
            try:
                # Open and cancel are serialized; cancellation cannot miss a
                # waiter whose pending state has not yet been persisted.
                self._on_open(session_id, pending)
            except Exception:
                self._waiters.pop(pending.id, None)
                raise
        try:
            if not waiter.event.wait(self._timeout):
                with self._lock:
                    if not waiter.closed:
                        self._close(waiter, None)
            return waiter.response.engine_payload() if waiter.response else {"answers": {}}
        finally:
            with self._lock:
                self._waiters.pop(pending.id, None)

    def resolve(self, session_id: str, question_id: str, response: QuestionResponse) -> None:
        with self._lock:
            waiter = self._waiters.get(question_id)
            if waiter is None or waiter.session_id != session_id or waiter.closed:
                raise KeyError("This question is no longer waiting for an answer")
            response.validate_for(waiter.pending)
            self._close(waiter, response)

    def _close(self, waiter: _Waiter, response: QuestionResponse | None) -> None:
        waiter.closed = True
        try:
            self._on_close(waiter.session_id, waiter.pending, response)
            waiter.response = response
        finally:
            # A failed write must not deliver an unrecorded answer or strand
            # the Codex reader. Empty answers cancel, never authorize a tool.
            waiter.event.set()

    def cancel_session(self, session_id: str) -> None:
        with self._lock:
            for waiter in self._waiters.values():
                if waiter.session_id == session_id and not waiter.closed:
                    self._close(waiter, None)
