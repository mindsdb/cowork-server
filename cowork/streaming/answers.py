"""Pending-question registry: where a blocked turn waits for an answer.

A separate object rather than a field on ``RunHandle`` for a practical
reason: ``registry.start()`` takes an already-constructed producer coroutine
and the ``RunHandle`` only exists afterwards, so the coroutine cannot reach
it. Keying by ``conversation_id`` sidesteps the ordering problem and keeps
this testable on its own.

Single-instance by design, exactly like ``RunRegistry`` and the
``FileStreamBuffer`` next to it — one cowork-server process per user, so an
in-memory ``asyncio.Future`` is sufficient: the POST carrying the answer
reaches the same process that holds the run by construction. A future reader
should not mistake this for an oversight that needs to become a database
table or a cross-process queue.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anton.core.interaction.elicit import AskRequest

logger = logging.getLogger(__name__)

__all__ = ["AnswerBroker", "SubmitResult", "broker"]


class SubmitResult(Enum):
    """Why a submit was accepted or not. Three states, not a bool: the
    endpoint has to tell "unknown question" (404) from "already answered"
    (409) from "not one of the options" (400), and those are three different
    situations for the frontend."""

    ACCEPTED = "accepted"
    NOT_FOUND = "not_found"
    ALREADY_ANSWERED = "already_answered"
    INVALID_OPTION = "invalid_option"


@dataclass
class _Pending:
    """One open question: the future its answer resolves, plus the request.

    The request is not a convenience — it is the *only* record of what was
    offered, and ``submit`` needs it to check an incoming answer against the
    declared contract (which options, how many of them, whether free text was
    invited). Without it the HTTP path would accept any shape and the model
    would receive an answer to a question it did not ask.
    """

    future: asyncio.Future
    request: "AskRequest"


class AnswerBroker:
    """Process-wide map of questions awaiting an answer."""

    def __init__(self) -> None:
        self._pending: dict[tuple[str, str], _Pending] = {}

    def open(self, conversation_id: str, question_id: str, request: "AskRequest") -> asyncio.Future:
        """Register *question_id* and return the future its answer resolves.

        Idempotent: a second call for the same key returns the same future
        rather than orphaning the first. ``elicit()`` calls ``begin()`` before
        publishing and the elicitor's ``ask()`` needs the same future, so both
        come through here.

        The request is stored alongside so ``submit`` can check the answer
        against the options that were actually offered.
        """
        key = (conversation_id, question_id)
        existing = self._pending.get(key)
        if existing is not None:
            # Unconditional, including an already-resolved future: the
            # invariant "one question_id, one future" is simpler to reason
            # about than "one, unless it was answered". Replacing a resolved
            # entry would make a duplicate submit look like a fresh one and
            # return 200 where the client expects 409.
            return existing.future
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[key] = _Pending(future=future, request=request)
        return future

    def submit(self, conversation_id: str, question_id: str, answer: dict) -> SubmitResult:
        """Resolve the question's future with *answer*, if it fits the offer.

        Validation lives here rather than in the endpoint on purpose: the
        request is only reachable from this side, and a second caller (a future
        transport, a test harness) must not be able to route around the rule.
        """
        pending = self._pending.get((conversation_id, question_id))
        if pending is None:
            return SubmitResult.NOT_FOUND
        if pending.future.done():
            return SubmitResult.ALREADY_ANSWERED

        if not answer.get("skipped"):
            rejection = self._violation(pending.request, answer)
            if rejection is not None:
                logger.info(
                    "Rejected answer for question %s on conversation %s: %s",
                    question_id, conversation_id, rejection,
                )
                return SubmitResult.INVALID_OPTION

        pending.future.set_result(answer)
        return SubmitResult.ACCEPTED

    @staticmethod
    def _violation(request: "AskRequest", answer: dict) -> str | None:
        """Why *answer* does not fit *request*, or None if it does.

        Rejects rather than truncates. ``CLIElicitor`` truncates a
        multi-selection down to one because a human typed it at a terminal and
        the intent is recoverable; the HTTP caller is a GUI that rendered this
        exact card, so a shape it never offered is a frontend bug and silently
        altering the answer would hide it. All three shapes map onto the
        spec's ``invalid_option``, since from the model's side they are the
        same failure: an answer that was not on the menu.
        """
        values = answer.get("values") or []
        if values:
            offered = {option.value for option in request.options}
            unknown = sorted(set(values) - offered)
            if unknown:
                return f"values {unknown} were never offered"
            if len(set(values)) != len(values):
                return "values contains duplicates"
            if request.select == "one" and len(values) > 1:
                return f"{len(values)} values for a single-choice question"
        if answer.get("text") and not request.allow_custom:
            return "free-form text for a question that does not allow it"
        return None

    def close(self, conversation_id: str, question_id: str) -> None:
        """Drop the entry. Idempotent — the owning ``elicit()`` calls this in
        a ``finally``, which can run after a submit or without one."""
        pending = self._pending.pop((conversation_id, question_id), None)
        if pending is not None and not pending.future.done():
            pending.future.cancel()

    def reset(self) -> None:
        """Drop every entry, cancelling any future still unresolved.

        For tests: this object is a process global, so a question opened by one
        test would otherwise stay answerable in the next. Cancelling (rather
        than just clearing) keeps the ``close()`` invariant — an entry never
        leaves this map with a live future nobody will ever resolve.
        """
        for key in list(self._pending):
            self.close(*key)


# Single global instance per server process, mirroring streaming.registry.
broker: AnswerBroker = AnswerBroker()
