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
        pending = self._pending.get((conversation_id, question_id))
        if pending is None:
            return SubmitResult.NOT_FOUND
        if pending.future.done():
            return SubmitResult.ALREADY_ANSWERED

        values = answer.get("values") or []
        if values and not answer.get("skipped"):
            offered = {option.value for option in pending.request.options}
            if not set(values) <= offered:
                logger.info(
                    "Rejected answer for %s: values %s not among the offered options.",
                    question_id, sorted(set(values) - offered),
                )
                return SubmitResult.INVALID_OPTION

        pending.future.set_result(answer)
        return SubmitResult.ACCEPTED

    def close(self, conversation_id: str, question_id: str) -> None:
        """Drop the entry. Idempotent — the owning ``elicit()`` calls this in
        a ``finally``, which can run after a submit or without one."""
        pending = self._pending.pop((conversation_id, question_id), None)
        if pending is not None and not pending.future.done():
            pending.future.cancel()


# Single global instance per server process, mirroring streaming.registry.
broker: AnswerBroker = AnswerBroker()
