"""Cowork's Elicitor: open a future, wait for the HTTP answer, close.

It does not publish the question -- anton's ``elicit()`` emits
``StreamAskUser`` through the session emitter, which the stream formatter
turns into an SSE record. And it does not clean up: ``elicit()`` pairs
``begin`` with ``end`` in a ``finally``, so every early exit is covered
without this class duplicating the logic.
"""

from __future__ import annotations

import asyncio
import logging

from anton.core.interaction.elicit import AskAnswer, AskRequest

from cowork.streaming.answers import AnswerBroker

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_TIMEOUT_S", "CoworkElicitor"]

# Long enough for a real decision, short enough that a forgotten card returns
# the model to work. There is no other bound: dispatch_tool is not wrapped in
# wait_for and there is no turn idle timeout. Named so the harness's call site
# does not have to restate it — one number, one place.
DEFAULT_TIMEOUT_S = 300


class CoworkElicitor:
    """Surfaces questions as inline cards in the cowork chat."""

    # No file-browser widget yet, so path questions stay CLI-only and
    # select_path keeps degrading to picker_unavailable here.
    supported_kinds = ("choice",)

    answer_hint = (
        "The user sees the options as clickable buttons and may also type a "
        "free-form answer, or skip the question entirely."
    )

    def __init__(
        self,
        conversation_id: str,
        broker: AnswerBroker,
        *,
        timeout_s: int = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._conversation_id = conversation_id
        self._broker = broker
        self.timeout_s = timeout_s

    async def begin(self, question_id: str, request: AskRequest) -> None:
        self._broker.open(self._conversation_id, question_id, request)

    async def ask(self, question_id: str, request: AskRequest) -> AskAnswer:
        future = self._broker.open(self._conversation_id, question_id, request)
        timeout = request.timeout_s if request.timeout_s is not None else self.timeout_s
        try:
            payload = await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
        except asyncio.TimeoutError:
            logger.info("Question %s timed out after %ss.", question_id, timeout)
            return AskAnswer(status="timeout")

        if payload.get("skipped"):
            return AskAnswer(status="cancelled")
        return AskAnswer(
            status="answered",
            values=tuple(payload.get("values") or ()),
            text=(payload.get("text") or ""),
        )

    async def end(self, question_id: str) -> None:
        self._broker.close(self._conversation_id, question_id)
