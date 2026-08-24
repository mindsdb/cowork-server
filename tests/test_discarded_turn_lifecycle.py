"""Deleting/editing a message while a turn waits on a question.

Before `ask_user` a turn could never be alive-but-idle, so `registry.discard`
only popped the dict entry: "a delete never races an in-flight stream". A turn
blocked on a question looks idle for up to 300 s, which is exactly when a user
edits an earlier message — and the popped-but-running producer then had its
answer endpoint 404 forever while it blocked to the full timeout.

So discard now cancels the producer too, and marks it discarded so its
`except CancelledError` handler does not persist rows into (or recreate a
stream buffer for) history that was just truncated.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from anton.core.interaction.elicit import AskOption, AskRequest
import cowork.handlers.responses as responses_mod
from cowork.handlers.responses import ResponsesHandler
from cowork.streaming.answers import SubmitResult, broker
from cowork.streaming.registry import RunHandle, TurnLifecycle, discard_conversation, registry

CID = "conv-discard-test"
QID = "ask:1"
_REQUEST = AskRequest(prompt="Which database?", options=(AskOption(value="pg", label="postgres"),))


@pytest.fixture(autouse=True)
def _clean_globals():
    yield
    broker.reset()
    registry.reset()


class _FakeBuffer:
    def __init__(self) -> None:
        self.records: list[tuple] = []
        self.closed: str | None = None

    @property
    def latest_seq(self) -> int:
        return len(self.records)

    async def append(self, type_, data):
        self.records.append((type_, data))
        return len(self.records)

    async def close(self, reason, extra=None):
        self.closed = reason


def _blocked_handler(monkeypatch, saved: dict, asked: asyncio.Event):
    """A handler whose turn publishes a question and blocks on the broker."""
    handler = object.__new__(ResponsesHandler)
    handler.principal = object()

    class FakeConversationService:
        def __init__(self, session):
            pass

        def get_conversation(self, conv_id):
            return object()

        def save_user_message(self, conv_id, content, *, created_at=None, pending=False):
            saved["user"] = content
            return SimpleNamespace(id=uuid4())

        def finalize_pending(self, conv_id, message_id=None):
            saved["finalized"] = True

        def save_assistant_turn(self, conv_id, text, events, harness=None, tool_rows=None):
            saved["assistant"] = text
            saved["events"] = events

    class FakeSession:
        def close(self):
            pass

    async def formatter(stream, model, event_sink):
        # What the elicitor does: publish the card, then wait on the broker.
        event_sink("response.ask_user", {"type": "response.ask_user", "question_id": QID})
        yield "event: response.ask_user\ndata: {}\n\n"
        future = broker.open(CID, QID, _REQUEST)
        asked.set()
        try:
            await future
        finally:
            broker.close(CID, QID)
        yield "event: response.completed\ndata: {}\n\n"

    monkeypatch.setattr(responses_mod, "ConversationService", FakeConversationService)
    monkeypatch.setattr(responses_mod, "ScopedSession", lambda s, scope: FakeSession())
    monkeypatch.setattr(responses_mod, "get_open_session", lambda: None)
    monkeypatch.setattr(responses_mod, "scope_from_principal", lambda p: None)
    monkeypatch.setattr(responses_mod, "get_harness", lambda name: SimpleNamespace(
        stream_response=lambda **kwargs: None, formatter=formatter,
    ))
    return handler


async def _start_blocked_turn(monkeypatch, saved, buffer):
    """Register a real producer that is blocked on a pending question."""
    asked = asyncio.Event()
    handler = _blocked_handler(monkeypatch, saved, asked)
    lifecycle = TurnLifecycle()
    coro = handler._run_turn(
        conv_id=uuid4(), harness_input=[], original_content="hi", model="anton",
        disabled=None, harness_name="anton", harness_id="anton", buffer=buffer,
        lifecycle=lifecycle,
    )
    handle = await registry.start(
        conversation_id=CID, turn_id=0, buffer=buffer, producer_coro=coro,
        lifecycle=lifecycle,
    )
    await asyncio.wait_for(asked.wait(), timeout=5)
    return handle


async def test_discard_unblocks_the_turn_and_persists_nothing(monkeypatch):
    saved: dict = {}
    buffer = _FakeBuffer()
    handle = await _start_blocked_turn(monkeypatch, saved, buffer)

    # The delete endpoint is a sync `def`, so FastAPI runs it in a threadpool
    # where there is no running loop — discard must work from there.
    await asyncio.to_thread(discard_conversation, CID)

    # 5 s, not 300: the producer must be cancelled, not left on its timeout.
    await asyncio.wait_for(asyncio.gather(handle.task, return_exceptions=True), timeout=5)

    assert handle.task.done()
    assert registry.get(CID) is None
    # Nothing written into the truncated conversation...
    assert "assistant" not in saved and "events" not in saved and "finalized" not in saved
    # ...and no terminal record, which would recreate the buffer file
    # discard_conversation just deleted (turn_id is reused after truncation).
    assert buffer.closed is None
    # The question is gone with the turn, so a late click cannot resolve a
    # future nobody is waiting on.
    assert broker.submit(CID, QID, {"values": ["pg"]}) is SubmitResult.NOT_FOUND


async def test_stop_still_persists_a_blocked_turn(monkeypatch):
    """The counterpart: an ordinary Stop (no discard) keeps its partial turn.

    Without this, "discarded => no persist" could be satisfied by never
    persisting on cancellation at all.
    """
    saved: dict = {}
    buffer = _FakeBuffer()
    handle = await _start_blocked_turn(monkeypatch, saved, buffer)

    assert await handle.cancel() is True
    assert "events" in saved
    assert buffer.closed == "cancelled"


async def test_cancel_returns_true_when_the_task_absorbs_the_cancellation():
    """RunHandle.cancel is `-> bool`; every producer swallows CancelledError to
    persist and close, so the normal-completion path must not return None."""

    async def _absorbs():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            return "cleaned up"

    task = asyncio.create_task(_absorbs())
    await asyncio.sleep(0)
    handle = RunHandle(conversation_id=CID, turn_id=0, buffer=_FakeBuffer(), task=task)

    assert await handle.cancel() is True
    assert await handle.cancel() is False  # already done


async def test_discard_of_an_unknown_or_finished_conversation_is_a_no_op():
    registry.discard("no-such-conversation")

    async def _done():
        return None

    task = asyncio.create_task(_done())
    await task
    lifecycle = TurnLifecycle()
    handle = RunHandle(conversation_id=CID, turn_id=0, buffer=_FakeBuffer(), task=task,
                       lifecycle=lifecycle)
    registry._by_cid[CID] = handle
    registry.discard(CID)
    # Flagged (the handle is gone either way) but nothing to cancel.
    assert lifecycle.discarded is True
    assert registry.get(CID) is None
