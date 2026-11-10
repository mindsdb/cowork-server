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
import importlib
from types import SimpleNamespace
from uuid import uuid4

import fakeredis
import fakeredis.aioredis
import pytest

from anton.core.interaction.elicit import AskOption, AskRequest
import cowork.handlers.responses as responses_mod
from cowork.handlers.responses import ResponsesHandler
from cowork.streaming import backend as backend_mod
from cowork.streaming import buffer as buffer_mod
from cowork.streaming import cancellation, turn_claim, turn_index
from cowork.streaming.answers import SubmitResult, broker
from cowork.streaming.buffer import stream_key
from cowork.streaming.registry import RunHandle, RunRegistry, TurnLifecycle, discard_conversation, registry
from cowork.streaming.turn_claim import CLAIMS_KEY, claim_key
from cowork.turnqueue.redis_client import cancel_cause_key

from _fakes import inline_run_db, opens

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

    @property
    def is_closed(self) -> bool:
        return self.closed is not None


def _blocked_handler(
    monkeypatch, saved: dict, asked: asyncio.Event, *, frame_on_cancel: str | None = None,
):
    """A handler whose turn publishes a question and blocks on the broker.
    With ``frame_on_cancel``, the stream sends that frame when the cancel
    lands and then raises the cancel on, as the anton harness does with a
    turn's memory record."""
    handler = object.__new__(ResponsesHandler)
    handler.principal = object()

    class FakeConversationService:
        def __init__(self, session):
            pass

        def get_conversation(self, conv_id):
            return SimpleNamespace(project=None)

        def save_user_message(self, conv_id, content, *, created_at=None, pending=False):
            saved["user"] = content
            return SimpleNamespace(id=uuid4())

        def clear_pending(self, conv_id, *, message_id=None):
            saved["finalized"] = True
            return True

        def save_assistant_turn(self, conv_id, text, events, harness=None, tool_rows=None):
            saved["assistant"] = text
            saved["events"] = events

    class FakeSession:
        def close(self):
            pass

    async def formatter(stream, model, event_sink, **_kwargs):
        # What the elicitor does: publish the card, then wait on the broker.
        event_sink("response.ask_user", {"type": "response.ask_user", "question_id": QID})
        yield "event: response.ask_user\ndata: {}\n\n"
        future = broker.open(CID, QID, _REQUEST)
        asked.set()
        try:
            await future
        except asyncio.CancelledError:
            if frame_on_cancel is not None:
                yield frame_on_cancel
            raise
        finally:
            broker.close(CID, QID)
        yield "event: response.completed\ndata: {}\n\n"

    monkeypatch.setattr(responses_mod, "ConversationService", FakeConversationService)
    monkeypatch.setattr(responses_mod, "run_db", inline_run_db(FakeSession()))
    monkeypatch.setattr(responses_mod, "scope_from_principal", lambda p: None)
    monkeypatch.setattr(responses_mod, "get_harness", lambda name: SimpleNamespace(
        stream_response=lambda **kwargs: None, formatter=formatter,
    ))
    return handler


async def _start_blocked_turn(monkeypatch, saved, buffer, *, frame_on_cancel: str | None = None):
    """Register a real producer that is blocked on a pending question."""
    asked = asyncio.Event()
    handler = _blocked_handler(monkeypatch, saved, asked, frame_on_cancel=frame_on_cancel)
    lifecycle = TurnLifecycle()
    handle = await registry.start(
        conversation_id=CID, turn_id=0, open_buffer=opens(buffer),
        produce=lambda buffer: handler._run_turn(
            conv_id=uuid4(), harness_input=[], original_content="hi", model="anton",
            disabled=None, harness_name="anton", harness_id="anton", buffer=buffer,
            lifecycle=lifecycle,
        ),
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


_MEMORY_FRAME = (
    'event: response.memory_read\ndata: {"type": "response.memory_read", "read": true, "entries": 1}\n\n'
)


def _memory_frames(buffer: _FakeBuffer) -> list[tuple]:
    return [record for record in buffer.records if "response.memory_read" in record[1].get("sse", "")]


async def test_discard_writes_no_frame_the_stream_sends_after_the_cancel(monkeypatch):
    """A delete removed the turn's buffer. On Redis, a write after that
    recreates the stream with a record that isn't terminal, and the next turn
    at this number keeps it."""
    saved: dict = {}
    buffer = _FakeBuffer()
    handle = await _start_blocked_turn(monkeypatch, saved, buffer, frame_on_cancel=_MEMORY_FRAME)

    await asyncio.to_thread(discard_conversation, CID)
    await asyncio.wait_for(asyncio.gather(handle.task, return_exceptions=True), timeout=5)

    assert handle.task.done()
    assert _memory_frames(buffer) == []
    assert buffer.closed is None
    assert "events" not in saved


async def test_stop_writes_the_frame_the_stream_sends_after_the_cancel(monkeypatch):
    """The counterpart: a Stop keeps the frame."""
    saved: dict = {}
    buffer = _FakeBuffer()
    handle = await _start_blocked_turn(monkeypatch, saved, buffer, frame_on_cancel=_MEMORY_FRAME)

    assert await handle.cancel() is True
    assert _memory_frames(buffer) == [("sse", {"sse": _MEMORY_FRAME})]
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


class _HeldCancelWatch:
    """The cancel watch's Redis client, holding each of its flag reads until
    ``allowed`` is set: the turn hears of a cancel only when the test says."""

    def __init__(self, client) -> None:
        self._client = client
        self.allowed = asyncio.Event()

    async def exists(self, *keys):
        await self.allowed.wait()
        return await self._client.exists(*keys)

    def __getattr__(self, name):
        return getattr(self._client, name)


@pytest.fixture
def two_replicas(monkeypatch):
    """One Redis, on the Redis backend, for two RunRegistry objects."""
    server = fakeredis.FakeServer()
    client = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    sync_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    watch = _HeldCancelWatch(client)
    for module in (turn_claim, turn_index, buffer_mod):
        monkeypatch.setattr(module, "get_redis", lambda: client)
    monkeypatch.setattr(cancellation, "get_redis", lambda: watch)
    for module in (turn_claim, turn_index, backend_mod, cancellation):
        monkeypatch.setattr(module, "get_sync_redis", lambda: sync_client)
    monkeypatch.setattr(cancellation, "CANCEL_POLL_SECONDS", 0.01)
    monkeypatch.setenv("COWORK_STREAM_BACKEND", "redis")
    return SimpleNamespace(redis=client, watch=watch)


async def _answered(buffer):
    await buffer.append("sse", {"sse": "event: response.completed\ndata: {}\n\n"})
    await buffer.close("completed")


async def test_a_delete_another_replica_serves_stops_the_turn_without_saving_and_admits_the_resend(
    monkeypatch, two_replicas,
):
    """Replica A runs a turn blocked on a question, and the delete lands on
    replica B, which holds no handle for it. B used to forget the index entry
    and remove the stream and leave A's turn running: A's keeper went on
    renewing the claim, so every replica refused the resend with 409 until
    the question timed out. B now writes the turn's cancel flag with the
    ``discarded`` cause, so A stops as a local delete stops it: nothing saved
    into the deleted history, no terminal record, and no stream left behind,
    not even the records A wrote before it heard of the delete."""
    registry_mod = importlib.import_module("cowork.streaming.registry")
    replica_a, replica_b = RunRegistry(), RunRegistry()
    saved: dict = {}
    asked = asyncio.Event()
    handler = _blocked_handler(monkeypatch, saved, asked)
    lifecycle = TurnLifecycle()
    handle = await replica_a.start(
        conversation_id=CID, turn_id=0, open_buffer=lambda: backend_mod.new_buffer(CID, 0),
        produce=lambda buffer: handler._run_turn(
            conv_id=uuid4(), harness_input=[], original_content="hi", model="anton",
            disabled=None, harness_name="anton", harness_id="anton", buffer=buffer,
            lifecycle=lifecycle,
        ),
        lifecycle=lifecycle,
    )
    await asyncio.wait_for(asked.wait(), timeout=5)
    monkeypatch.setattr(registry_mod, "registry", replica_b)

    await asyncio.to_thread(discard_conversation, CID)
    assert await two_replicas.redis.get(cancel_cause_key(lifecycle.identity.correlation_id)) == "discarded"
    # A frame the turn writes before it hears of the delete recreates its stream.
    await handle.buffer.append("sse", {"sse": "event: response.output_text.delta\ndata: {}\n\n"})
    two_replicas.watch.allowed.set()
    await asyncio.wait_for(asyncio.gather(handle.task, return_exceptions=True), timeout=5)

    assert lifecycle.discarded is True
    assert "assistant" not in saved and "events" not in saved and "finalized" not in saved
    assert await two_replicas.redis.exists(stream_key(CID, 0), claim_key(CID), CLAIMS_KEY) == 0
    for runs in (replica_a, replica_b):
        admitted = await runs.start(
            conversation_id=CID, turn_id=0, open_buffer=lambda: backend_mod.new_buffer(CID, 0), produce=_answered,
        )
        await asyncio.wait_for(admitted.task, timeout=2)
        records = await two_replicas.redis.xrange(stream_key(CID, 0))
        assert [fields["seq"] for _id, fields in records] == ["0", "1"]
