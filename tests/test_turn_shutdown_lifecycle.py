"""Server shutdown while a turn is in flight.

`RunRegistry.shutdown` cancels every running turn, marking each handle so
its own `CancelledError` handler persists an interrupted turn instead of
the silent, no-error-card save a user Stop gets. Mirrors
test_discarded_turn_lifecycle.py's fixtures.
"""
from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

import cowork.handlers.responses as responses_mod
from cowork.handlers.responses import ResponsesHandler
from cowork.handlers.turn_errors import GENERIC_TURN_ERROR_CODE, INTERRUPTED_TURN_MESSAGE
from cowork.streaming.registry import RunHandle, TurnLifecycle, registry

CID = "conv-shutdown-test"


@pytest.fixture(autouse=True)
def _clean_globals():
    yield
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


def _streaming_handler(monkeypatch, saved: dict, started: asyncio.Event):
    """A handler whose turn is mid-stream (a real delta already sent) when
    cancelled — the shape a shutdown actually interrupts."""
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
        event_sink("response.output_text.delta", {"type": "response.output_text.delta", "delta": "partial"})
        yield "event: response.output_text.delta\ndata: {}\n\n"
        started.set()
        await asyncio.sleep(3600)  # never reached without a cancel
        yield "event: response.completed\ndata: {}\n\n"

    monkeypatch.setattr(responses_mod, "ConversationService", FakeConversationService)
    monkeypatch.setattr(responses_mod, "ScopedSession", lambda s, scope: FakeSession())
    monkeypatch.setattr(responses_mod, "get_open_session", lambda: None)
    monkeypatch.setattr(responses_mod, "scope_from_principal", lambda p: None)
    monkeypatch.setattr(responses_mod, "get_harness", lambda name: SimpleNamespace(
        stream_response=lambda **kwargs: None, formatter=formatter,
    ))
    return handler


async def _start_streaming_turn(monkeypatch, saved, buffer):
    started = asyncio.Event()
    handler = _streaming_handler(monkeypatch, saved, started)
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
    await asyncio.wait_for(started.wait(), timeout=5)
    return handle


async def test_shutdown_cancels_and_persists_an_interrupted_turn(monkeypatch):
    saved: dict = {}
    buffer = _FakeBuffer()
    handle = await _start_streaming_turn(monkeypatch, saved, buffer)

    cancelled = await registry.shutdown(timeout_seconds=5)

    assert cancelled == 1
    assert handle.lifecycle.shutting_down is True
    assert handle.task.done()
    assert saved["assistant"] == "partial"
    assert any(
        e.get("code") == GENERIC_TURN_ERROR_CODE and e.get("error") == INTERRUPTED_TURN_MESSAGE
        for e in saved["events"]
    )
    # Distinct from a user Stop, which closes "cancelled" with no error event.
    assert buffer.closed == "interrupted"


async def test_shutdown_is_a_noop_with_nothing_running():
    assert await registry.shutdown(timeout_seconds=5) == 0


async def test_shutdown_reports_a_turn_still_unwinding_past_its_budget():
    async def _slow_to_unwind():
        # Swallows the first cancel (the shutdown's), then keeps running for
        # real past the budget below — a step that would not yield in time.
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(1)
        await asyncio.sleep(0.3)

    task = asyncio.create_task(_slow_to_unwind())
    await asyncio.sleep(0)
    lifecycle = TurnLifecycle()
    handle = RunHandle(
        conversation_id=CID, turn_id=0, buffer=_FakeBuffer(), task=task, lifecycle=lifecycle,
    )
    registry._by_cid[CID] = handle

    cancelled = await registry.shutdown(timeout_seconds=0.05)

    assert cancelled == 1
    assert lifecycle.shutting_down is True
    assert not task.done()
    await asyncio.wait_for(task, timeout=2)  # let it finish so nothing leaks past the test
