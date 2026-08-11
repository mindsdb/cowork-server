"""Stopping a turn while a question is on screen must still retire it.

anton's `elicit()` emits `StreamAskUserAnswered` from an `except Exception`
branch, so a `CancelledError` (Stop) skips it — and emitting it from anton at
that point would be futile anyway, since nothing drains the session queue once
the turn is cancelled. The server therefore synthesizes the retirement before
persisting, so the stored event log never holds a published `response.ask_user`
that nothing in it closes.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import cowork.handlers.responses as responses_mod
from cowork.handlers.responses import (
    ResponsesHandler,
    cancelled_ask_user_retirements,
)


def _ask(question_id: str) -> dict:
    return {"type": "response.ask_user", "question_id": question_id, "prompt": "?"}


def _answered(question_id: str, status: str = "answered") -> dict:
    return {
        "type": "response.ask_user_answered",
        "question_id": question_id,
        "status": status,
        "values": [],
        "text": "",
    }


def test_unanswered_question_gets_a_cancelled_retirement():
    [event] = cancelled_ask_user_retirements([_ask("ask:1")])
    assert event == {
        "type": "response.ask_user_answered",
        "question_id": "ask:1",
        "status": "cancelled",
        "values": [],
        "text": "",
    }


def test_answered_question_is_left_alone():
    events = [_ask("ask:1"), _answered("ask:1"), _ask("ask:2")]
    assert [e["question_id"] for e in cancelled_ask_user_retirements(events)] == ["ask:2"]


def test_one_retirement_per_question_id():
    # A duplicated publish of the same id must not produce two retirements —
    # the client matches on question_id, so the second would retire nothing.
    events = [_ask("ask:1"), _ask("ask:1")]
    assert len(cancelled_ask_user_retirements(events)) == 1


def test_no_questions_no_events():
    assert cancelled_ask_user_retirements([{"type": "response.output_text.delta"}]) == []


# ── the same guard, driven through the real producer ────────────────────


class _FakeBuffer:
    def __init__(self) -> None:
        self.closed: str | None = None

    async def append(self, type_, data):
        return 1

    async def close(self, reason, extra=None):
        self.closed = reason


def _cancellable_handler(monkeypatch, saved, published: asyncio.Event):
    """A handler whose turn publishes one ask_user and then blocks forever."""
    handler = object.__new__(ResponsesHandler)
    handler.principal = object()

    class FakeConversationService:
        def __init__(self, session):
            pass

        def get_conversation(self, conv_id):
            return object()

        def save_user_message(self, conv_id, content, *, created_at=None, pending=False):
            msg = SimpleNamespace(id=uuid4())
            saved["user_id"] = msg.id
            return msg

        def finalize_pending(self, conv_id, message_id=None):
            saved["finalized"] = True

        def save_assistant_turn(self, conv_id, text, events, harness=None, tool_rows=None):
            saved["events"] = events

    class FakeSession:
        def close(self):
            pass

    async def formatter(stream, model, event_sink):
        event_sink("response.ask_user", _ask("ask:1"))
        yield "event: response.ask_user\ndata: {}\n\n"
        published.set()
        await asyncio.sleep(3600)

    fake_harness = SimpleNamespace(
        stream_response=lambda **kwargs: None,
        formatter=formatter,
    )

    monkeypatch.setattr(responses_mod, "ConversationService", FakeConversationService)
    monkeypatch.setattr(responses_mod, "ScopedSession", lambda s, scope: FakeSession())
    monkeypatch.setattr(responses_mod, "get_open_session", lambda: None)
    monkeypatch.setattr(responses_mod, "scope_from_principal", lambda p: None)
    monkeypatch.setattr(responses_mod, "get_harness", lambda name: fake_harness)
    return handler


async def test_stopping_a_turn_persists_a_retirement_for_the_open_question(monkeypatch):
    saved: dict = {}
    published = asyncio.Event()
    handler = _cancellable_handler(monkeypatch, saved, published)
    buffer = _FakeBuffer()

    task = asyncio.create_task(handler._run_turn(
        conv_id=uuid4(), harness_input=[], original_content="hi", model="anton",
        disabled=None, harness_name="anton", harness_id="anton", buffer=buffer,
    ))
    await asyncio.wait_for(published.wait(), timeout=5)
    task.cancel()
    await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=5)

    assert buffer.closed == "cancelled"
    events = saved["events"]
    assert events[0]["type"] == "response.ask_user"
    # The persisted log is self-consistent: the question it published is retired.
    assert events[-1] == {
        "type": "response.ask_user_answered",
        "question_id": "ask:1",
        "status": "cancelled",
        "values": [],
        "text": "",
    }
