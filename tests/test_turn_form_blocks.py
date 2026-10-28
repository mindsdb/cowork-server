"""A UI turn saves the connection form its answer left out; other callers don't.

Drives the real handler paths with the real formatter. The stream summarizes
instead of pasting the form `request_credentials` returned, and ends with the
turn's history rows the way the harness yields them. Mirrors
test_turn_shutdown_lifecycle.py's fixtures.
"""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from anton.core.llm.provider import StreamTextDelta

import cowork.handlers.responses as responses_mod
from cowork.handlers.responses import ResponsesHandler
from cowork.harnesses.anton_harness.harness import _captured_turn_history
from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream
from cowork.harnesses.anton_harness.tools import _cowork_request_credentials
from cowork.streaming.registry import TurnLifecycle, registry

from _fakes import inline_run_db, opens

_SUMMARY = "The connection form is ready in the side panel."


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

    @property
    def is_closed(self) -> bool:
        return self.closed is not None


async def _summarizing_turn():
    """The events of a turn whose answer summarizes the form it was handed,
    and the form block the tool returned."""
    result = await _cowork_request_credentials(
        session=None,
        tc_input={"engine": "postgres", "_connector_id": "postgres", "title": "Connect to Postgres"},
    )
    turn = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "call_0", "name": "request_credentials", "input": {}},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_0", "content": result}]},
        {"role": "assistant", "content": _SUMMARY},
    ]
    history = _captured_turn_history(
        session=SimpleNamespace(turn_messages=lambda: turn), seed_len=None, compacted=False,
    )
    block = result[result.index("```data-vault-form"):result.rindex("```") + 3]
    return [StreamTextDelta(text=_SUMMARY), history], block


async def _events(events):
    for event in events:
        yield event


def _ui_handler(monkeypatch, saved: dict, events: list):
    """A streaming handler whose harness yields ``events`` through the real
    formatter, recording what it saves."""
    handler = object.__new__(ResponsesHandler)
    handler.principal = object()

    class FakeConversationService:
        def __init__(self, session):
            pass

        def get_conversation(self, conv_id):
            return SimpleNamespace(project=None)

        def save_user_message(self, conv_id, content, *, created_at=None, pending=False):
            return SimpleNamespace(id=uuid4())

        def clear_pending(self, conv_id, *, message_id=None):
            return True

        def save_assistant_turn(self, conv_id, text, events, harness=None, tool_rows=None):
            saved["assistant"] = text
            return SimpleNamespace(id=uuid4())

    class FakeSession:
        def close(self):
            pass

    monkeypatch.setattr(responses_mod, "ConversationService", FakeConversationService)
    monkeypatch.setattr(responses_mod, "run_db", inline_run_db(FakeSession()))
    monkeypatch.setattr(responses_mod, "scope_from_principal", lambda p: None)
    monkeypatch.setattr(responses_mod, "get_harness", lambda name: SimpleNamespace(
        stream_response=lambda **kwargs: _events(events),
        formatter=format_responses_stream,
    ))
    return handler


async def test_a_ui_turn_saves_the_form_its_answer_left_out(monkeypatch):
    events, block = await _summarizing_turn()
    saved: dict = {}
    buffer = _FakeBuffer()
    handler = _ui_handler(monkeypatch, saved, events)
    lifecycle = TurnLifecycle()

    handle = await registry.start(
        conversation_id="conv-form-blocks", turn_id=0, open_buffer=opens(buffer),
        produce=lambda buffer: handler._run_turn(
            conv_id=uuid4(), harness_input=[], original_content="connect postgres", model="anton",
            disabled=None, harness_name="anton", harness_id="anton", buffer=buffer,
            lifecycle=lifecycle,
        ),
        lifecycle=lifecycle,
    )
    await asyncio.wait_for(handle.task, timeout=10)

    assert buffer.closed == "completed"
    assert saved["assistant"] == f"{_SUMMARY}\n\n{block}\n"


def test_a_non_streaming_reply_gets_no_form():
    events, block = asyncio.run(_summarizing_turn())
    handler = object.__new__(ResponsesHandler)
    handler.principal = None
    handler.harness = SimpleNamespace(id="anton", formatter=format_responses_stream)
    handler.scope = None

    @contextlib.asynccontextmanager
    async def _no_lock(_conversation_id):
        yield

    with patch("cowork.handlers.responses.run_db", AsyncMock(return_value=uuid4())), \
         patch("cowork.handlers.responses.conversation_writes", _no_lock):
        response = asyncio.run(handler._collect(
            stream=_events(events), conversation_id=uuid4(), model="m", original_content="hi",
        ))

    assert response.output[0].content[0].text == _SUMMARY
    assert block not in response.output[0].content[0].text
