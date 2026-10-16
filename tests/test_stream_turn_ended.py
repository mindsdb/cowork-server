"""A turn that handed back says why on its final response.completed (ENG-3436)."""

from __future__ import annotations

import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from anton.core.llm.provider import StreamTextDelta, StreamTurnEnded
from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream

REASONS = [
    "spend_ceiling",
    "round_cap",
    "handback_budget",
    "handback_verifier_failure",
    "handback_stuck",
    "retry_exhausted",
]


async def _events(*items):
    for item in items:
        yield item


async def _final(*items, event_sink=None) -> dict:
    chunks = [c async for c in format_responses_stream(_events(*items), model="m", event_sink=event_sink)]
    assert chunks[-1].startswith("event: response.completed\n")
    return json.loads(chunks[-1].split("data:", 1)[1])


@pytest.mark.parametrize("reason", REASONS)
async def test_handback_marks_the_final_event_incomplete(reason):
    final = await _final(StreamTextDelta(text="Stopped at a limit."), StreamTurnEnded(ended_by=reason))
    resp = final["response"]
    assert resp["status"] == "incomplete"
    assert resp["incomplete_details"] == {"reason": reason}
    # The message is whole; only the task stopped early.
    assert resp["output"][0]["status"] == "completed"
    assert resp["output"][0]["content"][0]["text"] == "Stopped at a limit."


async def test_finished_turn_carries_no_reason():
    resp = (await _final(StreamTextDelta(text="Hello.")))["response"]
    assert resp["status"] == "completed"
    assert resp["incomplete_details"] is None


async def test_reason_is_persisted_for_reload():
    # Reload replays the events event_sink recorded, so the reason must be there.
    recorded = []
    await _final(
        StreamTextDelta(text="Stopped."), StreamTurnEnded(ended_by="round_cap"),
        event_sink=lambda event_type, data: recorded.append((event_type, data)),
    )
    event_type, data = recorded[-1]
    assert event_type == "response.completed"
    assert data["response"]["incomplete_details"] == {"reason": "round_cap"}


def _collect_handler():
    from cowork.handlers.responses import ResponsesHandler

    handler = object.__new__(ResponsesHandler)
    handler.principal = None

    class _Harness:
        id = "anton"
        formatter = staticmethod(format_responses_stream)

    handler.harness = _Harness()
    handler.scope = None
    return handler


def _run_collect(*items):
    @contextlib.asynccontextmanager
    async def _no_lock(_conversation_id):
        yield

    with patch("cowork.handlers.responses.run_db", AsyncMock(return_value=uuid4())), \
         patch("cowork.handlers.responses.conversation_writes", _no_lock):
        return asyncio.run(_collect_handler()._collect(
            stream=_events(*items), conversation_id=uuid4(), model="m", original_content="hi",
        ))


def test_non_streaming_response_carries_the_reason():
    resp = _run_collect(StreamTextDelta(text="Stopped."), StreamTurnEnded(ended_by="handback_stuck"))
    assert resp.status.value == "incomplete"
    assert resp.incomplete_details.reason == "handback_stuck"
    assert resp.output[0].content[0].text == "Stopped."


def test_non_streaming_finished_response_carries_no_reason():
    resp = _run_collect(StreamTextDelta(text="Hello."))
    assert resp.status.value == "completed"
    assert resp.incomplete_details is None
