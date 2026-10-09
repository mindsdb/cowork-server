"""A streamed answer reaches the client one delta per frame and is stored with
its adjacent text deltas merged into one event row.

Every other handler test fakes the save, so this one drives a turn through the
app and reads the stored events back the way a reload does.
"""
from __future__ import annotations

from uuid import uuid4

import httpx
import pytest

import cowork.handlers.responses as responses_mod
from cowork.handlers.response_routing import DELEGATED_AGENTIC, RouteDecision
from cowork.server import create_app
from cowork.streaming import registry, sse_frame

from _fakes import PausedHarness

_DELTA = "response.output_text.delta"


@pytest.fixture(autouse=True)
def _forget_turns():
    yield
    registry.reset()


class _ThreeDeltas(PausedHarness):
    """Answers at once, in three deltas of one output item."""

    async def formatter(self, stream, model, event_sink):
        yield sse_frame("response.created", {"type": "response.created"})
        for seq, text in enumerate(("Hello", " ", "there"), start=1):
            delta = {"type": _DELTA, "sequence_number": seq, "item_id": "msg-1", "delta": text, "at_ms": 1000 + seq}
            event_sink(_DELTA, delta)
            yield sse_frame(_DELTA, delta)
        yield sse_frame("response.completed", {"type": "response.completed", "response": {"output": []}})


async def test_the_stream_stays_per_delta_and_the_stored_answer_holds_one_delta(monkeypatch):
    monkeypatch.setattr(responses_mod, "get_harness", lambda name: _ThreeDeltas())

    async def decide(**kwargs):
        return RouteDecision(route=DELEGATED_AGENTIC, reason="test")

    monkeypatch.setattr(responses_mod, "decide_route", decide)
    conversation_id = str(uuid4())

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        answered = await client.post(
            "/api/v1/responses/",
            json={"input": "hi", "stream": True, "conversation": conversation_id},
        )
        items = (await client.get(f"/api/v1/conversations/{conversation_id}/items")).json()

    assert answered.status_code == 200, answered.text
    assert [line for line in answered.text.splitlines() if line == f"event: {_DELTA}"] == [f"event: {_DELTA}"] * 3
    (answer,) = [item for item in items if item["role"] == "assistant"]
    assert answer["content"] == "Hello there"
    assert [e for e in answer["events"] if e["type"] == _DELTA] == [
        {"type": _DELTA, "sequence_number": 1, "item_id": "msg-1", "delta": "Hello there", "at_ms": 1001},
    ]
