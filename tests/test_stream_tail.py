"""Stream reconnect buffers + tail/in-flight endpoints (ENG-289).

The Task view recovers a scheduled "Run now" (and any reconnect) by
probing GET /responses/in-flight and tailing GET /responses/tail. Both
were stubs after the cowork-server migration — the client expected the
legacy contract: {in_flight, has_buffer, latest_seq} and an SSE replay
from a sequence number, 404 when the buffer is gone.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import HTTPException

from cowork.services import stream_buffer
from cowork.services.stream_buffer import TurnBuffer


def test_replay_from_seq():
    buf = TurnBuffer("c1")
    for i in range(5):
        buf.append({"type": "e", "i": i})
    buf.finish()

    async def collect(from_seq):
        return [e["i"] async for e in buf.follow(from_seq)]

    assert asyncio.run(collect(0)) == [0, 1, 2, 3, 4]
    assert asyncio.run(collect(3)) == [3, 4]
    assert asyncio.run(collect(99)) == []


def test_follow_receives_live_events():
    async def scenario():
        buf = TurnBuffer("c1")
        buf.append({"i": 0})

        async def producer():
            await asyncio.sleep(0.01)
            buf.append({"i": 1})
            await asyncio.sleep(0.01)
            buf.append({"i": 2})
            buf.finish()

        task = asyncio.create_task(producer())
        seen = [e["i"] async for e in buf.follow(0)]
        await task
        return seen

    assert asyncio.run(scenario()) == [0, 1, 2]


def test_two_concurrent_followers_both_complete():
    async def scenario():
        buf = TurnBuffer("c1")

        async def producer():
            for i in range(4):
                await asyncio.sleep(0.005)
                buf.append({"i": i})
            buf.finish()

        async def follower():
            return [e["i"] async for e in buf.follow(0)]

        results = await asyncio.gather(follower(), follower(), producer())
        return results[0], results[1]

    a, b = asyncio.run(scenario())
    assert a == b == [0, 1, 2, 3]


def test_registry_replaces_and_prunes():
    buf = stream_buffer.begin_turn("conv-x")
    assert stream_buffer.get_buffer("conv-x") is buf
    second = stream_buffer.begin_turn("conv-x")
    assert stream_buffer.get_buffer("conv-x") is second

    second.finish()
    # Age it past the TTL — the next lookup prunes it.
    second.finished_at -= stream_buffer.FINISHED_TTL_SECONDS + 1
    assert stream_buffer.get_buffer("conv-x") is None


def test_in_flight_reports_buffer_state():
    from cowork.api.v1.endpoints.responses import in_flight

    none = asyncio.run(in_flight("missing-conv"))
    assert none == {
        "in_flight": False, "has_buffer": False, "latest_seq": 0,
        "conversation_id": "missing-conv",
    }

    buf = stream_buffer.begin_turn("conv-live")
    buf.append({"type": "response.created"})
    live = asyncio.run(in_flight("conv-live"))
    assert live["in_flight"] is True       # buffer not done counts as live
    assert live["has_buffer"] is True
    assert live["latest_seq"] == 1

    buf.finish()
    finished = asyncio.run(in_flight("conv-live"))
    assert finished["in_flight"] is False  # done + not in active registry
    assert finished["has_buffer"] is True  # still replayable


def test_tail_404_without_buffer():
    from cowork.api.v1.endpoints.responses import tail_response

    with pytest.raises(HTTPException) as err:
        asyncio.run(tail_response("no-such-conv"))
    assert err.value.status_code == 404


def test_tail_replays_buffer_as_sse():
    from cowork.api.v1.endpoints.responses import tail_response

    buf = stream_buffer.begin_turn("conv-sse")
    buf.append({"type": "response.created", "conversation_id": "conv-sse"})
    buf.append({"type": "response.output_text.delta", "delta": "hi"})
    buf.finish()

    async def drain():
        response = await tail_response("conv-sse", from_seq=0)
        return [chunk async for chunk in response.body_iterator]

    frames = asyncio.run(drain())
    assert frames[-1] == "data: [DONE]\n\n"
    payloads = [
        json.loads(frame.split("data:", 1)[1].strip())
        for frame in frames[:-1]
    ]
    assert payloads[0]["type"] == "response.created"
    assert payloads[1]["delta"] == "hi"
    # Same framing POST /responses emits — event: <type> then data:.
    assert frames[0].startswith("event: response.created\n")


def test_handler_event_sink_feeds_buffer():
    """The handler funnel must feed the buffer on both paths (the
    non-streaming one is what scheduled runs use)."""
    from uuid import uuid4
    from cowork.handlers.responses import ResponsesHandler

    handler = object.__new__(ResponsesHandler)  # skip __init__ (no DB/harness)
    conversation_id = uuid4()
    text, events = [], []
    sink, buf = handler._make_event_sink(conversation_id, text, events)

    sink("response.output_text.delta", {"type": "response.output_text.delta", "delta": "a"})
    sink("response.completed", {"type": "response.completed"})

    assert text == ["a"]
    assert len(events) == 2
    assert stream_buffer.get_buffer(str(conversation_id)) is buf
    assert buf.latest_seq == 2
