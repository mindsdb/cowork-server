"""Zero progress-event loss when a client disconnects mid-turn (T2).

The SSE pipeline must enforce write-ahead ordering: each streaming event's
message_events row is persisted AND committed before that event's SSE
string is yielded to the client. A user who closes the laptop mid-turn and
reopens later must find every event emitted so far durably retrievable, in
order, through the same code path /responses/tail replays from
(ConversationService.latest_assistant_message + get_turn_events).

Pre-fix behavior (regression this file pins down): ResponsesHandler._stream
buffered every event in memory and only persisted them via
save_assistant_turn AFTER the formatter loop completed, so closing the
generator mid-turn (what Starlette does on client disconnect) lost the
entire turn — no assistant row, no events, empty tail replay.

Async cases run via ``asyncio.run`` inside sync tests, matching the rest of
the suite (no pytest-asyncio dependency).
"""
from __future__ import annotations

import asyncio
import json

import pytest
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.handlers.responses import ResponsesHandler
from cowork.services.conversations import ConversationService
from cowork.services.projects import GENERAL_PROJECT_ID

N_EVENTS = 8
CONSUME = 4  # "client" disconnects after receiving this many SSE strings


class StubHarness:
    """Stands in for a real harness — emits N deterministic events through
    the real formatter contract: event_sink(type, data) is called first,
    then the corresponding SSE string is yielded (same order as the anton
    and hermes stream formatters)."""

    id = "stub"

    async def formatter(self, stream, model, event_sink):
        for i in range(N_EVENTS):
            data = {
                "type": "response.output_text.delta",
                "sequence_number": i + 1,
                "delta": f"step-{i} ",
            }
            event_sink("response.output_text.delta", data)
            yield f"event: response.output_text.delta\ndata: {json.dumps(data)}\n\n"


@pytest.fixture()
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


def _handler(session) -> ResponsesHandler:
    # Bypass __init__ (it resolves the user's default harness from user
    # settings); the streaming path only needs .session and .harness.
    handler = ResponsesHandler.__new__(ResponsesHandler)
    handler.session = session
    handler.harness = StubHarness()
    return handler


def _fresh_replay(conversation_id):
    """Read back through tail's replay path in a FRESH session so only
    durably committed rows count."""
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as fresh:
        svc = ConversationService(fresh)
        message = svc.latest_assistant_message(conversation_id)
        if message is None:
            return None, []
        return message, svc.get_turn_events(message.id)


def test_disconnect_mid_turn_loses_no_emitted_events(session):
    svc = ConversationService(session)
    conv = svc.create_conversation(topic="t", project_id=GENERAL_PROJECT_ID)
    handler = _handler(session)

    async def scenario():
        gen = handler._stream(None, conv.id, "stub-model")
        received = []
        async for chunk in gen:
            received.append(chunk)
            if len(received) >= CONSUME:
                break  # client stops reading…
        await gen.aclose()  # …and Starlette closes the generator
        return received

    received = asyncio.run(scenario())
    assert len(received) == CONSUME

    message, events = _fresh_replay(conv.id)
    assert message is not None, (
        "assistant turn row must exist mid-turn — events emitted before the "
        "disconnect were lost"
    )
    # Every event whose SSE string reached the client is durable, gapless
    # from 0, and in emission order.
    assert len(events) == CONSUME
    assert [e.sequence_number for e in events] == list(range(CONSUME))
    assert [e.event_data["delta"] for e in events] == [
        f"step-{i} " for i in range(CONSUME)
    ]
    # The partial text collected before the disconnect is stamped on the
    # assistant row (finalized on generator close).
    assert message.content == "".join(f"step-{i} " for i in range(CONSUME))


def test_completed_turn_replays_gapless_and_supports_from_seq(session):
    svc = ConversationService(session)
    conv = svc.create_conversation(topic="t", project_id=GENERAL_PROJECT_ID)
    handler = _handler(session)

    async def scenario():
        return [chunk async for chunk in handler._stream(None, conv.id, "stub-model")]

    received = asyncio.run(scenario())
    assert len(received) == N_EVENTS

    message, events = _fresh_replay(conv.id)
    assert message is not None
    assert [e.sequence_number for e in events] == list(range(N_EVENTS))
    assert message.content == "".join(f"step-{i} " for i in range(N_EVENTS))

    # from_seq resumes gapless mid-turn — a client that saw seq 0..2 asks
    # for from_seq=3 and gets exactly the remainder, in order.
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as fresh:
        tail = ConversationService(fresh).get_turn_events(message.id, from_seq=3)
    assert [e.sequence_number for e in tail] == list(range(3, N_EVENTS))
