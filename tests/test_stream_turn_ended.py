"""A turn that handed back says why on its final response.completed (ENG-3436)."""

from __future__ import annotations

import json

import pytest

from anton.core.llm.provider import StreamTextDelta
from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream

# Same guard as production: older anton builds have no StreamTurnEnded.
try:
    from anton.core.llm.provider import StreamTurnEnded
except ImportError:
    StreamTurnEnded = None

needs_turn_ended = pytest.mark.skipif(
    StreamTurnEnded is None,
    reason="anton build predates StreamTurnEnded (ENG-3436)",
)

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


@needs_turn_ended
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


@needs_turn_ended
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


async def test_formatter_survives_anton_without_turn_ended(monkeypatch):
    # The formatter runs on every turn; an unguarded import would break all chat.
    import anton.core.llm.provider as provider_mod

    monkeypatch.delattr(provider_mod, "StreamTurnEnded", raising=False)
    resp = (await _final(StreamTextDelta(text="hello")))["response"]
    assert resp["status"] == "completed"
    assert resp["output"][0]["content"][0]["text"] == "hello"
