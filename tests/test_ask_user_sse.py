"""The two new events on the SSE wire.

Both must go through event_sink as well as the wire: the sink is what the
client replays when a conversation is reopened, so a card that skips it
silently disappears on reload.
"""

from __future__ import annotations

import json

from anton.core.interaction.elicit import AskAnswer, AskOption, AskRequest
from anton.core.llm.provider import StreamAskUser, StreamAskUserAnswered
from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream

_REQUEST = AskRequest(
    prompt="Which database?",
    timeout_s=300,
    select="one",
    allow_custom=True,
    options=(
        AskOption(value="pg", label="postgres", detail="primary"),
        AskOption(value="my", label="mysql"),
    ),
)


async def _events(*stream_events):
    async def _gen():
        for event in stream_events:
            yield event

    sink_calls: list[tuple[str, dict]] = []
    frames: list[str] = []
    async for frame in format_responses_stream(
        _gen(), model="test-model", event_sink=lambda t, d: sink_calls.append((t, d))
    ):
        frames.append(frame)
    return frames, sink_calls


def _payload(frames, event_name):
    for frame in frames:
        if f"event: {event_name}" in frame:
            data_line = next(
                line for line in frame.split("\n") if line.startswith("data:")
            )
            return json.loads(data_line[len("data:"):].strip())
    raise AssertionError(f"{event_name} not on the wire: {frames}")


async def test_ask_user_carries_a_self_contained_payload():
    frames, _ = await _events(StreamAskUser(id="ask:1", request=_REQUEST))
    data = _payload(frames, "response.ask_user")
    assert data["type"] == "response.ask_user"
    assert data["question_id"] == "ask:1"
    assert data["prompt"] == "Which database?"
    assert data["select"] == "one"
    assert data["allow_custom"] is True
    assert data["timeout_s"] == 300
    assert data["options"] == [
        {"value": "pg", "label": "postgres", "detail": "primary"},
        {"value": "my", "label": "mysql", "detail": ""},
    ]


async def test_ask_user_answered_carries_the_outcome():
    frames, _ = await _events(
        StreamAskUserAnswered(
            id="ask:1", answer=AskAnswer(status="answered", values=("pg",), text="")
        )
    )
    data = _payload(frames, "response.ask_user_answered")
    assert data["question_id"] == "ask:1"
    assert data["status"] == "answered"
    assert data["values"] == ["pg"]
    assert data["text"] == ""


async def test_both_events_reach_the_event_sink_for_replay():
    _, sink_calls = await _events(
        StreamAskUser(id="ask:1", request=_REQUEST),
        StreamAskUserAnswered(id="ask:1", answer=AskAnswer(status="cancelled")),
    )
    names = [name for name, _ in sink_calls]
    assert "response.ask_user" in names
    assert "response.ask_user_answered" in names
