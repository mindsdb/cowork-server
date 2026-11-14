"""The two new events on the SSE wire.

Both must go through event_sink as well as the wire: the sink is what the
client replays when a conversation is reopened, so a card that skips it
silently disappears on reload.
"""

from __future__ import annotations

import itertools
import json
import time
from types import SimpleNamespace
from uuid import uuid4

from anton.core.interaction.elicit import AskAnswer, AskOption, AskRequest
from anton.core.llm.provider import StreamAskUser, StreamAskUserAnswered
from cowork.common import event_clock
from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream
from cowork.turnqueue.producer import step_stream_events

from test_turn_form_blocks import _FakeBuffer, _ui_handler

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


async def test_ask_user_carries_when_the_server_held_it_live_and_on_replay(monkeypatch):
    """The app approves a held browser step for the person only when the
    card's `at_ms` comes after the approvals-off stamp, live and after a
    reload. So the server stamps it, and the formatter hands the event sink
    the record with the frame's own time. This checks only that handoff: the
    handler sinks that save the record are pinned by
    test_an_in_process_turn_saves_the_ask_user_time_its_live_frame_carried
    below and by test_produce_remote_saves_the_ask_user_time_its_live_frame_carried
    in test_responses_remote_backend.py. The clock ticks a second on every
    read, so a second stamp can't land on the same value."""
    seconds = itertools.count(1_000)
    monkeypatch.setattr(time, "time", lambda: float(next(seconds)))
    before = int(time.time() * 1000)
    frames, sink_calls = await _events(StreamAskUser(id="browser:approve:t-a:pa-1", request=_REQUEST))
    after = int(time.time() * 1000)
    data = _payload(frames, "response.ask_user")
    assert before <= data["at_ms"] <= after
    replayed = next(record for name, record in sink_calls if name == "response.ask_user")
    assert replayed["at_ms"] == data["at_ms"]


async def test_an_in_process_turn_saves_the_ask_user_time_its_live_frame_carried(monkeypatch):
    """A reload replays the record the turn's sink saved, so that record keeps
    the live frame's `at_ms`: a later one would approve, after a reload, a card
    held before approvals went off. The clock ticks a second on every read, so
    a second stamp can't land on the same value."""
    seconds = itertools.count(1_000)
    monkeypatch.setattr(event_clock, "time", SimpleNamespace(time=lambda: float(next(seconds))))
    saved: dict = {}
    buffer = _FakeBuffer()
    handler = _ui_handler(monkeypatch, saved, [
        StreamAskUser(id="browser:approve:t-a:pa-1", request=_REQUEST),
        StreamAskUserAnswered(
            id="browser:approve:t-a:pa-1", answer=AskAnswer(status="answered", values=("pg",), text=""),
        ),
    ])

    await handler._run_turn(
        conv_id=uuid4(), harness_input=[], original_content="hi", model="anton",
        disabled=None, harness_name="anton", harness_id="anton", buffer=buffer,
    )

    assert buffer.closed == "completed"
    live = _payload([record["sse"] for _, record in buffer.records], "response.ask_user")
    stored = next(e for e in saved["events"] if e.get("type") == "response.ask_user")
    assert stored["at_ms"] == live["at_ms"]


async def test_a_pods_ask_user_carries_the_servers_time_not_its_own():
    """A remote turn's question comes from the pod as a `turn_step`. Its card
    is stamped when this server formats it, whatever `at_ms` the pod put in
    the step."""
    before = int(time.time() * 1000)
    frames, _ = await _events(*step_stream_events({"step": "ask_user", "id": "q1", "prompt": "x", "at_ms": 1}))
    after = int(time.time() * 1000)
    assert before <= _payload(frames, "response.ask_user")["at_ms"] <= after
