"""The shared answer-text accumulation rule, and the merge of adjacent text
deltas a turn's stored events go through."""

from __future__ import annotations

import copy
import random
from types import SimpleNamespace

import pytest

from cowork.harnesses.anton_harness.scratchpad_cell_replay import extract_scratchpad_cells
from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream
from cowork.streaming.answer_text import accumulate_answer_text, coalesce_text_deltas


def _feed(events) -> str:
    collected: list[str] = []
    for event_type, data in events:
        accumulate_answer_text(collected, event_type, data)
    return "".join(collected)


def test_deltas_accumulate_in_order():
    assert _feed([
        ("response.output_text.delta", {"delta": "Hello"}),
        ("response.output_text.delta", {"delta": " there."}),
    ]) == "Hello there."


def test_a_reset_drops_the_answer_it_supersedes():
    assert _feed([
        ("response.output_text.delta", {"delta": "SUPERSEDED"}),
        ("response.answer_reset", {"item_id": "msg-1"}),
        ("response.output_text.delta", {"delta": "REPLACEMENT"}),
    ]) == "REPLACEMENT"


def test_repeated_resets_each_drop_only_the_previous_attempt():
    """The verifier can force several continuations in one turn."""
    assert _feed([
        ("response.output_text.delta", {"delta": "FIRST"}),
        ("response.answer_reset", {}),
        ("response.output_text.delta", {"delta": "SECOND"}),
        ("response.answer_reset", {}),
        ("response.output_text.delta", {"delta": "THIRD"}),
    ]) == "THIRD"


def test_other_events_do_not_touch_the_answer():
    assert _feed([
        ("response.output_text.delta", {"delta": "kept"}),
        ("response.in_progress", {"phase": "continuation", "message": "..."}),
        ("response.completed", {"response": {}}),
    ]) == "kept"


def test_a_delta_without_text_contributes_nothing():
    assert _feed([("response.output_text.delta", {})]) == ""


def test_a_restore_returns_the_answer_a_reset_set_aside():
    """A hand-back means the continuation never delivered its replacement."""
    assert _feed([
        ("response.output_text.delta", {"delta": "THE ANSWER THE USER READ"}),
        ("response.answer_reset", {"item_id": "msg-1"}),
        ("response.output_text.delta", {"delta": "Checking."}),
        ("response.answer_restore", {"text": "THE ANSWER THE USER READ"}),
        ("response.output_text.delta", {"delta": " Giving up."}),
    ]) == "THE ANSWER THE USER READChecking. Giving up."


def test_a_restore_without_text_is_inert():
    assert _feed([
        ("response.output_text.delta", {"delta": "kept"}),
        ("response.answer_restore", {}),
    ]) == "kept"


_DELTA = "response.output_text.delta"


def _delta(text: str, seq: int, *, item_id: str = "msg-1", **extra) -> dict:
    return {"type": _DELTA, "sequence_number": seq, "item_id": item_id, "delta": text, "at_ms": 1000 + seq, **extra}


def _answer(events: list[dict]) -> str:
    return _feed([(event["type"], event) for event in events])


def test_adjacent_deltas_merge_into_the_first_one():
    events = [_delta("Hel", 1), _delta("lo", 2), _delta(" there", 3)]

    assert coalesce_text_deltas(events) == [
        {"type": _DELTA, "sequence_number": 1, "item_id": "msg-1", "delta": "Hello there", "at_ms": 1001},
    ]


def test_other_events_split_runs_and_pass_through_unchanged():
    created = {"type": "response.created", "sequence_number": 0}
    thought = {"type": "response.in_progress", "sequence_number": 3, "thought_role": "thought.progress"}
    reset = {"type": "response.answer_reset", "sequence_number": 6, "item_id": "msg-1"}
    restore = {"type": "response.answer_restore", "sequence_number": 8, "item_id": "msg-1", "text": "a"}
    completed = {"type": "response.completed", "sequence_number": 10}
    events = [
        created, _delta("a", 1), _delta("b", 2), thought, _delta("c", 4), _delta("d", 5),
        reset, _delta("e", 7), restore, _delta("f", 9), completed,
    ]

    stored = coalesce_text_deltas(events)

    assert [e.get("delta") for e in stored if e["type"] == _DELTA] == ["ab", "cd", "e", "f"]
    others = [e for e in stored if e["type"] != _DELTA]
    assert others == [created, thought, reset, restore, completed]
    assert all(a is b for a, b in zip(others, [created, thought, reset, restore, completed], strict=True))


@pytest.mark.parametrize("events", [
    [_delta("SUPERSEDED", 1), {"type": "response.answer_reset", "item_id": "msg-1"}, _delta("REPLACE", 2),
     _delta("MENT", 3)],
    [_delta("FIRST", 1), {"type": "response.answer_reset"}, _delta("SEC", 2), _delta("OND", 3),
     {"type": "response.answer_reset"}, _delta("THIRD", 4)],
    [_delta("THE ANSWER", 1), _delta(" THE USER READ", 2), {"type": "response.answer_reset", "item_id": "msg-1"},
     _delta("Checking.", 3), {"type": "response.answer_restore", "text": "THE ANSWER THE USER READ"},
     _delta(" Giving", 4), _delta(" up.", 5)],
    [_delta("kept", 1), {"type": "response.answer_restore"}, _delta("", 2), _delta(" too", 3)],
])
def test_the_answer_text_survives_the_merge(events):
    assert _answer(coalesce_text_deltas(events)) == _answer(events)


def test_a_delta_with_other_keys_or_another_item_is_not_merged():
    routed = _delta("direct", 2, response_route="direct_context", response_route_reason="r")
    events = [_delta("a", 1), routed, _delta("b", 3), _delta("c", 4, item_id="msg-2"), _delta("d", 5, item_id="msg-2")]

    stored = coalesce_text_deltas(events)

    assert [(e["item_id"], e["delta"]) for e in stored] == [
        ("msg-1", "a"), ("msg-1", "direct"), ("msg-1", "b"), ("msg-2", "cd"),
    ]
    assert stored[1] is routed


def test_a_delta_whose_text_is_not_a_string_is_not_merged():
    odd = {"type": _DELTA, "delta": None}

    assert coalesce_text_deltas([_delta("a", 1), odd, _delta("b", 2)]) == [_delta("a", 1), odd, _delta("b", 2)]


def test_the_input_is_left_as_it_was():
    events = [_delta("a", 1), _delta("b", 2), {"type": "response.completed"}]
    before = copy.deepcopy(events)

    coalesce_text_deltas(events)

    assert events == before


def test_a_single_delta_comes_back_as_the_same_object():
    only = _delta("a", 1)

    stored = coalesce_text_deltas([only])

    assert stored == [only] and stored[0] is only


def test_no_events_store_no_events():
    assert coalesce_text_deltas([]) == []


def _random_stream(rng: random.Random) -> list:
    """A turn's anton stream events, as a model and the agent loop emit them."""
    from anton.core.llm.provider import (
        StreamComplete,
        StreamReasoningDelta,
        StreamTaskProgress,
        StreamTextDelta,
        StreamToolResult,
        StreamToolUseDelta,
        StreamToolUseEnd,
        StreamToolUseStart,
    )

    events: list = []
    for step in range(rng.randrange(1, 30)):
        kind = rng.random()
        if kind < 0.55:
            events.append(StreamTextDelta(text=rng.choice(["", " ", "\n", "\n\n", "word", " more", "x" * 16])))
        elif kind < 0.65:
            events.append(StreamReasoningDelta(text="thinking"))
        elif kind < 0.75:
            tool_id = f"t{step}"
            name = rng.choice(["scratchpad", "memorize", "web_search"])
            action = rng.choice(["exec", "view", "reset"])
            events += [
                StreamToolUseStart(id=tool_id, name=name),
                StreamToolUseDelta(id=tool_id, json_delta=f'{{"action": "{action}", "code": "x = {step}"}}'),
                StreamToolUseEnd(id=tool_id),
                StreamToolResult(name=name, content=f'{{"code": "x = {step}", "stdout": "", "stderr": "", '
                                                    f'"error": null}}', action=action, id=tool_id),
            ]
        elif kind < 0.85:
            events.append(StreamTaskProgress(phase=rng.choice(["continuation", "handback", "scratchpad_start"]),
                                             message="m"))
        else:
            stop = rng.choice(["end_turn", "tool_use", "max_tokens"])
            events.append(StreamComplete(response=SimpleNamespace(stop_reason=stop, tool_calls=[])))
    return events


async def test_merging_a_real_formatters_events_keeps_what_every_reader_rebuilds():
    """The events come from format_responses_stream, collected the way the
    handlers' event_sink collects them, so the merge is checked against the
    shapes a turn really stores."""
    rng = random.Random(3362)
    merged = 0
    for _ in range(300):
        items = _random_stream(rng)
        events: list[dict] = []

        async def stream():
            for item in items:
                yield item

        async for _frame in format_responses_stream(stream(), "m", lambda _type, data: events.append(data)):
            pass
        before = copy.deepcopy(events)

        stored = coalesce_text_deltas(events)

        assert events == before
        assert _answer(stored) == _answer(events)
        assert [e for e in stored if e["type"] != _DELTA] == [e for e in events if e["type"] != _DELTA]
        assert extract_scratchpad_cells(stored) == extract_scratchpad_cells(events)
        assert not any(
            a["type"] == b["type"] == _DELTA and a.get("item_id") == b.get("item_id")
            for a, b in zip(stored, stored[1:])
        )
        merged += len(events) - len(stored)
    assert merged > 0
