"""The shared answer-text accumulation rule."""

from __future__ import annotations

from cowork.streaming.answer_text import accumulate_answer_text


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
