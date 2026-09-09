"""The formatter's continuation boundary: replace the answer, don't append.

anton's completion verifier can judge a turn incomplete and force a
continuation, which streams a fresh answer into the same output item as the one
it supersedes — so the client reads the answer, then reads it again. The
formatter turns that boundary into one `response.answer_reset`, but only
once replacement text actually exists: a continuation that never speaks must
leave the answer the user already read intact.
"""

from __future__ import annotations

import json

from anton.core.llm.provider import (
    LLMResponse,
    StreamComplete,
    StreamTaskProgress,
    StreamTextDelta,
    StreamToolUseEnd,
    StreamToolUseStart,
    ToolCall,
)

from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream

_BOUNDARY = StreamTaskProgress(
    phase="continuation", message="Task incomplete — continuing (1/3)..."
)
#: anton gives up and explains instead of continuing, so nothing supersedes.
_HANDBACK = StreamTaskProgress(phase="handback", message="")


async def _drain(events):
    """Run the formatter over `events`; return (ordered payloads, completed text)."""

    async def _gen():
        for e in events:
            yield e

    payloads: list[dict] = []
    completed = ""
    async for sse in format_responses_stream(_gen(), model="m"):
        payload = json.loads(sse.split("data: ", 1)[1])
        payloads.append(payload)
        if payload["type"] == "response.completed":
            completed = payload["response"]["output"][0]["content"][0]["text"]
    return payloads, completed


def _round_end(stop_reason="end_turn"):
    return StreamComplete(response=LLMResponse(
        content="", tool_calls=[], stop_reason=stop_reason,
    ))


async def test_the_replacement_supersedes_the_answer_it_replaces():
    payloads, completed = await _drain([
        StreamTextDelta(text="SUPERSEDED"),
        _round_end(),
        _BOUNDARY,
        StreamTextDelta(text="REPLACEMENT"),
        _round_end(),
    ])
    # Exactly "REPLACEMENT" also pins that the round end before the boundary
    # does not leave its paragraph break armed: that break belongs between two
    # rounds of the answer being discarded, not in front of its replacement.
    assert completed == "REPLACEMENT"

    types = [p["type"] for p in payloads]
    assert types.count("response.answer_reset") == 1, (
        f"expected exactly one reset; event types were {types}"
    )
    reset = types.index("response.answer_reset")
    # Immediately before the replacement text, so the bubble is never empty.
    assert types[reset + 1] == "response.output_text.delta"
    assert payloads[reset + 1]["delta"] == "REPLACEMENT"
    # Names the item it resets, like every delta does.
    assert payloads[reset]["item_id"] == payloads[reset + 1]["item_id"]


async def test_a_boundary_with_no_replacement_changes_nothing():
    """Stop pressed, a failed turn, or a continuation that only calls tools all
    end the stream with no replacement text. Clearing at the boundary itself
    would leave an empty answer where the user had already read one."""
    payloads, completed = await _drain([
        StreamTextDelta(text="THE ANSWER THE USER READ"),
        _round_end(),
        _BOUNDARY,
    ])
    assert completed == "THE ANSWER THE USER READ"
    assert [p["type"] for p in payloads].count("response.answer_reset") == 0


async def test_a_throttled_progress_notice_cannot_suppress_the_boundary():
    """Progress notices are rate-limited, and the boundary arrives right behind
    one. Its own notice may be throttled away; the reset may not."""
    payloads, completed = await _drain([
        StreamTextDelta(text="SUPERSEDED"),
        _round_end(),
        StreamTaskProgress(phase="analyzing", message="Analyzing results..."),
        _BOUNDARY,
        StreamTextDelta(text="REPLACEMENT"),
        _round_end(),
    ])
    assert completed == "REPLACEMENT"
    assert [p["type"] for p in payloads].count("response.answer_reset") == 1


async def test_a_handback_after_the_boundary_does_not_replace_the_answer():
    """A continuation can run its rounds, produce no answer of its own, and hand
    back instead: budget gone, round cap hit, verifier stuck. The diagnosis it
    streams then is an additional message, not a replacement — dropping the
    answer for it loses the only real content the turn produced, from the bubble
    and from the persisted message the next turn's history is rebuilt from.
    """
    payloads, completed = await _drain([
        StreamTextDelta(text="THE ANSWER THE USER READ"),
        _round_end(),
        _BOUNDARY,
        # The continuation's rounds only call tools, so it never speaks.
        StreamComplete(response=LLMResponse(
            content="", tool_calls=[ToolCall(id="t1", name="scratchpad", input={})],
            stop_reason="tool_use",
        )),
        _HANDBACK,
        StreamTextDelta(text="I could not finish; shall I continue?"),
        _round_end(),
    ])
    assert "THE ANSWER THE USER READ" in completed
    assert completed.endswith("I could not finish; shall I continue?")
    assert [p["type"] for p in payloads].count("response.answer_reset") == 0


async def test_whitespace_alone_does_not_spend_the_boundary():
    """`"\\n"` is text enough to satisfy a truthiness check and not enough to be
    an answer. Spending the boundary on it persists a blank message where the
    user had read a real one."""
    payloads, completed = await _drain([
        StreamTextDelta(text="THE ANSWER THE USER READ"),
        _round_end(),
        _BOUNDARY,
        StreamTextDelta(text="\n"),
        _round_end(),
    ])
    assert "THE ANSWER THE USER READ" in completed
    assert [p["type"] for p in payloads].count("response.answer_reset") == 0


async def test_a_handback_gives_back_the_answer_the_boundary_set_aside():
    """The common shape, and the one a cancel alone cannot cover.

    A continuation narrates before its first tool call, so that narration spends
    the boundary — and if the turn then hands back instead of answering, the
    answer the user had already read is gone from the transcript and from the
    history the next turn is rebuilt from. The hand-back has to give it back.
    """
    payloads, completed = await _drain([
        StreamTextDelta(text="THE ANSWER THE USER READ"),
        _round_end(),
        _BOUNDARY,
        StreamTextDelta(text="Let me double-check the file."),
        StreamToolUseStart(id="t1", name="scratchpad"),
        StreamToolUseEnd(id="t1"),
        _round_end(stop_reason="tool_use"),
        _HANDBACK,
        StreamTextDelta(text="I ran out of budget; shall I continue?"),
        _round_end(),
    ])
    assert "THE ANSWER THE USER READ" in completed
    assert completed.endswith("I ran out of budget; shall I continue?")
    types = [p["type"] for p in payloads]
    assert types.count("response.answer_reset") == 1
    assert types.count("response.answer_restore") == 1
    restore = next(p for p in payloads if p["type"] == "response.answer_restore")
    assert restore["text"] == "THE ANSWER THE USER READ"
    # Ordered so a client replaying the stream lands in the same place.
    assert types.index("response.answer_reset") < types.index("response.answer_restore")


async def test_a_delivered_replacement_is_never_given_back():
    """The restore is for a boundary the turn failed to honour. A continuation
    that answers keeps its replacement, and nothing returns."""
    payloads, completed = await _drain([
        StreamTextDelta(text="SUPERSEDED"),
        _round_end(),
        _BOUNDARY,
        StreamTextDelta(text="Checking."),
        StreamToolUseStart(id="t1", name="scratchpad"),
        StreamToolUseEnd(id="t1"),
        _round_end(stop_reason="tool_use"),
        StreamTextDelta(text="REPLACEMENT"),
        _round_end(),
    ])
    assert "SUPERSEDED" not in completed
    assert completed.endswith("REPLACEMENT")
    assert [p["type"] for p in payloads].count("response.answer_restore") == 0


async def test_a_second_boundary_sets_aside_the_answer_it_supersedes():
    """The stash tracks the current answer, not the turn's first one, so a
    hand-back after two continuations returns the second attempt."""
    payloads, _ = await _drain([
        StreamTextDelta(text="FIRST"),
        _round_end(),
        _BOUNDARY,
        StreamTextDelta(text="SECOND"),
        _round_end(),
        _BOUNDARY,
        StreamTextDelta(text="THIRD"),
        _round_end(),
        _HANDBACK,
        StreamTextDelta(text="giving up"),
        _round_end(),
    ])
    restore = next(p for p in payloads if p["type"] == "response.answer_restore")
    assert restore["text"] == "SECOND"
