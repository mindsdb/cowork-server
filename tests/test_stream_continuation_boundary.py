"""The formatter's continuation boundary: replace the answer, don't append.

anton's completion verifier can judge a turn incomplete and force a
continuation, which streams a fresh answer into the same output item as the one
it supersedes — so the client reads the answer, then reads it again. The
formatter turns that boundary into one `response.output_text.reset`, but only
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
)

from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream

_BOUNDARY = StreamTaskProgress(
    phase="continuation", message="Task incomplete — continuing (1/3)..."
)


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
    assert types.count("response.output_text.reset") == 1, (
        f"expected exactly one reset; event types were {types}"
    )
    reset = types.index("response.output_text.reset")
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
    assert [p["type"] for p in payloads].count("response.output_text.reset") == 0


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
    assert [p["type"] for p in payloads].count("response.output_text.reset") == 1
