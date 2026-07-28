"""Round-boundary paragraphing in the anton SSE formatter (ENG-887).

anton runs several LLM rounds per turn (narrate → run tools → narrate …)
and streams every round's text into the same output item. The formatter
must open a new paragraph when a round ends, or the rounds fuse
mid-sentence in the live stream, the persisted message, and every replay —
EXCEPT after a max_tokens truncation, whose continuation round resumes
mid-sentence.
"""

from __future__ import annotations

import json

from anton.core.llm.provider import (
    LLMResponse,
    StreamComplete,
    StreamTextDelta,
    StreamToolResult,
    StreamToolUseEnd,
    StreamToolUseStart,
    ToolCall,
)

from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream

_TOOL_CALL = ToolCall(id="t1", name="scratchpad", input={})


async def _drain(events):
    """Run the formatter over `events`; return (deltas, completed_text)."""

    async def _gen():
        for e in events:
            yield e

    deltas: list[str] = []
    completed = ""
    async for sse in format_responses_stream(_gen(), model="m"):
        payload = json.loads(sse.split("data: ", 1)[1])
        if payload["type"] == "response.output_text.delta":
            deltas.append(payload["delta"])
        elif payload["type"] == "response.completed":
            completed = payload["response"]["output"][0]["content"][0]["text"]
    return deltas, completed


def _round_end(*, stop_reason="tool_use", tool_calls=()):
    return StreamComplete(response=LLMResponse(
        content="", tool_calls=list(tool_calls), stop_reason=stop_reason,
    ))


def _tool_round():
    """The tool activity that sits between two narration rounds."""
    return [
        StreamToolUseStart(id="t1", name="scratchpad"),
        StreamToolUseEnd(id="t1"),
        StreamToolResult(name="scratchpad", action="exec", content="ok", id="t1"),
    ]


async def test_rounds_are_separated_by_a_paragraph_break():
    events = [
        StreamTextDelta(text="I'll pull the movie list."),
        _round_end(tool_calls=[_TOOL_CALL]),
        *_tool_round(),
        StreamTextDelta(text="My regex is splitting the columns wrong."),
        _round_end(stop_reason="end_turn"),
    ]
    deltas, completed = await _drain(events)
    assert completed == (
        "I'll pull the movie list.\n\nMy regex is splitting the columns wrong."
    )
    # The break must ride the live delta stream too, not just the final text.
    assert "".join(deltas) == completed


async def test_single_round_is_untouched():
    events = [
        StreamTextDelta(text="Hello"),
        StreamTextDelta(text=" there."),
        _round_end(stop_reason="end_turn"),
    ]
    deltas, completed = await _drain(events)
    assert completed == "Hello there."
    assert deltas == ["Hello", " there."]


async def test_max_tokens_continuation_stays_seamless():
    events = [
        StreamTextDelta(text="and the winner is"),
        _round_end(stop_reason="max_tokens"),
        StreamTextDelta(text=" Avatar."),
        _round_end(stop_reason="end_turn"),
    ]
    _, completed = await _drain(events)
    assert completed == "and the winner is Avatar."


async def test_max_tokens_with_tool_calls_still_breaks():
    # anton only injects the mid-sentence continuation when the truncated
    # round had NO tool calls; with tool calls the loop proceeds normally,
    # so the next round is fresh narration.
    events = [
        StreamTextDelta(text="Running it now."),
        _round_end(stop_reason="max_tokens", tool_calls=[_TOOL_CALL]),
        *_tool_round(),
        StreamTextDelta(text="Done."),
        _round_end(stop_reason="end_turn"),
    ]
    _, completed = await _drain(events)
    assert completed == "Running it now.\n\nDone."


async def test_zero_text_truncated_round_keeps_the_armed_break():
    # A round can burn its whole output budget on thinking tokens: it ends
    # at max_tokens having streamed nothing visible. Its truncation belongs
    # to text the user never saw, so the break armed by the previous
    # narration round must survive it.
    events = [
        StreamTextDelta(text="Sentence one."),
        _round_end(tool_calls=[_TOOL_CALL]),
        *_tool_round(),
        _round_end(stop_reason="max_tokens"),
        StreamTextDelta(text="Sentence two."),
        _round_end(stop_reason="end_turn"),
    ]
    _, completed = await _drain(events)
    assert completed == "Sentence one.\n\nSentence two."


async def test_no_break_before_the_first_visible_text():
    # anthropic can emit an empty leading text delta; an empty round must
    # not make the real first text open with a blank line.
    events = [
        StreamTextDelta(text=""),
        _round_end(tool_calls=[_TOOL_CALL]),
        *_tool_round(),
        StreamTextDelta(text="Hello."),
        _round_end(stop_reason="end_turn"),
    ]
    _, completed = await _drain(events)
    assert completed == "Hello."


async def test_no_double_blank_line_when_round_already_ends_with_one():
    events = [
        StreamTextDelta(text="First thought.\n"),
        StreamTextDelta(text="\n"),
        _round_end(tool_calls=[_TOOL_CALL]),
        *_tool_round(),
        StreamTextDelta(text="Second thought."),
        _round_end(stop_reason="end_turn"),
    ]
    _, completed = await _drain(events)
    assert completed == "First thought.\n\nSecond thought."
