from __future__ import annotations

import json

import pytest

from anton.core.llm.provider import StreamTextDelta
from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream

# Mirror the production guard in anton_harness/stream_formatter.py: StreamReasoningDelta
# arrived with the reasoning-subtype channel (ENG-1109) and does not exist in anton
# builds before it (e.g. anton/main until staging is promoted). Skip rather than fail
# collection — the formatter degrades to a dead branch on older anton, so there is
# nothing to exercise until the symbol is present.
try:
    from anton.core.llm.provider import StreamReasoningDelta
except ImportError:
    StreamReasoningDelta = None

pytestmark = pytest.mark.skipif(
    StreamReasoningDelta is None,
    reason="anton build predates StreamReasoningDelta (ENG-1109); reasoning mapping is inert until anton is promoted",
)


async def _events(*items):
    for item in items:
        yield item


def _parse_sse(chunks: list[str]) -> list[dict]:
    parsed = []
    for chunk in chunks:
        for frame in chunk.strip("\n").split("\n\n"):
            for line in frame.split("\n"):
                if line.startswith("data:"):
                    parsed.append(json.loads(line[len("data:"):].strip()))
    return parsed


class TestReasoningDeltaMapping:
    async def test_reasoning_delta_maps_to_thought_progress_with_subtype(self):
        chunks = [
            c async for c in format_responses_stream(
                _events(StreamReasoningDelta(text="Let me check that first.")),
                model="claude-sonnet-4-6",
            )
        ]
        events = _parse_sse(chunks)
        reasoning_events = [e for e in events if e.get("thought_role") == "thought.progress" and e.get("subtype") == "reasoning"]
        assert len(reasoning_events) == 1
        assert reasoning_events[0]["content"] == "Let me check that first."

    async def test_reasoning_delta_never_becomes_output_text(self):
        # Reasoning text must never leak into the final-answer channel —
        # that's the whole point of routing it separately (ENG-1108/1109).
        chunks = [
            c async for c in format_responses_stream(
                _events(
                    StreamReasoningDelta(text="Thinking out loud..."),
                    StreamTextDelta(text="The real answer."),
                ),
                model="claude-sonnet-4-6",
            )
        ]
        events = _parse_sse(chunks)
        text_deltas = [e for e in events if e.get("type") == "response.output_text.delta"]
        assert len(text_deltas) == 1
        assert text_deltas[0]["delta"] == "The real answer."
