from __future__ import annotations

import json

from anton.core.llm.provider import StreamReasoningDelta, StreamTextDelta
from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream


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


class TestReasoningDeltaImportGuard:
    async def test_formatter_survives_anton_without_reasoning_delta(self, monkeypatch):
        # ENG-1167: StreamReasoningDelta (ENG-1109) is newer than the anton a
        # version-skewed install can resolve — desktop staging builds install
        # anton from PyPI, which lags the staging branch. The formatter runs on
        # every turn, so an unguarded `from anton.core.llm.provider import
        # StreamReasoningDelta` would raise ImportError and break all chat with
        # a generic error. Simulate a pre-ENG-1109 anton by removing the symbol
        # and assert the turn still streams cleanly.
        import anton.core.llm.provider as provider_mod

        monkeypatch.delattr(provider_mod, "StreamReasoningDelta", raising=False)

        chunks = [
            c async for c in format_responses_stream(
                _events(StreamTextDelta(text="hello")),
                model="claude-sonnet-4-6",
            )
        ]
        events = _parse_sse(chunks)
        types = [e.get("type") for e in events]
        assert "response.created" in types
        assert "response.completed" in types
        text_deltas = [e for e in events if e.get("type") == "response.output_text.delta"]
        assert [d["delta"] for d in text_deltas] == ["hello"]
