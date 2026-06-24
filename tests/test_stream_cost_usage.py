"""The per-turn USD cost meter surfaces on response.completed.

anton prices every LLM call additively on usage.cost_usd (see the anton fork's
core/llm/pricing.py). The Anton harness formatter sums those across a turn's
StreamComplete events and forwards the total on the final response.completed SSE
event so the UI can show "$ this turn". This is surfacing only — no enforcement.
"""

from __future__ import annotations

import asyncio
import json

from anton.core.llm.pricing import compute_cost
from anton.core.llm.provider import LLMResponse, StreamComplete, StreamTextDelta, Usage

from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream


def _collect(events: list) -> list[str]:
    async def _run() -> list[str]:
        async def _gen():
            for ev in events:
                yield ev

        return [frame async for frame in format_responses_stream(_gen(), model="claude-sonnet-4-6")]

    return asyncio.run(_run())


def _completed_payload(frames: list[str]) -> dict:
    completed = [f for f in frames if "response.completed" in f]
    assert len(completed) == 1, frames
    return json.loads(completed[0].split("data: ", 1)[1].strip())


def _usage(input_tokens: int, output_tokens: int, *, model="claude-sonnet-4-6") -> Usage:
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=compute_cost(model, input_tokens, output_tokens),
    )


class TestCostSurfacesOnCompleted:
    def test_single_round_cost_forwarded(self):
        # Sonnet 1M in + 1M out → $18.00, surfaced verbatim on completed.
        events = [
            StreamTextDelta(text="hello"),
            StreamComplete(response=LLMResponse(content="hello", usage=_usage(1_000_000, 1_000_000))),
        ]
        payload = _completed_payload(_collect(events))
        assert payload["usage"]["cost_usd"] == 18.0
        assert payload["usage"]["input_tokens"] == 1_000_000
        assert payload["usage"]["output_tokens"] == 1_000_000

    def test_multiple_rounds_summed(self):
        # A tool-use turn has several LLM rounds; their costs add up.
        events = [
            StreamComplete(response=LLMResponse(content="", usage=_usage(1000, 2000))),
            StreamComplete(response=LLMResponse(content="done", usage=_usage(3000, 4000))),
        ]
        payload = _completed_payload(_collect(events))
        expected = compute_cost("claude-sonnet-4-6", 1000, 2000) + compute_cost(
            "claude-sonnet-4-6", 3000, 4000
        )
        assert payload["usage"]["cost_usd"] == round(expected, 6)
        assert payload["usage"]["input_tokens"] == 4000
        assert payload["usage"]["output_tokens"] == 6000

    def test_zero_cost_when_no_complete_event(self):
        # No StreamComplete → no usage recorded; meter reads zero, no crash.
        payload = _completed_payload(_collect([StreamTextDelta(text="hi")]))
        assert payload["usage"]["cost_usd"] == 0.0
        assert payload["usage"]["input_tokens"] == 0

    def test_unpriced_model_costs_zero(self):
        # An unpriced model prices each call at 0.0 — the turn total is 0.0,
        # and the meter still surfaces (does not break the stream).
        events = [
            StreamComplete(
                response=LLMResponse(content="x", usage=_usage(5000, 5000, model="some-unlisted-model"))
            ),
        ]
        payload = _completed_payload(_collect(events))
        assert payload["usage"]["cost_usd"] == 0.0
