"""Per-turn usage telemetry surfaces TOKENS CONSUMED on response.completed.

Product policy: we do NOT expose any computed dollar cost to the user. anton
still prices each call internally (``usage.cost_usd`` — reserved for operator-side
budgeting), but the Anton harness formatter forwards only input/output tokens on
``response.completed`` — never cost. These tests pin that: tokens surface, and
``cost_usd`` is never present in the client-facing payload even when anton computed it.
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


def _completed_usage(frames: list[str]) -> dict:
    completed = [f for f in frames if "response.completed" in f]
    assert len(completed) == 1, frames
    return json.loads(completed[0].split("data: ", 1)[1].strip())["usage"]


def _usage(input_tokens: int, output_tokens: int, *, model="claude-sonnet-4-6") -> Usage:
    # cost_usd is deliberately populated (anton computes it) so the tests prove
    # the formatter HIDES it rather than simply never computing it.
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=compute_cost(model, input_tokens, output_tokens),
    )


class TestTokensSurfaceNotCost:
    def test_tokens_forwarded_and_cost_hidden(self):
        events = [
            StreamTextDelta(text="hello"),
            StreamComplete(response=LLMResponse(content="hello", usage=_usage(1_000_000, 1_000_000))),
        ]
        usage = _completed_usage(_collect(events))
        assert usage["input_tokens"] == 1_000_000
        assert usage["output_tokens"] == 1_000_000
        assert "cost_usd" not in usage  # policy: never expose dollar cost to the user

    def test_multiple_rounds_tokens_summed(self):
        # A tool-use turn has several LLM rounds; tokens add up, cost stays hidden.
        events = [
            StreamComplete(response=LLMResponse(content="", usage=_usage(1000, 2000))),
            StreamComplete(response=LLMResponse(content="done", usage=_usage(3000, 4000))),
        ]
        usage = _completed_usage(_collect(events))
        assert usage["input_tokens"] == 4000
        assert usage["output_tokens"] == 6000
        assert "cost_usd" not in usage

    def test_zero_tokens_when_no_complete_event(self):
        usage = _completed_usage(_collect([StreamTextDelta(text="hi")]))
        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0
        assert "cost_usd" not in usage
