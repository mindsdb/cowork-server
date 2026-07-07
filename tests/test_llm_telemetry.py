"""Context-consumption telemetry (ENG-642).

Covers the three capture points and the report parser:
  - format_responses_stream records anton's per-call usage via the event
    sink WITHOUT changing the wire protocol (no new SSE event types)
  - build_prompt_anatomy sizes the components cowork controls
  - scripts/context_report.py parses the tagged log lines back
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from anton.core.llm.provider import LLMResponse, StreamComplete, StreamTextDelta, Usage

from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream
from cowork.services.llm_telemetry import (
    anton_static_section_chars,
    build_prompt_anatomy,
)


async def _fake_turn(events):
    for event in events:
        yield event


def _complete(input_tokens: int, output_tokens: int, pressure: float) -> StreamComplete:
    return StreamComplete(response=LLMResponse(
        content="",
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            context_pressure=pressure,
        ),
        stop_reason="end_turn",
    ))


class TestUsageCapture:
    async def test_usage_reaches_sink_not_wire(self):
        events = [
            StreamTextDelta(text="hello"),
            _complete(16000, 50, 0.08),
            StreamTextDelta(text=" world"),
            _complete(17000, 80, 0.085),
        ]
        sunk: list[tuple[str, dict]] = []
        sse: list[str] = []
        async for chunk in format_responses_stream(
            _fake_turn(events), "test-model", lambda t, d: sunk.append((t, d)),
        ):
            sse.append(chunk)

        usage = [d for t, d in sunk if t == "response.usage"]
        assert [u["call_seq"] for u in usage] == [1, 2]
        assert usage[0]["input_tokens"] == 16000
        assert usage[1]["output_tokens"] == 80
        assert usage[0]["model"] == "test-model"
        assert usage[0]["cache_read_input_tokens"] is None  # until caching lands
        assert all(isinstance(u.get("at_ms"), int) for u in usage)
        # Wire protocol unchanged: usage is recorded, never streamed.
        assert not any("response.usage" in chunk for chunk in sse)
        # And the existing events still flow.
        assert any("response.output_text.delta" in chunk for chunk in sse)


class TestPromptAnatomy:
    def test_sizes_components(self):
        payload = build_prompt_anatomy(
            conversation_id="c-1",
            turn_id=4,
            initial_history=[
                {"role": "user", "content": "ab"},
                {"role": "assistant", "content": "cdef"},
            ],
            suffix_parts={"project_context": "xyz", "attachment_context": None},
            tool_defs=[{"name": "scratchpad", "description": "d" * 10, "input_schema": {}}],
        )
        assert payload["history_messages"] == 2
        assert payload["history_chars"] == 6
        assert payload["suffix_chars"] == {"project_context": 3, "attachment_context": 0}
        assert payload["tool_chars"]["scratchpad"] == payload["tools_total_chars"] > 10
        json.dumps(payload)  # must be loggable as-is

    def test_anton_static_sections_measured(self):
        sizes = anton_static_section_chars()
        # anton is an install-time dependency; the big blocks must be visible.
        assert sizes.get("CHAT_SYSTEM_PROMPT", 0) > 1000
        assert sizes.get("BACKEND_GENERATION_PROMPT", 0) > 1000


class TestReportParser:
    def test_round_trip(self):
        spec = importlib.util.spec_from_file_location(
            "context_report",
            Path(__file__).resolve().parents[1] / "scripts" / "context_report.py",
        )
        report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report)

        lines = [
            'INFO cowork [llm_usage] {"conversation_id": "c", "turn_id": 0, '
            '"call_seq": 1, "input_tokens": 16000, "output_tokens": 10, '
            '"context_pressure": 0.72, "model": "m"}\n',
            'INFO cowork [turn_summary] {"conversation_id": "c", "turn_id": 0, '
            '"calls": 3, "input_tokens": 50000, "output_tokens": 900, '
            '"ttft_ms": 1200, "duration_ms": 9000}\n',
            "noise line without any tag\n",
        ]
        rec = report.parse(lines)
        assert rec["[llm_usage]"][0]["input_tokens"] == 16000
        assert rec["[turn_summary]"][0]["calls"] == 3
        assert report.pct([1, 2, 3, None], 50) == 2
        assert report.pct([], 50) is None
