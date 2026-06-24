"""Tests for the per-turn "memory loaded" signal.

The harness counts how many long-term memory entries the Cortex injects into
a turn's system prompt and emits a `MemoryLoaded` event up front; the stream
formatter maps it to a `thought.memory.loaded` SSE event (and drops a zero
count) so the desktop UI can show an honest "used N memories this turn" chip.

These guard both halves: the count helper reads the same hippocampus the
prompt is built from, and the formatter only surfaces a positive count.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from cowork.harnesses.anton_harness.harness import _count_loaded_memories
from cowork.harnesses.anton_harness.stream_formatter import (
    MemoryLoaded,
    format_responses_stream,
)


class _FakeHippocampus:
    def __init__(self, identities=0, rules=0, lessons=0):
        # The real getters return list[Engram]; only the length is read.
        self._identities = [object()] * identities
        self._rules = [object()] * rules
        self._lessons = [object()] * lessons

    def get_identities(self):
        return self._identities

    def get_rules(self):
        return self._rules

    def get_lessons(self, token_budget=None):
        return self._lessons


def _cortex(mode="autopilot", global_hc=None, project_hc=None):
    return SimpleNamespace(mode=mode, global_hc=global_hc, project_hc=project_hc)


class TestCountLoadedMemories:
    def test_none_cortex_is_zero(self):
        assert _count_loaded_memories(None) == 0

    def test_mode_off_is_zero_even_with_entries(self):
        cortex = _cortex(mode="off", global_hc=_FakeHippocampus(rules=5))
        assert _count_loaded_memories(cortex) == 0

    def test_sums_identities_rules_lessons_across_both_scopes(self):
        cortex = _cortex(
            global_hc=_FakeHippocampus(identities=1, rules=2, lessons=3),
            project_hc=_FakeHippocampus(identities=0, rules=4, lessons=1),
        )
        # 1+2+3 (global) + 0+4+1 (project) = 11
        assert _count_loaded_memories(cortex) == 11

    def test_missing_hippocampus_is_tolerated(self):
        cortex = _cortex(global_hc=_FakeHippocampus(rules=2), project_hc=None)
        assert _count_loaded_memories(cortex) == 2

    def test_getter_error_counts_as_zero(self):
        class _Boom:
            def get_identities(self):
                raise RuntimeError("boom")

        cortex = _cortex(global_hc=_Boom())
        assert _count_loaded_memories(cortex) == 0


async def _collect(events, model="test-model"):
    async def _gen():
        for e in events:
            yield e

    out = []
    async for chunk in format_responses_stream(_gen(), model=model):
        out.append(chunk)
    return out


def _parse_in_progress(chunks):
    """Pull the parsed data dicts of every response.in_progress SSE frame."""
    found = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if line.startswith("data:"):
                data = json.loads(line[len("data:"):].strip())
                if data.get("type") == "response.in_progress":
                    found.append(data)
    return found


class TestFormatterMemoryLoaded:
    @pytest.mark.asyncio
    async def test_positive_count_emits_thought_memory_loaded(self):
        chunks = await _collect([MemoryLoaded(count=7)])
        events = _parse_in_progress(chunks)
        assert len(events) == 1
        ev = events[0]
        assert ev["thought_role"] == "thought.memory.loaded"
        assert ev["memory_count"] == 7
        assert ev["content"] == "7"

    @pytest.mark.asyncio
    async def test_zero_count_emits_nothing(self):
        chunks = await _collect([MemoryLoaded(count=0)])
        assert _parse_in_progress(chunks) == []
