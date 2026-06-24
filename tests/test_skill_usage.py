"""Skill usage signal: the `recall_skill` -> `response.skill_recalled`
stream event, the `used` counter bump it drives, and stat serialization.

No real ~/.cowork is touched — these use the conftest temp SQLite DB
(get_open_session) and drive the formatter with synthetic Anton stream
events, so nothing hits the Anton skill store on disk.
"""

from __future__ import annotations

import asyncio
import json
import uuid

from anton.core.llm.provider import (
    StreamComplete,
    StreamToolUseDelta,
    StreamToolUseEnd,
    StreamToolUseStart,
)

from cowork.db.session import get_open_session
from cowork.harnesses.anton_harness.stream_formatter import (
    _extract_recall_label,
    format_responses_stream,
)
from cowork.schemas.skills import SkillResponse
from cowork.services.skills import SkillService


# ── helpers ──────────────────────────────────────────────────────────


async def _aiter(events):
    for e in events:
        yield e


def _run_formatter(events):
    """Drive the formatter and return the parsed SSE event dicts."""

    async def collect():
        out = []
        async for sse in format_responses_stream(_aiter(events), model="test"):
            # Each SSE frame is `event: <type>\ndata: <json>\n\n`.
            for line in sse.split("\n"):
                if line.startswith("data:"):
                    out.append(json.loads(line[5:].strip()))
        return out

    return asyncio.run(collect())


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _make_skill(label: str, name: str):
    svc = SkillService(get_open_session())
    return svc.create_skill(label=label, name=name, instructions="# do the thing")


# ── _extract_recall_label ────────────────────────────────────────────


class TestExtractRecallLabel:
    def test_clean_json(self):
        assert _extract_recall_label('{"label": "csv-summary"}') == "csv-summary"

    def test_whitespace_trimmed(self):
        assert _extract_recall_label('{"label": "  csv-summary  "}') == "csv-summary"

    def test_truncated_json_falls_back_to_regex(self):
        # The formatter caps tool input; a clipped blob must still yield
        # the label via the regex fallback.
        assert _extract_recall_label('{"label": "csv-summary", "extra": "abcde') == "csv-summary"

    def test_empty_or_unusable(self):
        assert _extract_recall_label("") == ""
        assert _extract_recall_label("not json at all") == ""
        assert _extract_recall_label('{"other": "x"}') == ""


# ── formatter emits response.skill_recalled ──────────────────────────


class TestSkillRecalledEvent:
    def test_recall_skill_emits_skill_recalled(self):
        tid = "tool-1"
        events = [
            StreamToolUseStart(id=tid, name="recall_skill"),
            StreamToolUseDelta(id=tid, json_delta='{"label": "csv-summary"}'),
            StreamToolUseEnd(id=tid),
            StreamComplete(response=None),
        ]
        parsed = _run_formatter(events)
        recalled = [e for e in parsed if e.get("type") == "response.skill_recalled"]
        assert len(recalled) == 1
        assert recalled[0]["skill_label"] == "csv-summary"
        assert recalled[0]["tool_use_id"] == tid

    def test_recall_skill_still_emits_thought_recall(self):
        # Additive: the existing thinking-block recall signal must remain.
        tid = "tool-2"
        events = [
            StreamToolUseStart(id=tid, name="recall_skill"),
            StreamToolUseDelta(id=tid, json_delta='{"label": "csv-summary"}'),
            StreamToolUseEnd(id=tid),
            StreamComplete(response=None),
        ]
        parsed = _run_formatter(events)
        roles = [e.get("thought_role") for e in parsed]
        assert "thought.recall.start" in roles
        assert "thought.recall.end" in roles

    def test_generic_memory_recall_does_not_emit_skill_recalled(self):
        # The memory `recall` tool also contains the substring "recall"
        # but is NOT a skill — it must not produce a skill chip.
        tid = "tool-3"
        events = [
            StreamToolUseStart(id=tid, name="recall"),
            StreamToolUseDelta(id=tid, json_delta='{"query": "what did we decide"}'),
            StreamToolUseEnd(id=tid),
            StreamComplete(response=None),
        ]
        parsed = _run_formatter(events)
        assert not [e for e in parsed if e.get("type") == "response.skill_recalled"]

    def test_recall_skill_without_label_emits_nothing(self):
        tid = "tool-4"
        events = [
            StreamToolUseStart(id=tid, name="recall_skill"),
            StreamToolUseDelta(id=tid, json_delta="{}"),
            StreamToolUseEnd(id=tid),
            StreamComplete(response=None),
        ]
        parsed = _run_formatter(events)
        assert not [e for e in parsed if e.get("type") == "response.skill_recalled"]


# ── SkillService.record_use bumps the counter ────────────────────────


class TestRecordUse:
    def test_record_use_increments(self):
        label = _unique("csv-summary")
        skill = _make_skill(label, _unique("CSV Summary"))
        assert skill.used == 0

        updated = SkillService(get_open_session()).record_use(label)
        assert updated is not None
        assert updated.used == 1

        # Second recall keeps climbing, and the new value is persisted
        # (visible to a fresh service/session).
        SkillService(get_open_session()).record_use(label)
        again = SkillService(get_open_session()).get_skill(skill.id)
        assert again.used == 2

    def test_record_use_unknown_label_is_noop(self):
        assert SkillService(get_open_session()).record_use("no-such-skill-xyz") is None


# ── SkillResponse serializes the stats ───────────────────────────────


class TestSkillResponseSerialization:
    def test_used_and_confidence_serialized(self):
        label = _unique("ser")
        skill = _make_skill(label, _unique("Ser Skill"))
        SkillService(get_open_session()).record_use(label)
        fresh = SkillService(get_open_session()).get_skill(skill.id)

        payload = SkillResponse.serialize(fresh)
        assert payload["used"] == 1
        assert payload["confidence"] == 0.0
        # camelCase contract preserved for the existing field.
        assert "declarative" in payload
