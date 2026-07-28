"""Mid-turn history compaction must not corrupt the captured tool-row slice.

anton summarizes history inside the turn under context pressure, which
reassigns `session.history` and invalidates the `seed_len` index. When that
happens the harness must skip tool-row capture and fall back to text-only
replay rather than emit a scrambled / orphaned tool sequence.
"""
import asyncio
from types import SimpleNamespace

import cowork.services.task_objects as task_objects
from cowork.harnesses.anton_harness.harness import AntonHarness
from cowork.harnesses.anton_harness.stream_formatter import TurnHistory


class StreamContextCompacted:
    """Name matches anton's event class — the harness detects it by class name."""


class _StreamTextDelta:
    pass


class _FakeSession:
    """anton ChatSession stand-in: turn_stream appends this turn's block-rows
    to `history`; when `compact` is set it also reassigns `history` mid-turn
    (as _summarize_history does) and yields a compaction event."""

    def __init__(self, compact: bool):
        self.history = [{"role": "user", "content": "seed"}]  # seed_len == 1
        self._compact = compact

    async def turn_stream(self, *_args, **_kwargs):
        self.history.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "x", "input": {}}]})
        self.history.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "B"}]})
        self.history.append({"role": "assistant", "content": [
            {"type": "text", "text": "done"}]})
        if self._compact:
            # anton folds the tool_result into a summary but keeps the tool_use
            # and the answer. Without the guard, seed_len slicing would capture
            # the tool_use with no matching tool_result — an orphan that breaks
            # the next turn's replay.
            tool_use, answer = self.history[1], self.history[3]
            self.history = [
                {"role": "assistant", "content": "summary"},
                tool_use,
                answer,
            ]
            yield StreamContextCompacted()
        yield _StreamTextDelta()


def _drive(monkeypatch, compact: bool):
    monkeypatch.setattr(task_objects, "snapshot_artifact_slugs", lambda *_a, **_k: set())
    monkeypatch.setattr(task_objects, "finalize_turn_artifacts", lambda *_a, **_k: [])

    session = _FakeSession(compact=compact)

    async def _fake_build(self, conversation, disabled_connections, channel_context=None):
        return session, None

    monkeypatch.setattr(AntonHarness, "_build_chat_session", _fake_build)

    conversation = SimpleNamespace(
        id="c1", project_id="p1", project=SimpleNamespace(path="/tmp"),
    )

    async def _drain():
        return [
            ev
            async for ev in AntonHarness().stream_response(
                conversation=conversation,
                input=[{"type": "text", "text": "hi"}],
            )
        ]

    return asyncio.run(_drain())


def test_captures_tool_rows_without_compaction(monkeypatch):
    events = _drive(monkeypatch, compact=False)
    histories = [e for e in events if isinstance(e, TurnHistory)]
    assert len(histories) == 1
    assert [r["role"] for r in histories[0].rows] == ["assistant", "user"]


def test_skips_capture_after_midturn_compaction(monkeypatch):
    events = _drive(monkeypatch, compact=True)
    assert not any(isinstance(e, TurnHistory) for e in events)
