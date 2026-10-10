"""Rebuilding Anton's scratchpad cells from a conversation's stored events.

The harness reads only the events whose thought_role the extractor reacts to
(SCRATCHPAD_REPLAY_ROLES), so these tests pin both the state machine and the
claim that dropping every other event leaves the cells unchanged. The last
test pins that the harness hands the cells it rebuilds to the session.
"""
from __future__ import annotations

import json
import random

import pytest
from anton.core.backends.base import Cell

from cowork.harnesses.anton_harness.scratchpad_cell_replay import (
    SCRATCHPAD_REPLAY_ROLES,
    extract_scratchpad_cells,
)


def _start() -> dict:
    return {"type": "response.in_progress", "thought_role": "thought.scratchpad.start", "content": "{}"}


def _end(action: str) -> dict:
    return {
        "type": "response.in_progress",
        "thought_role": "thought.scratchpad.end",
        "content": json.dumps({"action": action, "code": "print(1)"}),
    }


def _result(code: str) -> dict:
    return {
        "type": "response.in_progress",
        "thought_role": "thought.scratchpad.result",
        "content": json.dumps({"code": code, "stdout": "1\n", "stderr": "", "error": None}),
    }


def test_an_exec_end_and_its_result_make_a_cell():
    cells = extract_scratchpad_cells([_start(), _end("exec"), _result("x = 1")])

    assert cells == [Cell(code="x = 1", stdout="1\n", stderr="", error=None)]


def test_a_result_after_a_non_exec_end_makes_no_cell():
    cells = extract_scratchpad_cells([_end("view"), _result("x = 1")])

    assert cells == []


def test_a_result_with_no_end_before_it_makes_no_cell():
    cells = extract_scratchpad_cells([_end("exec"), _result("x = 1"), _result("y = 2")])

    assert [c.code for c in cells] == ["x = 1"]


def test_reset_clears_the_cells_before_it():
    cells = extract_scratchpad_cells([
        _end("exec"), _result("x = 1"),
        _end("reset"),
        _end("exec"), _result("y = 2"),
    ])

    assert [c.code for c in cells] == ["y = 2"]


def test_malformed_content_is_ignored():
    broken_end = {**_end("exec"), "content": "{not json"}
    broken_result = {**_result("x = 1"), "content": "[1, 2]"}

    cells = extract_scratchpad_cells([
        broken_end, _result("lost"),
        _end("exec"), broken_result,
        _end("exec"), _result("kept"),
    ])

    assert [c.code for c in cells] == ["kept"]


def test_events_that_are_not_objects_are_skipped():
    cells = extract_scratchpad_cells([_end("exec"), "a string", ["a", "list"], None, _result("x = 1")])

    assert [c.code for c in cells] == ["x = 1"]


def test_dropping_events_outside_the_replay_roles_keeps_the_cells():
    """The harness reads only the events whose thought_role is one of
    SCRATCHPAD_REPLAY_ROLES. If the extractor ever reacts to another
    thought_role, this fails, rather than that read silently dropping the
    events it needs."""
    rng = random.Random(3362)
    pool = [
        _start,
        lambda: _end("exec"),
        lambda: _end("reset"),
        lambda: _end("view"),
        lambda: _result(f"c{rng.randrange(1000)}"),
        lambda: {"type": "response.output_text.delta", "delta": "hi"},
        lambda: {"type": "response.in_progress", "thought_role": "thought.progress", "content": "p"},
        lambda: {"type": "response.completed"},
        lambda: "not an object",
    ]
    for _ in range(200):
        events = [rng.choice(pool)() for _ in range(rng.randrange(1, 40))]
        kept = [
            e for e in events
            if isinstance(e, dict) and e.get("thought_role") in SCRATCHPAD_REPLAY_ROLES
        ]

        assert extract_scratchpad_cells(kept) == extract_scratchpad_cells(events)


@pytest.mark.asyncio
async def test_the_harness_hands_the_session_the_cells_a_conversation_stored(monkeypatch):
    """A resumed conversation's session starts with the cells its earlier
    answers ran, read from their stored events by the real harness."""
    from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
    from cowork.db.session import get_open_session
    from cowork.harnesses.anton_harness import harness
    from cowork.services.conversations import ConversationService

    # Capture the config rather than a session: no scratchpad, no connectors.
    monkeypatch.setattr(harness, "build_chat_session", lambda config: config)
    monkeypatch.setattr("anton.core.datasources.data_vault.LocalDataVault", None)
    monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "false")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only-key")
    with get_open_session() as db:
        service = ConversationService(ScopedSession(db, LOCAL_SCOPE))
        conversation = service.create_conversation(topic="cell replay")
        service.save_user_message(conversation.id, "load it")
        service.save_assistant_turn(conversation.id, "loaded", [
            _start(), _end("exec"),
            {"type": "response.output_text.delta", "delta": "loaded"},
            _result("x = 1"),
            {"type": "response.in_progress", "thought_role": "thought.progress", "content": "p"},
        ])
        service.save_user_message(conversation.id, "add one")
        service.save_assistant_turn(conversation.id, "added", [_end("exec"), _result("y = x + 1")])

        config, _, _ = await harness.AntonHarness()._build_chat_session(conversation)

    assert config.cells == [
        Cell(code="x = 1", stdout="1\n", stderr="", error=None),
        Cell(code="y = x + 1", stdout="1\n", stderr="", error=None),
    ]
