"""Rebuilding Anton's scratchpad cells from a conversation's stored events.

The harness reads only the events whose thought_role the extractor reacts to
(SCRATCHPAD_REPLAY_ROLES), so these tests pin both the state machine and the
claim that dropping every other event leaves the cells unchanged. The last
test pins that the harness hands the cells it rebuilds to the session.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict

import pytest
from anton.core.backends.base import Cell

from cowork.harnesses.anton_harness.scratchpad_cell_replay import (
    SCRATCHPAD_REPLAY_ROLES,
    extract_scratchpad_cells,
)


def _start() -> dict:
    return {"type": "response.in_progress", "thought_role": "thought.scratchpad.start", "content": "{}"}


def _end(action: str, *, code: str = "print(1)", tool_use_id: str | None = None) -> dict:
    event = {
        "type": "response.in_progress",
        "thought_role": "thought.scratchpad.end",
        "content": json.dumps({"action": action, "code": code}),
    }
    return event if tool_use_id is None else {**event, "tool_use_id": tool_use_id}


def _result(code: str, *, tool_use_id: str | None = None) -> dict:
    event = {
        "type": "response.in_progress",
        "thought_role": "thought.scratchpad.result",
        "content": json.dumps({"code": code, "stdout": "1\n", "stderr": "", "error": None}),
    }
    return event if tool_use_id is None else {**event, "tool_use_id": tool_use_id}


async def _stored_events(*, calls: list[tuple[str, str]]) -> list[dict]:
    """The events the real stream formatter stores for one scratchpad call
    per (code, result) pair, each streamed and then answered in turn."""
    from anton.core.llm.provider import (
        StreamToolResult,
        StreamToolUseDelta,
        StreamToolUseEnd,
        StreamToolUseStart,
    )

    from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream

    async def turn():
        for n, (code, result) in enumerate(calls):
            tool_input = json.dumps({"action": "exec", "name": "main", "code": code}, ensure_ascii=False)
            yield StreamToolUseStart(id=f"t{n}", name="scratchpad")
            yield StreamToolUseDelta(id=f"t{n}", json_delta=tool_input)
            yield StreamToolUseEnd(id=f"t{n}")
            yield StreamToolResult(name="scratchpad", action="exec", content=result, id=f"t{n}")

    stored: list[dict] = []
    async for _ in format_responses_stream(
        turn(), model="claude-sonnet-4-6", event_sink=lambda _type, data: stored.append(data)
    ):
        pass
    return stored


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


@pytest.mark.asyncio
async def test_a_result_the_store_cut_still_replays_its_code():
    """The stream formatter stores at most 65,536 characters of a cell's
    result. A cell whose logs pass that (1,000 HTTP calls log about 83,000
    characters) still replays its code, so the next turn's write check sees
    the statement it bound."""
    code = "fix_sql = \"UPDATE suppliers SET lead_time_days = 21 WHERE region = 'EU'\""
    ran = Cell(
        code=code,
        stdout="1000 lookups\n",
        stderr="",
        error=None,
        logs="INFO HTTP Request: GET https://api.example.com/x \"HTTP/1.1 200 OK\"\n" * 1200,
    )
    result = json.dumps(asdict(ran))
    assert len(result) > 65_536

    stored = await _stored_events(calls=[(code, result)])

    [stored_result] = [e for e in stored if e.get("thought_role") == "thought.scratchpad.result"]
    assert len(stored_result["content"]) == 65_536
    assert extract_scratchpad_cells(stored) == [
        Cell(code=code, stdout="1000 lookups\n", stderr="", error=None),
    ]


@pytest.mark.asyncio
async def test_a_result_cut_inside_its_code_replays_the_code_the_model_sent():
    """json.dumps(asdict(cell)) escapes each non-ASCII character to six, so
    the result of a cell with about 11,000 characters of Japanese report text
    passes the 65,536 characters the store keeps before its code ends. The
    model's tool input keeps the text as written, so the call's end event
    still holds the whole code, and the next turn's write check sees the
    statement it bound."""
    text = "サプライヤーの納期は地域ごとに異なります。" * 520
    code = (
        "fix_sql = \"UPDATE suppliers SET lead_time_days = 21 WHERE region = 'EU'\"\n"
        f'report = """{text}"""\n'
        'open("report.md", "w").write(report)'
    )
    result = json.dumps(asdict(Cell(code=code, stdout="", stderr="", error=None)))
    assert len(code) > 10_900
    assert len(result) > 65_536

    stored = await _stored_events(calls=[(code, result)])

    [end] = [e for e in stored if e.get("thought_role") == "thought.scratchpad.end"]
    [stored_result] = [e for e in stored if e.get("thought_role") == "thought.scratchpad.result"]
    assert json.loads(end["content"])["code"] == code
    assert len(stored_result["content"]) == 65_536
    assert extract_scratchpad_cells(stored) == [Cell(code=code, stdout="", stderr="", error=None)]


def test_a_result_cut_inside_its_code_makes_no_cell_when_its_end_event_was_cut_too():
    """Only a code field the cut left whole replays: half a statement is
    never handed on as the code a cell ran."""
    long_code = "y = " + "1" * 100
    cut_end = {**_end("exec"), "content": json.dumps({"action": "exec", "code": long_code})[:50]}
    cut = {**_result("x = 1"), "content": json.dumps({"code": long_code})[:40]}

    assert extract_scratchpad_cells([cut_end, cut]) == []


@pytest.mark.asyncio
async def test_a_cell_anton_never_ran_makes_no_cell():
    """anton closes a cell it never ran, such as a write the user didn't
    approve, with its reason as plain text. Its end event holds the whole
    code, but the next turn must not see that code as one that ran."""
    refused = (
        "Not run: this step would change data in a connected source. That needs the "
        "user's approval, and they declined it. Nothing in this cell ran."
    )
    stored = await _stored_events(calls=[
        ("print(1)", json.dumps(asdict(Cell(code="print(1)", stdout="1\n", stderr="", error=None)))),
        ("cur.execute(\"UPDATE suppliers SET lead_time_days = 21\")", refused),
    ])

    results = [e for e in stored if e.get("thought_role") == "thought.scratchpad.result"]
    assert [e["cell_status"] for e in results] == ["ok", "error"]
    assert extract_scratchpad_cells(stored) == [Cell(code="print(1)", stdout="1\n", stderr="", error=None)]


def test_each_result_pairs_with_the_call_of_its_own_id():
    """The model can send several calls at once, so every end event comes
    before the first result. Each result makes its own cell, and one cut
    inside its code takes its own call's code."""
    cut_a = {**_result("a = 1", tool_use_id="a"), "content": json.dumps({"code": "a = 1"})[:12]}

    cells = extract_scratchpad_cells([
        _end("exec", code="a = 1", tool_use_id="a"),
        _end("exec", code="b = 2", tool_use_id="b"),
        cut_a,
        _result("b = 2", tool_use_id="b"),
    ])

    assert cells == [
        Cell(code="a = 1", stdout="", stderr="", error=None),
        Cell(code="b = 2", stdout="1\n", stderr="", error=None),
    ]


def test_a_reset_sent_with_other_calls_drops_the_ones_sent_before_it():
    cells = extract_scratchpad_cells([
        _end("exec", code="a = 1", tool_use_id="a"),
        _end("reset", tool_use_id="r"),
        _end("exec", code="b = 2", tool_use_id="b"),
        _result("a = 1", tool_use_id="a"),
        _result("", tool_use_id="r"),
        _result("b = 2", tool_use_id="b"),
    ])

    assert [c.code for c in cells] == ["b = 2"]


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
        # Two calls sent at once: both end events come before either result.
        service.save_assistant_turn(conversation.id, "added", [
            _end("exec", code="y = x + 1", tool_use_id="t1"),
            _end("exec", code="z = y + 1", tool_use_id="t2"),
            _result("y = x + 1", tool_use_id="t1"),
            _result("z = y + 1", tool_use_id="t2"),
        ])

        config, _, _ = await harness.AntonHarness()._build_chat_session(conversation)

    assert config.cells == [
        Cell(code="x = 1", stdout="1\n", stderr="", error=None),
        Cell(code="y = x + 1", stdout="1\n", stderr="", error=None),
        Cell(code="z = y + 1", stdout="1\n", stderr="", error=None),
    ]
