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
from types import SimpleNamespace

import pytest
from anton.core.backends.base import Cell
from anton.core.interaction.elicit import AskAnswer, AskRequest
from anton.core.interaction.emitter import TurnEmitter
from anton.core.utils.scratchpad import WriteGate, hold_unapproved_write
from anton.core.utils.write_check import snapshot_project_files

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


async def _stored_events(*, calls: list[tuple[str, str]], interleaved: bool = False) -> list[dict]:
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
        results = []
        for n, (code, result) in enumerate(calls):
            tool_input = json.dumps({"action": "exec", "name": "main", "code": code}, ensure_ascii=False)
            yield StreamToolUseStart(id=f"t{n}", name="scratchpad")
            yield StreamToolUseDelta(id=f"t{n}", json_delta=tool_input)
            yield StreamToolUseEnd(id=f"t{n}")
            event = StreamToolResult(name="scratchpad", action="exec", content=result, id=f"t{n}")
            if interleaved:
                results.append(event)
            else:
                yield event
        for event in reversed(results):
            yield event

    stored: list[dict] = []
    async for _ in format_responses_stream(
        turn(), model="claude-sonnet-4-6", event_sink=lambda _type, data: stored.append(data)
    ):
        pass
    return stored


def _persisted_cells(events: list[dict]) -> list[Cell]:
    from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
    from cowork.db.session import get_open_session
    from cowork.services.conversations import ConversationService

    with get_open_session() as db:
        service = ConversationService(ScopedSession(db, LOCAL_SCOPE))
        conversation = service.create_conversation(topic="long cell replay")
        service.save_user_message(conversation.id, "prepare the report")
        service.save_assistant_turn(conversation.id, "prepared", events)
        conversation_id = conversation.id
    with get_open_session() as db:
        service = ConversationService(ScopedSession(db, LOCAL_SCOPE))
        history = service.get_replay_history(conversation_id, event_roles=SCRATCHPAD_REPLAY_ROLES)
    return extract_scratchpad_cells(history.events)


class _CancelWrites:
    supported_kinds = ("choice",)
    timeout_s = None

    def __init__(self) -> None:
        self.requests: list[AskRequest] = []

    async def begin(self, question_id: str, request: AskRequest) -> None:
        pass

    async def ask(self, question_id: str, request: AskRequest) -> AskAnswer:
        self.requests.append(request)
        return AskAnswer(status="answered", values=("cancel",))

    async def end(self, question_id: str) -> None:
        pass


@pytest.mark.parametrize("character", ["x", "納"], ids=["ascii", "unicode"])
async def test_long_call_code_survives_storage_and_holds_a_later_bound_write(character):
    """Both event payloads used to lose the code, hiding the later SQL binding
    from the next turn's real approval check."""
    code = f"report = {character * 70_000!r}\nfix_sql = 'DELETE FROM suppliers WHERE id = 7'"
    result = json.dumps(asdict(Cell(code=code, stdout="", stderr="", error=None)))
    stored = await _stored_events(calls=[(code, result)])
    elicitor = _CancelWrites()
    emitter = TurnEmitter()
    session = SimpleNamespace(
        _write_gate=WriteGate(), elicitor=elicitor, emitter=emitter, emit=emitter.emit, answer_wait_s=0.0,
    )
    cells = _persisted_cells(stored)

    outcome = await hold_unapproved_write(session, pad=SimpleNamespace(cells=cells), code="cur.execute(fix_sql)")

    assert outcome is not None and outcome.reason == "write_not_approved"
    assert len(elicitor.requests) == 1
    assert "DELETE FROM suppliers WHERE id = 7" in elicitor.requests[0].prompt
    assert cells == [Cell(code=code, stdout="", stderr="", error=None)]
    [end] = [e for e in stored if e.get("thought_role") == "thought.scratchpad.end"]
    assert json.loads(end["content"])["code"] == code
    [stored_result] = [e for e in stored if e.get("thought_role") == "thought.scratchpad.result"]
    assert len(stored_result["content"]) == 65_536


@pytest.mark.parametrize("error", [None, "ZeroDivisionError: division by zero", "Cell timed out"])
@pytest.mark.parametrize("code_size", [1000, 70_000], ids=["error-cut", "code-and-error-cut"])
async def test_long_cell_failure_survives_storage_without_granting_file_ownership(tmp_path, monkeypatch, error, code_size):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "report.md").write_text("user's report")
    code = f"report = {'x' * code_size!r}\n"
    if error:
        code += "x = 1 / 0\n"
    code += "open('report.md', 'w').write('draft')"
    result = json.dumps(asdict(Cell(code=code, stdout="x" * 70_000, stderr="", error=error)))
    cells = _persisted_cells(await _stored_events(calls=[(code, result)]))
    elicitor = _CancelWrites()
    emitter = TurnEmitter()
    session = SimpleNamespace(
        _write_gate=WriteGate(snapshots=(snapshot_project_files(root=tmp_path, label="project", taken_at=10**10),)),
        elicitor=elicitor,
        emitter=emitter,
        emit=emitter.emit,
        answer_wait_s=0.0,
    )

    outcome = await hold_unapproved_write(
        session, pad=SimpleNamespace(cells=cells), code="open('report.md', 'w').write('replacement')",
    )

    assert (outcome is not None, len(elicitor.requests)) == (bool(error), int(bool(error)))
    assert len(cells) == 1 and cells[0].code == code
    assert bool(cells[0].error) is bool(error)


async def test_interleaved_long_calls_keep_their_own_code_and_status_after_storage():
    calls = [
        Cell(code=f"first = {'x' * 70_000!r}", stdout="", stderr="", error="ZeroDivisionError"),
        Cell(code=f"second = {'納' * 70_000!r}", stdout="", stderr="", error=None),
    ]
    events = await _stored_events(
        calls=[(cell.code, json.dumps(asdict(cell))) for cell in calls], interleaved=True,
    )

    cells = _persisted_cells(events)

    assert [cell.code for cell in cells] == [cell.code for cell in reversed(calls)]
    assert [bool(cell.error) for cell in cells] == [False, True]


@pytest.mark.parametrize("status", [None, "ok", "error", "timeout", "unrecognized", ["ok"]])
@pytest.mark.parametrize("cut", ["code", "stdout", "stderr", "error"], ids=lambda field: f"{field}-cut")
def test_a_missing_error_uses_only_an_explicit_success_status(status, cut):
    """A result cut inside any field before its error is whole counts as a
    failure unless its status says ok, and says how it ended where it knows."""
    code = "open('report.md', 'w').write('draft')"
    whole = json.dumps({"code": code, "stdout": "long output", "stderr": "a warning",
                        "error": "ZeroDivisionError: division by zero"})
    content = whole[:whole.index(f'"{cut}": ') + len(cut) + 8]
    result = {**_result(code), "content": content}
    if status is not None:
        result["cell_status"] = status

    [cell] = extract_scratchpad_cells([_end("exec", code=code), result])

    assert cell.code == code
    assert (cell.error is None) is (status == "ok")
    expected = {"error": "The cell failed", "timeout": "The cell timed out", "ok": None}.get(
        status if isinstance(status, str) else None, "whether it succeeded is unknown"
    )
    assert cell.error is None if expected is None else expected in cell.error


@pytest.mark.parametrize("status", [None, "ok", "error", "timeout", "unrecognized"])
@pytest.mark.parametrize("error", [None, "", "ZeroDivisionError: division by zero"])
def test_an_explicit_error_field_wins_over_the_event_status(status, error):
    result = _result("x = 1")
    result["content"] = json.dumps(asdict(Cell(code="x = 1", stdout="", stderr="", error=error, logs="x" * 70_000)))[:65_536]
    if status is not None:
        result["cell_status"] = status

    [cell] = extract_scratchpad_cells([_end("exec"), result])

    assert cell.error == error


async def test_only_scratchpad_call_inputs_lose_the_length_cap():
    from anton.core.llm.provider import StreamToolResult, StreamToolUseDelta, StreamToolUseEnd, StreamToolUseStart
    from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream

    content = "x" * 70_000

    async def turn():
        yield StreamToolUseStart(id="other", name="recall")
        yield StreamToolUseDelta(id="other", json_delta=content)
        yield StreamToolUseEnd(id="other")
        yield StreamToolResult(name="recall", action="message", content=content, id="other")
        yield StreamToolResult(name="recall", action="read", content=content, id="other")
        # An MCP tool named like the scratchpad keeps the cap on its input.
        yield StreamToolUseStart(id="notes", name="notes__scratchpad_append")
        yield StreamToolUseDelta(id="notes", json_delta=content)
        yield StreamToolUseEnd(id="notes")

    stored: list[dict] = []
    async for _ in format_responses_stream(
        turn(), model="claude-sonnet-4-6", event_sink=lambda _type, data: stored.append(data),
    ):
        pass

    contents = [e["content"] for e in stored if e.get("thought_role") in {
        "thought.recall.end", "thought.tool_call.message", "thought.scratchpad.result", "thought.scratchpad.end",
    }]
    assert contents == [content[:65_536]] * 4


@pytest.mark.xfail(
    strict=True,
    reason="anton-agent 2.26.10.19.3 drops a compacted cell's error flag; the floor moves to the release that keeps it",
)
@pytest.mark.parametrize("code_size", [1000, 70_000], ids=["error-cut", "code-and-error-cut"])
async def test_a_replayed_failure_folded_by_compaction_grants_no_file_ownership(tmp_path, monkeypatch, code_size):
    """A replayed cell that failed while writing the user's report still
    makes no file the agent's own after compaction folds it."""
    from anton.core.backends.local import LocalScratchpadRuntime

    monkeypatch.chdir(tmp_path)
    (tmp_path / "report.md").write_text("user's report")
    code = f"report = {'x' * code_size!r}\nx = 1 / 0\nopen('report.md', 'w').write('draft')"
    result = json.dumps(asdict(Cell(code=code, stdout="x" * 70_000, stderr="", error="ZeroDivisionError")))
    cells = _persisted_cells(await _stored_events(calls=[(code, result)]))
    assert len(cells) == 1 and cells[0].error
    pad = LocalScratchpadRuntime(
        name="main", coding_provider="anthropic", coding_model="", coding_api_key="", coding_base_url="",
        cells=cells, workspace_path=tmp_path, _venvs_base=tmp_path / "venvs",
    )
    pad.cells += [Cell(code=f"total_{n} = {n}", stdout="", stderr="", error=None) for n in range(6)]
    assert pad._compact_cells()
    assert all(cell.code != code for cell in pad.cells)
    elicitor = _CancelWrites()
    emitter = TurnEmitter()
    session = SimpleNamespace(
        _write_gate=WriteGate(snapshots=(snapshot_project_files(root=tmp_path, label="project", taken_at=10**10),)),
        elicitor=elicitor, emitter=emitter, emit=emitter.emit, answer_wait_s=0.0,
    )

    outcome = await hold_unapproved_write(session, pad=pad, code="open('report.md', 'w').write('replacement')")

    assert outcome is not None and len(elicitor.requests) == 1


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
@pytest.mark.parametrize("padding", ["", "#" + "x" * 70_000 + "\n"], ids=["short", "long"])
async def test_a_cell_anton_never_ran_makes_no_cell(padding):
    """anton closes a cell it never ran, such as a write the user didn't
    approve, with its reason as plain text. Its end event holds the whole
    code, but the next turn must not see that code as one that ran."""
    from anton.core.utils.scratchpad import _refusal
    from anton.core.utils.write_check import CellWrites

    refused = _refusal(CellWrites(unchecked=True), "cancelled").content
    stored = await _stored_events(calls=[
        ("print(1)", json.dumps(asdict(Cell(code="print(1)", stdout="1\n", stderr="", error=None)))),
        (padding + "cur.execute(\"UPDATE suppliers SET lead_time_days = 21\")", refused),
    ])

    results = [e for e in stored if e.get("thought_role") == "thought.scratchpad.result"]
    assert [e["cell_status"] for e in results] == ["ok", "error"]
    assert extract_scratchpad_cells(stored) == [Cell(code="print(1)", stdout="1\n", stderr="", error=None)]


def test_each_result_pairs_with_the_call_of_its_own_id():
    """The model can send several calls at once, so every end event comes
    before the first result. Each result makes its own cell, and one cut
    inside its code takes its own call's code."""
    cut_a = {**_result("a = 1", tool_use_id="a"), "content": json.dumps({"code": "a = 1"})[:12], "cell_status": "ok"}

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
