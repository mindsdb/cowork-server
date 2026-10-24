from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from anton.core.backends.base import Cell
from anton.core.llm.provider import (
    StreamComplete,
    StreamContextCompacted,
    StreamTaskProgress,
    StreamTextDelta,
    StreamToolResult,
    StreamToolUseDelta,
    StreamToolUseEnd,
    StreamToolUseStart,
    StreamTurnEnded,
    ToolCall,
    damaged_tool_call_result,
)
from anton.core.mcp.client import McpSession
from anton.core.mcp.names import namespaced_tool_name
from anton.core.mcp.registry import discover_tool_defs
from anton.core.session import ChatSession, _discarded_tool_call_events
from anton.core.tools import tool_handlers, web_tools
from anton.core.tools.registry import ToolOutcome
from anton.core.utils.scratchpad import format_cell_result
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from cowork.harnesses.anton_harness import figure_sources as figure_sources_mod
from cowork.harnesses.anton_harness import stream_formatter as stream_formatter_mod
from cowork.harnesses.anton_harness.agent_files import AgentFiles
from cowork.harnesses.anton_harness.stream_formatter import TurnContext, TurnHistory, format_responses_stream
from cowork.schemas.figure_sources import EarlierAnswer, Figure, FigureOrigin
from cowork.turnqueue.producer import step_stream_events
from tests._hubspot_records import HUBSPOT_DEALS, SEARCH_DEALS


async def _events(*items):
    for item in items:
        yield item


def _parse_sse(chunks: list[str]) -> list[tuple[str, dict]]:
    parsed = []
    for chunk in chunks:
        event_type = ""
        for line in chunk.strip("\n").split("\n"):
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                parsed.append((event_type, json.loads(line[len("data:"):].strip())))
    return parsed


def _read_stock_cell(tool_use_id: str = "toolu_read") -> list:
    code = "print(open('/workspace/stock.csv').read())"
    cell = {
        "code": code, "stdout": "part,site,qty\nP100,S1,250\nP100,S2,90\n", "stderr": "", "error": None,
        "description": "Read stock.csv", "estimated_time": "", "logs": "",
    }
    return [
        StreamToolUseStart(id=tool_use_id, name="scratchpad"),
        StreamToolUseDelta(id=tool_use_id, json_delta=json.dumps({"action": "exec", "code": code})),
        StreamToolUseEnd(id=tool_use_id),
        StreamToolResult(name="scratchpad", content=json.dumps(cell), action="exec", id=tool_use_id),
    ]


async def _run(
    *items, question: str = "", files: AgentFiles | None = None, sink: list | None = None,
) -> list[tuple[str, dict]]:
    def event_sink(event_type: str, data: dict) -> None:
        if sink is not None:
            sink.append((event_type, data))

    return _parse_sse([
        chunk async for chunk in format_responses_stream(
            _events(TurnContext(question=question, files=files), *items), model="test-model", event_sink=event_sink,
        )
    ])


class TestFigureSourcesEvent:
    async def test_an_answer_with_figures_gets_one_event_just_before_completed(self):
        sink: list = []

        events = await _run(
            *_read_stock_cell(),
            StreamTextDelta(text="Site S1 holds 250 units; overtime costs $30."),
            question="How many units of P100 are at site S1?",
            sink=sink,
        )

        types = [event_type for event_type, _ in events]
        assert types[-2:] == ["response.figure_sources", "response.completed"]
        payload = events[-2][1]
        assert payload["item_id"] == events[-1][1]["response"]["output"][0]["id"]
        assert payload["project_id"] is None
        assert payload["steps"] == [{
            "tool_use_id": "toolu_read", "kind": "file", "label": "stock.csv",
            "path": None, "attachment_id": None, "url": None, "queries": [], "requests": [],
        }]
        by_text = {f["text"]: f for f in payload["figures"]}
        assert by_text["250"]["origin"] == "step"
        assert by_text["250"]["sources"] == [{"tool_use_id": "toolu_read", "page": None}]
        answer = "Site S1 holds 250 units; overtime costs $30."
        assert by_text["$30"] == {
            "start": answer.index("$30"), "end": answer.index("$30") + 3, "text": "$30", "origin": "none", "sources": [],
        }
        # Recorded like every other event, so it saves with the answer and
        # comes back on reload.
        assert ("response.figure_sources", payload) in sink

    async def test_a_turn_that_handed_back_gets_its_sources_before_the_incomplete_final_event(self):
        events = await _run(
            *_read_stock_cell(),
            StreamTextDelta(text="Site S1 holds 250 units."),
            StreamTurnEnded(ended_by="round_cap"),
        )

        assert [event_type for event_type, _ in events][-2:] == ["response.figure_sources", "response.completed"]
        assert [(f["text"], f["origin"]) for f in events[-2][1]["figures"]] == [("250", "step")]
        final = events[-1][1]["response"]
        assert final["status"] == "incomplete"
        assert final["incomplete_details"] == {"reason": "round_cap"}

    @pytest.mark.parametrize(("limit", "number"), [("round_cap", "5"), ("spend_ceiling", "1,250,000")])
    async def test_the_number_in_antons_limit_sentence_gets_no_mark(self, limit, number):
        # The installed anton writes the sentence, so a reword there fails
        # here. It reads only these four fields of the session.
        limits = SimpleNamespace(
            _max_tool_rounds=5, _round_cap_continued_rounds=0,
            _max_turn_tokens=1_250_000, _spend_ceiling_continued_tokens=0,
        )
        sentence = ChatSession._limit_reached(limits, limit)
        assert number in sentence

        events = await _run(
            *_read_stock_cell(),
            StreamTextDelta(text=f"{sentence}.\n\nSite S1 holds 250 units."),
            StreamTurnEnded(ended_by=limit),
        )

        assert [(f["text"], f["origin"]) for f in events[-2][1]["figures"]] == [("250", "step")]

    async def test_a_turn_that_handed_back_on_antons_failure_reply_gets_no_sources(self):
        events = await _run(
            *_read_stock_cell(),
            StreamTextDelta(text="An unexpected error occurred: Server returned 413 Payload Too Large"),
            StreamTurnEnded(ended_by="retry_exhausted"),
        )

        assert "response.figure_sources" not in [event_type for event_type, _ in events]
        assert events[-1][1]["response"]["incomplete_details"] == {"reason": "retry_exhausted"}

    async def test_a_file_the_pod_read_carries_its_path_in_the_project(self, tmp_path: Path):
        conversation_id = uuid4()
        workspace = tmp_path / "conversations" / str(conversation_id)
        workspace.mkdir(parents=True)
        (workspace / "stock.csv").write_text("part,site,qty\nP100,S1,250\n")

        events = await _run(
            *_read_stock_cell(),
            StreamTextDelta(text="Site S1 holds 250 units."),
            files=AgentFiles.in_cloud_pod(tmp_path, conversation_id=conversation_id, workspace_rel_path=None),
        )

        payload = next(data for event_type, data in events if event_type == "response.figure_sources")
        assert payload["steps"][0]["path"] == f"conversations/{conversation_id}/stock.csv"

    async def test_the_context_event_streams_nothing_and_marks_figures_the_question_states(self):
        sink: list = []

        events = await _run(
            StreamTextDelta(text="I found nothing for the next 3 weeks."),
            question="What ships in the next 3 weeks?",
            sink=sink,
        )

        assert [event_type for event_type, _ in events] == [
            "response.created", "response.output_text.delta", "response.figure_sources", "response.completed",
        ]
        assert [event_type for event_type, _ in sink] == [event_type for event_type, _ in events]
        assert events[2][1]["figures"][0]["origin"] == "question"

    async def test_an_answer_with_no_figures_streams_as_before(self):
        with_cell = await _run(*_read_stock_cell(), StreamTextDelta(text="The data doesn't hold that."))

        types = [event_type for event_type, _ in with_cell]
        assert "response.figure_sources" not in types
        assert types[-1] == "response.completed"
        assert [e["sequence_number"] for _, e in with_cell] == list(range(1, len(with_cell) + 1))

    async def test_the_figures_follow_a_continuation_that_replaced_the_answer(self):
        from anton.core.llm.provider import StreamTaskProgress

        events = await _run(
            *_read_stock_cell(),
            StreamTextDelta(text="Roughly 300 units."),
            StreamTaskProgress(phase="continuation", message=""),
            StreamTextDelta(text="S1 holds 250 units."),
        )

        payload = next(data for event_type, data in events if event_type == "response.figure_sources")
        assert [f["text"] for f in payload["figures"]] == ["250"]

    async def test_a_matcher_failure_leaves_the_answer_whole(self, monkeypatch, caplog):
        import cowork.harnesses.anton_harness.stream_formatter as formatter

        def broken(**_kwargs):
            raise ValueError("boom")

        monkeypatch.setattr(formatter, "find_figure_sources", broken)

        events = await _run(StreamTextDelta(text="S1 holds 250 units."))

        assert [event_type for event_type, _ in events][-1] == "response.completed"
        assert "response.figure_sources" not in [event_type for event_type, _ in events]
        assert events[-1][1]["response"]["output"][0]["content"][0]["text"] == "S1 holds 250 units."
        assert [
            r for r in caplog.records
            if r.name == formatter.logger.name and r.levelname == "ERROR"
        ]

    async def test_the_sources_are_found_off_the_event_loop(self, monkeypatch):
        import threading

        import cowork.harnesses.anton_harness.stream_formatter as formatter

        found_on: list[threading.Thread] = []
        find_figure_sources = formatter.find_figure_sources

        def recording(**kwargs):
            found_on.append(threading.current_thread())
            return find_figure_sources(**kwargs)

        monkeypatch.setattr(formatter, "find_figure_sources", recording)

        events = await _run(*_read_stock_cell(), StreamTextDelta(text="S1 holds 250 units."))

        assert found_on and found_on[0] is not threading.current_thread()
        assert "response.figure_sources" in [event_type for event_type, _ in events]


class TestFigureSourcesSurviveReload:
    async def test_the_saved_turn_returns_the_same_figures_and_their_text(self):
        from sqlmodel import Session

        from cowork.common.settings.app_settings import get_app_settings
        from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
        from cowork.db.session import get_engine
        from cowork.services.conversations import ConversationService
        from cowork.streaming.answer_text import accumulate_answer_text

        # The same capture the responses route's event_sink does.
        collected_text: list[str] = []
        collected_events: list[dict] = []
        sink: list = []
        streamed = await _run(
            *_read_stock_cell(),
            *_tool_call("native", "web_fetch", {"url": "https://example.com/rates"}),
            *_tool_done("native", None),
            StreamTextDelta(text="Site S1 holds 250 units; "),
            StreamTextDelta(text="overtime costs $30; status 404."),
            TurnHistory(_tool_rows("native", "web_fetch", {"url": "https://example.com/rates"}, "404")),
            sink=sink,
        )
        for event_type, data in sink:
            collected_events.append(data)
            accumulate_answer_text(collected_text, event_type, data)
        streamed_figures = next(data for event_type, data in streamed if event_type == "response.figure_sources")

        with Session(get_engine(get_app_settings().database.uri)) as session:
            service = ConversationService(ScopedSession(session, LOCAL_SCOPE))
            conversation = service.create_conversation(topic="t")
            service.save_assistant_turn(conversation.id, "".join(collected_text), collected_events, harness="anton")
            [assistant] = [m for m in service.get_messages(conversation.id) if m["role"] == "assistant"]

        saved = [e for e in assistant["events"] if e.get("type") == "response.figure_sources"]
        assert saved == [streamed_figures]
        types = [e.get("type") for e in assistant["events"]]
        assert types.index("response.figure_sources") == types.index("response.completed") - 1
        content = assistant["content"]
        assert [content[f["start"]:f["end"]] for f in saved[0]["figures"]] == ["250", "$30", "404"]
        assert _by_text(saved[0])["404"]["origin"] == "none"
        assert [step["tool_use_id"] for step in saved[0]["steps"]] == ["toolu_read"]


def _tool_rows(tool_use_id: str, name: str, tool_input: dict, result) -> list[dict]:
    """One tool call and its result as history rows hold them."""
    return [
        {"role": "assistant", "content": [{"type": "tool_use", "id": tool_use_id, "name": name, "input": tool_input}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": result}]},
    ]


def _refused_rows(call: ToolCall) -> list[dict]:
    """A call anton refused to run as its history keeps it: the call's
    arguments, and the result anton answered it with in the tool's place."""
    return [
        {"role": "assistant", "content": [{"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}]},
        {"role": "user", "content": [damaged_tool_call_result(call)]},
    ]


def _cell_rows(tool_use_id: str, *, code: str, stdout: str, logs: str = "", description: str = "") -> list[dict]:
    """A scratchpad cell that ran, as history rows hold it: anton hands the
    model each part of the result under a label (format_cell_result), and a
    cell that printed nothing as "Code executed successfully (no output).\""""
    tool_input = {"action": "exec", "name": "main", "code": code}
    if description:
        tool_input["one_line_description"] = description
    result = format_cell_result(Cell(code=code, stdout=stdout, stderr="", error=None, logs=logs))
    return _tool_rows(tool_use_id, "scratchpad", tool_input, result)


def _tool_call(tool_use_id: str, name: str, tool_input: dict) -> list:
    return [
        StreamToolUseStart(id=tool_use_id, name=name),
        StreamToolUseDelta(id=tool_use_id, json_delta=json.dumps(tool_input)),
        StreamToolUseEnd(id=tool_use_id),
    ]


def _tool_done(tool_use_id: str | None, ok: bool | None, *, cloud: bool = False) -> list:
    if cloud:
        return step_stream_events({"step": "progress", "phase": "tool_done", "id": tool_use_id, "ok": ok})
    return [StreamTaskProgress(phase="tool_done", message="", id=tool_use_id, ok=ok)]


async def _run_turn(context: TurnContext, *items) -> list[tuple[str, dict]]:
    return _parse_sse([
        chunk async for chunk in format_responses_stream(_events(context, *items), model="test-model")
    ])


def _figures(events: list[tuple[str, dict]]) -> dict:
    """The answer's figure_sources payload, or {} when the turn has none."""
    return next((data for event_type, data in events if event_type == "response.figure_sources"), {})


def _by_text(payload: dict) -> dict[str, dict]:
    return {figure["text"]: figure for figure in payload["figures"]}


class TestStepsFromTheTurnsToolRows:
    """anton streams a result only for the scratchpad. Every other tool's
    result reaches the formatter in the turn's tool rows."""

    async def test_a_figure_a_web_page_or_a_file_read_holds_names_that_step(self):
        rows = [
            *_tool_rows("toolu_web", "web_fetch", {"url": "https://example.com/rates"}, "Overtime rate: $42 per hour"),
            *_tool_rows(
                "toolu_file", "read_text_file", {"path": "/workspace/stock.csv"},
                [{"type": "text", "text": "part,site,qty\nP100,S1,250\n"}],
            ),
        ]

        events = await _run_turn(
            TurnContext(question="What does overtime cost, and what is at S1?"),
            *_tool_call("toolu_web", "web_fetch", {"url": "https://example.com/rates"}),
            *_tool_call("toolu_file", "read_text_file", {"path": "/workspace/stock.csv"}),
            *_tool_done("toolu_web", True),
            *_tool_done("toolu_file", True),
            StreamTextDelta(text="Overtime costs $42 an hour and S1 holds 250 units."),
            TurnHistory(rows),
        )

        payload = _figures(events)
        figures = _by_text(payload)
        assert figures["$42"]["sources"] == [{"tool_use_id": "toolu_web", "page": None}]
        assert figures["250"]["sources"] == [{"tool_use_id": "toolu_file", "page": None}]
        steps = {step["tool_use_id"]: step for step in payload["steps"]}
        assert (steps["toolu_web"]["kind"], steps["toolu_web"]["url"]) == ("web", "https://example.com/rates")
        assert (steps["toolu_file"]["kind"], steps["toolu_file"]["label"]) == ("file", "stock.csv")

    async def test_a_memory_file_read_is_not_a_step_and_a_project_file_read_is(self):
        memory = {"path": ".anton/memory/lessons.md"}
        stock = {"path": "data/stock.csv"}
        rows = [
            *_tool_rows("toolu_mem", "read_text_file", memory, "Overtime costs 30 dollars an hour.\n"),
            *_tool_rows("toolu_file", "read_text_file", stock, "part,site,qty\nP100,S1,250\n"),
        ]

        events = await _run_turn(
            TurnContext(question="What does overtime cost, and what is at S1?"),
            *_tool_call("toolu_mem", "read_text_file", memory),
            *_tool_call("toolu_file", "read_text_file", stock),
            *_tool_done("toolu_mem", True),
            *_tool_done("toolu_file", True),
            StreamTextDelta(text="Overtime costs $30 an hour and S1 holds 250 units."),
            TurnHistory(rows),
        )

        payload = _figures(events)
        figures = _by_text(payload)
        assert (figures["$30"]["origin"], figures["$30"]["sources"]) == ("none", [])
        assert figures["250"]["sources"] == [{"tool_use_id": "toolu_file", "page": None}]
        assert [step["tool_use_id"] for step in payload["steps"]] == ["toolu_file"]

    async def test_a_streamed_scratchpad_result_stays_one_step(self):
        events = await _run_turn(
            TurnContext(question="What is at S1?"),
            *_read_stock_cell(),
            StreamTextDelta(text="S1 holds 250 units."),
            TurnHistory(_cell_rows(
                "toolu_read", code="print(open('/workspace/stock.csv').read())", stdout="part,site,qty\nP100,S1,250\n",
            )),
        )

        assert [step["tool_use_id"] for step in _figures(events)["steps"]] == ["toolu_read"]

    async def test_the_users_answer_to_the_agents_question_is_not_a_step(self):
        events = await _run_turn(
            TurnContext(question="What does overtime cost?"),
            *_tool_call("toolu_ask", "ask_user", {"prompt": "Which rate?"}),
            StreamTextDelta(text="At $42 an hour."),
            TurnHistory(_tool_rows("toolu_ask", "ask_user", {"prompt": "Which rate?"}, "The user answered: $42")),
        )

        assert _figures(events)["steps"] == []

    @pytest.mark.parametrize(("name", "tool_input", "result"), [
        (
            "recall_history", {"query": "overtime rate"},
            "--- turn 2 ---\n[user] What does overtime cost?\n[assistant] Overtime typically costs about $30 an hour.",
        ),
        (
            "open_artifact", {"slug": "overtime-plan"},
            json.dumps({
                "path": "/p/.anton/artifacts/overtime-plan", "files": ["report.md"],
                "primary_content": "# Plan\nOvertime at $30 an hour for 12 people.",
            }),
        ),
    ])
    async def test_the_agents_own_earlier_writing_is_not_a_step(self, name: str, tool_input: dict, result: str):
        events = await _run_turn(
            TurnContext(question="What did we say overtime costs?"),
            *_tool_call("toolu_own", name, tool_input),
            StreamTextDelta(text="Overtime costs $30 an hour."),
            TurnHistory(_tool_rows("toolu_own", name, tool_input, result)),
        )

        payload = _figures(events)
        assert _by_text(payload)["$30"]["origin"] == "none"
        assert payload["steps"] == []

    async def test_a_cell_only_the_rows_carry_is_read_by_what_it_printed(self):
        rows = _cell_rows(
            "toolu_rows", code="print(open('/workspace/stock.csv').read())", stdout="part,site,qty\nP100,S1,250\n",
            logs="HTTP Request: GET /stock 200 OK",
        )

        events = await _run_turn(
            TurnContext(question="What is at S1?"),
            StreamTextDelta(text="S1 holds 250 units; the API answered 200."),
            TurnHistory(rows),
        )

        figures = _by_text(_figures(events))
        assert figures["250"]["sources"] == [{"tool_use_id": "toolu_rows", "page": None}]
        assert figures["200"]["origin"] == "none"

    async def test_a_replayed_cells_artifact_lint_is_not_what_it_printed(self):
        """anton's session appends [artifact lint] to the result of a cell
        that changed an artifact."""
        code = "print(open('/workspace/stock.csv').read())"
        result = (
            format_cell_result(Cell(code=code, stdout="P100,S1,250\n", stderr="", error=None))
            + "\n\n[artifact lint]\nsales-report/index.html, line 42: unclosed <div>"
        )

        events = await _run_turn(
            TurnContext(question="What is at S1?"),
            StreamTextDelta(text="S1 holds 250 units; line 42 needs a fix."),
            TurnHistory(_tool_rows("toolu_rows", "scratchpad", {"action": "exec", "name": "main", "code": code}, result)),
        )

        figures = _by_text(_figures(events))
        assert figures["250"]["sources"] == [{"tool_use_id": "toolu_rows", "page": None}]
        assert figures["42"]["origin"] == "none"

    async def test_a_replayed_cells_live_app_lines_are_not_what_it_printed(self):
        """With live apps on, anton appends [live app] to the result of a cell
        that relaunched a running app."""
        code = "print(open('/workspace/stock.csv').read())"
        result = (
            format_cell_result(Cell(code=code, stdout="P100,S1,250\n", stderr="", error=None))
            + "\n\n[live app]\nsales-dash relaunched in slot 7"
        )

        events = await _run_turn(
            TurnContext(question="What is at S1?"),
            StreamTextDelta(text="S1 holds 250 units; the app runs in slot 7."),
            TurnHistory(_tool_rows("toolu_rows", "scratchpad", {"action": "exec", "name": "main", "code": code}, result)),
        )

        figures = _by_text(_figures(events))
        assert figures["250"]["sources"] == [{"tool_use_id": "toolu_rows", "page": None}]
        assert figures["7"]["origin"] == "none"

    @pytest.mark.parametrize(("logs", "lint"), [
        ("", ""),
        ("HTTP Request: GET /stock 200 OK", ""),
        ("", "\n\n[artifact lint]\nsales-report/index.html, line 42: unclosed <div>"),
    ], ids=["alone", "before-logs", "before-lint"])
    async def test_a_replayed_cells_truncation_note_is_not_what_it_printed(self, logs: str, lint: str):
        """format_cell_result cuts printed output past 10,000 characters and
        notes how long it was."""
        stdout = "P100,S1,250\n" + "x" * 10_100
        code = "print(dump)"
        result = format_cell_result(Cell(code=code, stdout=stdout, stderr="", error=None, logs=logs)) + lint

        events = await _run_turn(
            TurnContext(question="What is at S1?"),
            StreamTextDelta(text=f"S1 holds 250 units; the dump ran {len(stdout)} characters."),
            TurnHistory(_tool_rows("toolu_rows", "scratchpad", {"action": "exec", "name": "main", "code": code}, result)),
        )

        figures = _by_text(_figures(events))
        assert figures["250"]["origin"] == "step"
        assert figures[str(len(stdout))]["origin"] == "none"

    async def test_a_web_call_anton_refused_is_not_a_step(self):
        """anton answers a call whose arguments it could not read with an
        is_error result of its own, which quotes the parse error."""
        refused = ToolCall(
            id="toolu_w", name="web_fetch", input={}, parse_error="Expecting ',' delimiter: line 1 column 523 (char 522)",
        )

        events = await _run_turn(
            TurnContext(question="How long is the page?"),
            *_tool_call("toolu_w", "web_fetch", {}),
            StreamTextDelta(text="The page is 523 words."),
            TurnHistory(_refused_rows(refused)),
        )

        payload = _figures(events)
        assert _by_text(payload)["523"]["origin"] == "none"
        assert payload["steps"] == []

    @pytest.mark.parametrize("compacted", [
        [StreamContextCompacted(message="Context compacted")],
        step_stream_events({"step": "compacted", "message": "Context compacted"}),
    ], ids=["in-process", "cloud-pod"])
    @pytest.mark.parametrize("name", ["web_search", "web_fetch", "read_text_file"])
    async def test_a_compacted_turn_with_no_tool_rows_leaves_a_figure_no_step_holds_unmarked(
        self, name: str, compacted: list,
    ):
        """A producer whose anton predates ChatSession.turn_messages sends no
        tool rows for a turn that compacts mid-way, so the result of a tool
        anton runs itself is missing."""
        events = await _run_turn(
            TurnContext(question="What does overtime cost, and what is at S1?"),
            *_read_stock_cell(),
            *_tool_call("toolu_evidence", name, {"query": "overtime rate", "url": "https://example.com/rates"}),
            *_tool_done("toolu_evidence", True),
            *compacted,
            StreamTextDelta(text="Overtime costs $42 an hour and S1 holds 250 units."),
        )

        figures = _by_text(_figures(events))
        assert "$42" not in figures
        assert figures["250"]["sources"] == [{"tool_use_id": "toolu_read", "page": None}]

    @pytest.mark.parametrize("verdict", [
        _tool_done("toolu_web", False),
        _tool_done("toolu_web", None),
        [],
    ], ids=["failed", "no-verdict", "no-tool-done"])
    async def test_a_compacted_turn_with_no_tool_rows_keeps_the_mark_when_its_web_call_did_not_succeed(
        self, verdict: list,
    ):
        """The matcher counts a call of a tool anton runs itself as a step
        only when its tool_done said ok=True, so the missing result of a call
        that failed, gave no verdict or never ran hides no source."""
        events = await _run_turn(
            TurnContext(question="What does overtime cost?"),
            *_tool_call("toolu_web", "web_fetch", {"url": "https://example.com/rates"}),
            *verdict,
            StreamContextCompacted(message="Context compacted"),
            StreamTextDelta(text="Overtime costs $42 an hour and 17 shifts."),
        )

        assert {text: figure["origin"] for text, figure in _by_text(_figures(events)).items()} == {
            "$42": "none", "17": "none",
        }

    @pytest.mark.parametrize("compacted", [
        [StreamContextCompacted(message="Context compacted")],
        step_stream_events({"step": "compacted", "message": "Context compacted"}),
    ], ids=["in-process", "cloud-pod"])
    async def test_a_compacted_turns_tool_rows_give_a_figure_no_step_holds_no_source(self, compacted: list):
        """A turn's tool rows hold every call that ran, through a compaction
        too, so a call they miss never ran: a figure no step holds has no
        source."""
        events = await _run_turn(
            TurnContext(question="What does overtime cost, and what is at S1?"),
            *_tool_call("toolu_file", "read_text_file", {"path": "/workspace/stock.csv"}),
            *_tool_call("toolu_web", "web_fetch", {"url": "https://example.com/rates"}),
            *_tool_done("toolu_file", True),
            *compacted,
            StreamTextDelta(text="Overtime costs $42 an hour and S1 holds 250 units."),
            TurnHistory(rows=[], read=_tool_rows(
                "toolu_file", "read_text_file", {"path": "/workspace/stock.csv"}, "part,site,qty\nP100,S1,250\n",
            )),
        )

        figures = _by_text(_figures(events))
        assert figures["$42"]["origin"] == "none"
        assert figures["250"]["sources"] == [{"tool_use_id": "toolu_file", "page": None}]

    @pytest.mark.parametrize(("path", "origins"), [
        (".anton/memory/lessons.md", {"$30": "none", "250": "step"}),
        ("docs/lessons.md", {"250": "step"}),
    ], ids=["memory-file", "project-file"])
    async def test_a_compacted_turn_with_no_tool_rows_that_read_only_a_memory_file_keeps_the_mark(
        self, path: str, origins: dict[str, str],
    ):
        """A memory file read is never a step, so its missing row hides no
        source. A missing read of any other file still may."""
        lessons = {"path": path}

        events = await _run_turn(
            TurnContext(question="What does overtime cost, and what is at S1?"),
            *_read_stock_cell(),
            *_tool_call("toolu_lessons", "read_text_file", lessons),
            *_tool_done("toolu_lessons", True),
            StreamContextCompacted(message="Context compacted"),
            StreamTextDelta(text="Overtime costs $30 an hour and S1 holds 250 units."),
        )

        assert {text: figure["origin"] for text, figure in _by_text(_figures(events)).items()} == origins

    async def test_a_round_anton_threw_away_keeps_the_mark_on_a_figure_no_step_holds(self):
        """A round cut at max_tokens is thrown away whole and retried: none of
        its calls run or reach the rows, the intact ones included."""
        thrown_away = [
            ToolCall(id="toolu_web_cut", name="web_fetch", input={"url": "https://example.com/rates"}),
            ToolCall(id="toolu_cell_cut", name="scratchpad", input={"action": "exec", "name": "main"}, repaired=True),
        ]

        events = await _run_turn(
            TurnContext(question="What is at S1?"),
            *_tool_call("toolu_web_cut", "web_fetch", {"url": "https://example.com/rates"}),
            StreamToolUseStart(id="toolu_cell_cut", name="scratchpad"),
            *(event for call in thrown_away for event in _discarded_tool_call_events(call)),
            *_read_stock_cell(),
            StreamTextDelta(text="S1 holds 250 units, and overtime costs $42 an hour."),
            TurnHistory(_cell_rows(
                "toolu_read", code="print(open('/workspace/stock.csv').read())",
                stdout="part,site,qty\nP100,S1,250\nP100,S2,90\n",
            )),
        )

        figures = _by_text(_figures(events))
        assert figures["$42"]["origin"] == "none"
        assert figures["250"]["sources"] == [{"tool_use_id": "toolu_read", "page": None}]

    async def test_a_number_a_replayed_web_calls_query_states_is_only_echoed_by_its_results(self):
        """The row's call reads with the text of its arguments, so a number
        the query states and the results repeat is an echo, not a figure the
        search found."""
        query = {"query": "overtime rates 2024"}

        events = await _run_turn(
            TurnContext(question="What did overtime cost?"),
            *_tool_call("toolu_search", "web_search", query),
            *_tool_done("toolu_search", True),
            StreamTextDelta(text="In 2024 overtime was $42."),
            TurnHistory(_tool_rows("toolu_search", "web_search", query, "2024 overtime: $42 per hour")),
        )

        figures = _by_text(_figures(events))
        assert figures["2024"]["origin"] == "none"
        assert figures["$42"]["sources"] == [{"tool_use_id": "toolu_search", "page": None}]

    async def test_a_replayed_calls_arguments_are_not_parsed_again(self, monkeypatch):
        """A history row holds a call's arguments already parsed, so neither
        the web call nor the cell that only the rows hold goes back through
        the text parser."""
        def never(_raw):
            raise AssertionError("a replayed call's arguments were parsed from text")

        monkeypatch.setattr(figure_sources_mod, "_leading_object", never)
        rows = [
            *_tool_rows("toolu_web", "web_fetch", {"url": "https://example.com/rates"}, "Overtime rate: $42 per hour"),
            *_cell_rows("toolu_cell", code="print(5000 + 150)", stdout="5150\n", description="Count rows"),
        ]

        events = await _run_turn(
            TurnContext(question="What does overtime cost?"),
            *_tool_call("toolu_web", "web_fetch", {"url": "https://example.com/rates"}),
            *_tool_done("toolu_web", True),
            StreamTextDelta(text="Overtime costs $42 an hour across 5150 rows."),
            TurnHistory(rows),
        )

        assert "response.figure_sources" in [event_type for event_type, _ in events]
        figures = _by_text(_figures(events))
        assert figures["$42"]["sources"] == [{"tool_use_id": "toolu_web", "page": None}]
        assert figures["5150"]["sources"] == [{"tool_use_id": "toolu_cell", "page": None}]


class TestNativeEvidenceVerdicts:
    @pytest.mark.parametrize("cloud", [False, True], ids=["in-process", "cloud-pod"])
    @pytest.mark.parametrize("name", ["web_fetch", "web_search", "read_text_file"])
    @pytest.mark.parametrize("verdict", [True, False, None, "absent"])
    async def test_only_explicit_success_can_source_a_native_result(self, cloud, name, verdict, monkeypatch):
        # No tool_progress precedes tool_done, and the preceding ordinary
        # progress occupies the throttle window. Neither can lose a verdict.
        monkeypatch.setattr(stream_formatter_mod.time, "time", lambda: 1000.0)
        tool_input = {"url": "https://example.com/rates", "query": "rates", "path": "/workspace/rates.txt"}
        done = [] if verdict == "absent" else _tool_done("native", verdict, cloud=cloud)

        events = await _run(
            *_tool_call("native", name, tool_input),
            StreamTaskProgress(phase="analyzing", message="Working"),
            *done,
            StreamTextDelta(text="There are 42 units."),
            TurnHistory(_tool_rows("native", name, tool_input, "42 units")),
        )

        figure = _by_text(_figures(events))["42"]
        assert figure["origin"] == ("step" if verdict is True else "none")
        assert figure["sources"] == ([{"tool_use_id": "native", "page": None}] if verdict is True else [])

    @pytest.mark.parametrize("cloud", [False, True], ids=["in-process", "cloud-pod"])
    @pytest.mark.parametrize("tool_use_id", [None, "", "another-call"])
    async def test_a_verdict_without_the_matching_id_cannot_source_a_result(self, cloud, tool_use_id):
        events = await _run(
            *_tool_call("native", "web_fetch", {"url": "https://example.com/rates"}),
            *_tool_done(tool_use_id, True, cloud=cloud),
            StreamTextDelta(text="There are 42 units."),
            TurnHistory(_tool_rows("native", "web_fetch", {"url": "https://example.com/rates"}, "42 units")),
        )

        assert _by_text(_figures(events))["42"]["origin"] == "none"

    async def test_a_cloud_verdict_with_no_ok_field_cannot_source_a_result(self):
        events = await _run(
            *_tool_call("native", "web_fetch", {"url": "https://example.com/rates"}),
            *step_stream_events({"step": "progress", "phase": "tool_done", "id": "native"}),
            StreamTextDelta(text="There are 42 units."),
            TurnHistory(_tool_rows("native", "web_fetch", {"url": "https://example.com/rates"}, "42 units")),
        )

        assert _by_text(_figures(events))["42"]["origin"] == "none"

    @pytest.mark.parametrize("status", [200, 404])
    async def test_the_locked_web_producer_distinguishes_content_from_an_http_error(self, status, monkeypatch):
        # Uses uv.lock's Anton web producer, including its generated HTTP
        # error and ToolOutcome. No external network or copied error string.
        client = web_tools.httpx.AsyncClient
        transport = web_tools.httpx.MockTransport(
            lambda request: web_tools.httpx.Response(status, text="42 failed shipments.", request=request),
        )
        monkeypatch.setattr(web_tools, "_check_url_ssrf", lambda _url: None)
        monkeypatch.setattr(web_tools.httpx, "AsyncClient", lambda **kwargs: client(transport=transport, **kwargs))
        tool_input = {"url": "https://example.com/shipments"}
        outcome = await web_tools.handle_web_fetch_fallback(None, tool_input)
        assert isinstance(outcome, ToolOutcome)
        assert outcome.ok is (True if status == 200 else None)
        figure_text = "42" if status == 200 else "404"

        events = await _run(
            *_tool_call("native", "web_fetch", tool_input),
            *_tool_done("native", outcome.ok),
            StreamTextDelta(text=f"There are {figure_text} shipments."),
            TurnHistory(_tool_rows("native", "web_fetch", tool_input, outcome.content)),
        )

        assert _by_text(_figures(events))[figure_text]["origin"] == ("step" if status == 200 else "none")

    async def test_a_native_file_error_cannot_source_its_line_count(self, tmp_path):
        (tmp_path / "stock.csv").write_text("".join(f"row{i}\n" for i in range(42)))
        tool_input = {"path": "stock.csv", "start_line": 1000}
        outcome = await tool_handlers.handle_read_text_file(
            SimpleNamespace(_workspace=SimpleNamespace(base=tmp_path)), tool_input,
        )
        assert outcome.ok is False
        assert "(42 lines)" in outcome.content

        events = await _run(
            *_tool_call("native", "read_text_file", tool_input),
            *_tool_done("native", outcome.ok),
            StreamTextDelta(text="There are 42 units."),
            TurnHistory(_tool_rows("native", "read_text_file", tool_input, outcome.content)),
        )

        assert _by_text(_figures(events))["42"]["origin"] == "none"
        assert _figures(events)["steps"] == []

    async def test_a_repaired_call_anton_refused_never_becomes_a_source(self):
        call = ToolCall(id="native", name="web_fetch", input={"max_chars": 523}, repaired=True)
        events = await _run(
            *_tool_call(call.id, call.name, call.input),
            *_discarded_tool_call_events(call),
            StreamTextDelta(text="There are 523 words."),
            TurnHistory(_refused_rows(call)),
        )

        assert _by_text(_figures(events))["523"]["origin"] == "none"
        assert _figures(events)["steps"] == []

    async def test_a_later_failed_verdict_revokes_an_earlier_success(self):
        call = ToolCall(id="native", name="web_fetch", input={"url": "https://example.com/rates"})
        events = await _run(
            *_tool_call(call.id, call.name, call.input),
            *_tool_done(call.id, True),
            *_discarded_tool_call_events(call),
            StreamTextDelta(text="There are 42 units."),
            TurnHistory(_tool_rows(call.id, call.name, call.input, "42 units")),
        )

        assert _by_text(_figures(events))["42"]["origin"] == "none"

    @pytest.mark.parametrize("cloud", [False, True], ids=["in-process", "cloud-pod"])
    async def test_partial_scratchpad_stdout_survives_a_failed_cell(self, cloud):
        code = "print(open('/workspace/stock.csv').read()); raise ValueError('failed')"
        cell = {"code": code, "stdout": "250 units", "stderr": "", "error": "ValueError: failed"}
        done = (
            step_stream_events({"step": "progress", "phase": "scratchpad_done", "id": "cell", "ok": False})
            if cloud else [StreamTaskProgress(phase="scratchpad_done", message="Failed", id="cell", ok=False)]
        )
        events = await _run(
            *_tool_call("cell", "scratchpad", {"action": "exec", "code": code}),
            *done,
            StreamToolResult(name="scratchpad", action="exec", id="cell", content=json.dumps(cell)),
            StreamTextDelta(text="There are 250 units."),
        )

        assert _by_text(_figures(events))["250"]["sources"] == [{"tool_use_id": "cell", "page": None}]


def _hubspot_server(*, search_error: str | None) -> MCPServer:
    """An in-process MCP server with two of HubSpot's tools, under their
    HubSpot names. Its search fails with ``search_error`` when one is given,
    the way HubSpot's server reports an API error inside the tool's result."""
    server = MCPServer(name="hubspot")

    @server.tool()
    def search_crm_objects(objectType: str, limit: int = 10) -> str:
        if search_error is not None:
            raise ToolError(search_error)
        return HUBSPOT_DEALS

    @server.tool()
    def tool_guidance() -> str:
        return "search_crm_objects returns up to 200 records per call."

    return server


async def _connector_call(
    tool: str, tool_input: dict, *, connection_name: str | None = None, search_error: str | None = None,
) -> tuple[str, ToolOutcome]:
    """A read-mode HubSpot connection's call of ``tool`` through anton's
    connector wiring: the name anton registers the tool under, and the
    outcome its handler returns."""
    async with McpSession(server=_hubspot_server(search_error=search_error)) as session:
        tool_defs = await discover_tool_defs(
            session, engine="hubspot", access_mode="read", connection_name=connection_name,
        )
        [tool_def] = [tool_def for tool_def in tool_defs if tool_def.name.endswith(f"__{tool}")]
        return tool_def.name, await tool_def.handler(None, tool_input)


class TestConnectorToolVerdicts:
    """A connector tool's result reaches the formatter only in the turn's tool
    rows, which hold a failed call's error as an ordinary result. Only the
    call's tool_done verdict tells the two apart."""

    @pytest.mark.parametrize("cloud", [False, True], ids=["in-process", "cloud-pod"])
    @pytest.mark.parametrize(("connection_name", "label"), [
        (None, "hubspot: search_crm_objects"),
        ("crm2", "hubspot (crm2): search_crm_objects"),
    ], ids=["one-connection", "two-connections"])
    async def test_a_connector_call_that_returned_records_sources_their_figures(
        self, cloud: bool, connection_name: str | None, label: str,
    ):
        name, outcome = await _connector_call("search_crm_objects", SEARCH_DEALS, connection_name=connection_name)
        assert outcome.ok is True

        events = await _run(
            *_tool_call("toolu_hs", name, SEARCH_DEALS),
            *_tool_done("toolu_hs", outcome.ok, cloud=cloud),
            StreamTextDelta(text="Acme's deal is $48,250."),
            TurnHistory(_tool_rows("toolu_hs", name, SEARCH_DEALS, outcome.content)),
        )

        payload = _figures(events)
        assert _by_text(payload)["$48,250"]["sources"] == [{"tool_use_id": "toolu_hs", "page": None}]
        assert [(step["kind"], step["label"]) for step in payload["steps"]] == [("query", label)]

    async def test_a_connector_call_that_failed_is_not_a_source(self):
        name, outcome = await _connector_call(
            "search_crm_objects", SEARCH_DEALS, search_error="HubSpot API request failed with status 500",
        )
        assert outcome.ok is False
        assert "500" in outcome.content

        events = await _run(
            *_tool_call("toolu_hs", name, SEARCH_DEALS),
            *_tool_done("toolu_hs", outcome.ok),
            StreamTextDelta(text="Your pipeline has 500 open deals."),
            TurnHistory(_tool_rows("toolu_hs", name, SEARCH_DEALS, outcome.content)),
        )

        assert _by_text(_figures(events))["500"]["origin"] == "none"
        assert _figures(events)["steps"] == []

    @pytest.mark.parametrize("cloud", [False, True], ids=["in-process", "cloud-pod"])
    @pytest.mark.parametrize("verdict", [None, "absent"])
    async def test_a_connector_call_without_a_success_verdict_is_not_a_source(self, cloud: bool, verdict):
        name, outcome = await _connector_call("search_crm_objects", SEARCH_DEALS)
        done = [] if verdict == "absent" else _tool_done("toolu_hs", verdict, cloud=cloud)

        events = await _run(
            *_tool_call("toolu_hs", name, SEARCH_DEALS),
            *done,
            StreamTextDelta(text="Acme's deal is $48,250."),
            TurnHistory(_tool_rows("toolu_hs", name, SEARCH_DEALS, outcome.content)),
        )

        assert _by_text(_figures(events))["$48,250"]["origin"] == "none"

    @pytest.mark.parametrize(("tool", "origins"), [
        ("search_crm_objects", {"250": "step"}),
        ("tool_guidance", {"$48,250": "none", "250": "step"}),
    ], ids=["records", "docs"])
    async def test_a_compacted_turn_with_no_tool_rows_hides_only_a_records_calls_figures(
        self, tool: str, origins: dict[str, str],
    ):
        """A producer whose anton predates ChatSession.turn_messages sends no
        tool rows for a turn that compacts mid-way. The missing result of a
        call that returns records may hold a figure no step holds; a call
        that returns HubSpot's docs holds none."""
        events = await _run(
            *_read_stock_cell(),
            *_tool_call("toolu_hs", namespaced_tool_name("hubspot", tool), {}),
            *_tool_done("toolu_hs", True),
            StreamContextCompacted(message="Context compacted"),
            StreamTextDelta(text="Acme's deal is $48,250 and S1 holds 250 units."),
        )

        assert {text: figure["origin"] for text, figure in _by_text(_figures(events)).items()} == origins


def _earlier_cell(tool_use_id: str = "toolu_earlier") -> list[dict]:
    """An earlier turn's cell as replayed history holds it."""
    return [
        {"role": "user", "content": "[2026-10-01 09:00] What is at S1 and S2?"},
        *_cell_rows(
            tool_use_id, code="print(open('/workspace/stock.csv').read())", stdout="part,site,qty\nP100,S1,250\nP100,S2,90\n",
            logs="HTTP Request: GET /stock 200 OK", description="Read stock.csv",
        ),
        {"role": "assistant", "content": "S1 holds 250 units and S2 holds 90."},
    ]


class TestStepsFromEarlierTurns:
    async def test_a_figure_an_earlier_turn_printed_has_no_source_when_no_step_of_this_turn_holds_it(self):
        """Turn 1 printed a stock table that happens to hold 30, 360 and 12.
        Turn 2 runs no step and works out an overtime cost."""
        quantities = [250, 90, 30, 410, 75, 12, 360, 88, 140, 55, 64, 300, 18, 220, 95, 47, 130, 66, 210, 150]
        table = "part,site,qty\n" + "\n".join(f"P{100 + i},S{1 + i % 3},{q}" for i, q in enumerate(quantities))
        history = [
            {"role": "user", "content": "[2026-10-01 09:00] What is in stock?"},
            *_cell_rows("toolu_t1", code="print(open('stock.csv').read())", stdout=table, description="Read stock.csv"),
            {"role": "assistant", "content": "Here is the stock by part."},
        ]

        events = await _run_turn(
            TurnContext(question="What would 12 hours of overtime cost?", history=history),
            StreamTextDelta(text="Overtime typically costs $30 an hour, so 12 hours cost $360."),
        )

        payload = _figures(events)
        assert [(f["text"], f["origin"], f["sources"]) for f in payload["figures"]] == [
            ("$30", "none", []), ("12", "question", []), ("$360", "none", []),
        ]
        assert payload["steps"] == []

    async def test_a_call_anton_discarded_in_an_earlier_turn_binds_no_names(self):
        spell = {"action": "exec", "name": "main", "code": 'base_sql = "SELECT qty FROM stock"'}
        history = _refused_rows(ToolCall(id="toolu_cut", name="scratchpad", input=spell, repaired=True))
        code = "print(query_minds_data(base_sql))"
        cell = {"code": code, "stdout": "250\n", "stderr": "", "error": None, "description": ""}

        events = await _run_turn(
            TurnContext(question="How many?", history=history),
            StreamToolUseStart(id="toolu_q", name="scratchpad"),
            StreamToolUseDelta(id="toolu_q", json_delta=json.dumps({"action": "exec", "name": "main", "code": code})),
            StreamToolUseEnd(id="toolu_q"),
            StreamToolResult(name="scratchpad", content=json.dumps(cell), action="exec", id="toolu_q"),
            StreamTextDelta(text="S1 holds 250 units."),
        )

        [step] = _figures(events)["steps"]
        assert (step["tool_use_id"], step["queries"]) == ("toolu_q", [])

    async def test_a_figure_this_turn_and_an_earlier_turn_printed_names_only_this_turns_step(self):
        events = await _run_turn(
            TurnContext(question="Check S1 again.", history=_earlier_cell()),
            *_read_stock_cell(),
            StreamTextDelta(text="S1 still holds 250 units."),
        )

        payload = _figures(events)
        assert _by_text(payload)["250"]["sources"] == [{"tool_use_id": "toolu_read", "page": None}]
        assert [step["tool_use_id"] for step in payload["steps"]] == ["toolu_read"]

    async def test_a_name_an_earlier_turns_cell_bound_reaches_this_turns_cell(self):
        history = _cell_rows("toolu_load", code="df = pd.read_csv('stock.csv')", stdout="")
        code = "print(df.qty.sum())"
        cell = {"code": code, "stdout": "340\n", "stderr": "", "error": None, "description": ""}

        events = await _run_turn(
            TurnContext(question="And in all?", history=history),
            StreamToolUseStart(id="toolu_sum", name="scratchpad"),
            StreamToolUseDelta(id="toolu_sum", json_delta=json.dumps({"action": "exec", "name": "main", "code": code})),
            StreamToolUseEnd(id="toolu_sum"),
            StreamToolResult(name="scratchpad", content=json.dumps(cell), action="exec", id="toolu_sum"),
            StreamTextDelta(text="340 units in all."),
        )

        payload = _figures(events)
        assert _by_text(payload)["340"]["sources"] == [
            {"tool_use_id": "toolu_sum", "page": None}, {"tool_use_id": "toolu_load", "page": None},
        ]
        assert [(step["tool_use_id"], step["kind"], step["label"]) for step in payload["steps"]] == [
            ("toolu_load", "file", "stock.csv"), ("toolu_sum", "step", ""),
        ]

    async def test_an_earlier_turns_row_with_no_id_names_no_step(self):
        """Stored conversations can hold tool calls with an empty id."""
        history = [
            {"role": "user", "content": "Load the stock."},
            *_cell_rows("", code="import pandas as pd\ndf = pd.read_csv('stock.csv')", stdout=""),
            {"role": "assistant", "content": "Loaded."},
        ]
        code = "print(df.qty.sum())"
        cell = {"code": code, "stdout": "340\n", "stderr": "", "error": None, "description": ""}

        events = await _run_turn(
            TurnContext(question="And in all?", history=history),
            StreamToolUseStart(id="toolu_b", name="scratchpad"),
            StreamToolUseDelta(id="toolu_b", json_delta=json.dumps({"action": "exec", "name": "main", "code": code})),
            StreamToolUseEnd(id="toolu_b"),
            StreamToolResult(name="scratchpad", content=json.dumps(cell), action="exec", id="toolu_b"),
            StreamTextDelta(text="340 units in all."),
        )

        payload = _figures(events)
        assert [step["tool_use_id"] for step in payload["steps"]] == ["toolu_b"]
        assert [(f["text"], f["origin"], [s["tool_use_id"] for s in f["sources"]]) for f in payload["figures"]] == [
            ("340", "step", ["toolu_b"]),
        ]

    async def test_an_answer_with_no_figures_reads_no_history(self, monkeypatch):
        """Nor does it parse the rows or the arguments the stream carried."""
        read: list = []
        parsed: list[str] = []
        read_rows = stream_formatter_mod.ToolRows.read
        from_text = figure_sources_mod.ToolInput.from_text

        def recording_rows(*, rows):
            read.append(rows)
            return read_rows(rows=rows)

        def recording_text(*, text):
            parsed.append(text)
            return from_text(text=text)

        monkeypatch.setattr(stream_formatter_mod.ToolRows, "read", staticmethod(recording_rows))
        monkeypatch.setattr(figure_sources_mod.ToolInput, "from_text", staticmethod(recording_text))

        events = await _run_turn(
            TurnContext(question="Plot it.", history=_earlier_cell()),
            *_read_stock_cell(),
            StreamTextDelta(text="Here is the chart."),
            TurnHistory(_cell_rows("toolu_read", code="print(open('/workspace/stock.csv').read())", stdout="")),
        )

        assert read == []
        assert parsed == []
        assert "response.figure_sources" not in [event_type for event_type, _ in events]

    async def test_a_turn_that_ran_no_cell_reads_no_history(self, monkeypatch):
        read: list[str] = []
        read_cell = figure_sources_mod.read_cell

        def recording(code, **kwargs):
            read.append(code)
            return read_cell(code, **kwargs)

        monkeypatch.setattr(figure_sources_mod, "read_cell", recording)

        events = await _run_turn(
            TurnContext(question="What was at S1?", history=_earlier_cell()), StreamTextDelta(text="S1 holds 250 units."),
        )

        assert read == []
        assert _by_text(_figures(events))["250"]["origin"] == "none"


def _earlier_answer(*figures: tuple[str, FigureOrigin]) -> EarlierAnswer:
    """An earlier answer whose saved event lists these figures."""
    return EarlierAnswer(figures=[
        Figure(start=0, end=len(text), text=text, origin=origin) for text, origin in figures
    ])


class TestFiguresRepeatedFromEarlierAnswers:
    async def test_the_event_marks_a_repeated_figure_and_names_no_step_for_it(self):
        events = await _run_turn(
            TurnContext(question="And now?", earlier_answers=(_earlier_answer(("4,600", FigureOrigin.step)),)),
            StreamTextDelta(text="Up from 4,600."),
        )

        payload = _figures(events)
        assert payload["figures"] == [
            {"start": 8, "end": 13, "text": "4,600", "origin": "earlier_answer", "sources": []},
        ]
        assert payload["steps"] == []

    async def test_a_compacted_turn_keeps_a_repeated_figures_mark(self):
        """The turn compacted and no tool rows arrived for its web call, so a
        figure no step holds is left out, but one an earlier answer gave keeps
        its mark."""
        events = await _run_turn(
            TurnContext(
                question="What does overtime cost?", earlier_answers=(_earlier_answer(("$42", FigureOrigin.step)),),
            ),
            *_tool_call("toolu_web", "web_fetch", {"url": "https://example.com/rates"}),
            *_tool_done("toolu_web", True),
            StreamContextCompacted(message="Context compacted"),
            StreamTextDelta(text="Overtime costs $42 an hour over 7 shifts."),
        )

        assert {text: figure["origin"] for text, figure in _by_text(_figures(events)).items()} == {
            "$42": "earlier_answer",
        }

    async def test_a_failure_marking_repeated_figures_keeps_this_turns_sources(self, monkeypatch, caplog):
        def broken(**_kwargs):
            raise ValueError("boom")

        monkeypatch.setattr(stream_formatter_mod, "mark_repeated_figures", broken)

        events = await _run_turn(
            TurnContext(question="How many at S1?", earlier_answers=(_earlier_answer(("90", FigureOrigin.step)),)),
            *_read_stock_cell(),
            StreamTextDelta(text="S1 holds 250 units, against 4,600 last year."),
        )

        figures = _by_text(_figures(events))
        assert figures["250"]["origin"] == "step"
        assert figures["250"]["sources"] == [{"tool_use_id": "toolu_read", "page": None}]
        assert figures["4,600"]["origin"] == "none"
        assert [
            r for r in caplog.records if r.name == stream_formatter_mod.logger.name and r.levelname == "ERROR"
        ]


class TestQuestionText:
    async def test_a_value_the_user_typed_answering_the_agent_counts_as_the_question(self):
        from anton.core.interaction.elicit import AskAnswer
        from anton.core.llm.provider import StreamAskUserAnswered

        events = await _run_turn(
            TurnContext(question="What does overtime cost?"),
            StreamAskUserAnswered(id="q1", answer=AskAnswer(status="answered", values=("$50",), text="$42")),
            StreamTextDelta(text="At $42 an hour rather than $50."),
        )

        figures = _by_text(_figures(events))
        assert figures["$42"]["origin"] == "question"
        # A picked option is the agent's own wording.
        assert figures["$50"]["origin"] == "none"

    async def test_context_the_client_appends_for_the_agent_is_not_the_question(self):
        sent = (
            "How many units ship over 2 weeks?\n\n"
            "[Notion pages added for this message — Anton-only context, do not echo back]\n"
            "The user just added the following Notion page(s).\n"
            "--- Notion page: Plan, https://www.notion.so/plan ---\n"
            "Overtime is $30 an hour.\n"
            "--- end of page ---"
        )

        events = await _run_turn(TurnContext(question=sent), StreamTextDelta(text="Over 2 weeks, at $30 an hour."))

        figures = _by_text(_figures(events))
        assert figures["2"]["origin"] == "question"
        assert figures["$30"]["origin"] == "none"


class TestAnswersWithoutSources:
    async def test_a_cell_whose_arguments_the_pod_cut_still_reads_its_code(self):
        """The pod cuts a call's arguments at 65,536 characters; the cell's
        result still carries its code, so a figure the code states is not
        sourced to the cell that only printed it back."""
        code = "# " + "x" * 70_000 + "\nprint(250)"
        cell = {"code": code, "stdout": "250\n", "stderr": "", "error": None, "description": ""}
        args = json.dumps({"action": "exec", "name": "main", "code": code})[:65_536]

        events = await _run_turn(
            TurnContext(question="How many?"),
            StreamToolUseStart(id="toolu_cut", name="scratchpad"),
            StreamToolUseDelta(id="toolu_cut", json_delta=args),
            StreamToolUseEnd(id="toolu_cut"),
            StreamToolResult(name="scratchpad", content=json.dumps(cell), action="exec", id="toolu_cut"),
            StreamTextDelta(text="There are 250."),
        )

        assert _by_text(_figures(events))["250"]["origin"] == "none"

    @pytest.mark.parametrize("before, after", [
        ([], []),
        (
            [
                StreamTextDelta(text="I read stock.csv: S1 holds 250 units."),
                *_read_stock_cell(),
                StreamComplete(response=SimpleNamespace(stop_reason="tool_use", tool_calls=[object()])),
            ],
            [],
        ),
        ([*_read_stock_cell(), StreamTextDelta(text="S1 holds 250 units, but")], []),
        ([], [StreamTextDelta(text="\n")]),
    ], ids=["whole-answer", "after-a-tool-round", "after-partial-text", "followed-by-a-blank-delta"])
    async def test_antons_failure_reply_gets_no_sources(self, before: list, after: list):
        """A pod image or a local editable anton older than mindsdb/anton#569
        streams its failure reply as one delta and ends the turn on it, after
        whatever the turn streamed first. From #569 on, which the anton-agent
        floor carries, the turn raises instead, so this check can go once the
        pod image carries #569 and no local editable anton older than #569 is
        in use."""
        events = await _run_turn(
            TurnContext(question="How many?"),
            *before,
            StreamTextDelta(text="An unexpected error occurred: Server returned 413 Payload Too Large"),
            *after,
        )

        types = [event_type for event_type, _ in events]
        assert "response.figure_sources" not in types
        assert types[-1] == "response.completed"

    async def test_an_error_line_the_model_quotes_keeps_the_answers_sources(self):
        events = await _run_turn(
            TurnContext(question="How many?"),
            *_read_stock_cell(),
            StreamTextDelta(text="Your last run failed with:\n\n"),
            StreamTextDelta(text="An unexpected error occurred: disk full\n\n"),
            StreamTextDelta(text="This time S1 holds 250 units."),
        )

        assert [(f["text"], f["origin"]) for f in _figures(events)["figures"]] == [("250", "step")]

    async def test_a_replaced_round_that_quoted_the_error_leaves_the_new_answer_sourced(self):
        events = await _run_turn(
            TurnContext(question="How many?"),
            *_read_stock_cell(),
            StreamTextDelta(text="An unexpected error occurred: earlier, so 300 units."),
            StreamTaskProgress(phase="continuation", message=""),
            StreamTextDelta(text="S1 holds 250 units."),
        )

        assert [f["text"] for f in _figures(events)["figures"]] == ["250"]


class TestTheInProcessHandlerSavesTheSources:
    """ResponsesHandler._produce runs a turn through the real formatter and
    saves its events with the answer, whatever the UI's
    show_figure_sources setting."""

    @staticmethod
    async def _produce(items: list) -> tuple[list[str], list[dict]]:
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        from cowork.handlers.responses import ResponsesHandler

        async def stream_response(**_kwargs):
            for item in items:
                yield item

        harness = SimpleNamespace(
            id="anton", formatter=staticmethod(format_responses_stream), stream_response=stream_response,
        )
        handler = object.__new__(ResponsesHandler)
        handler.principal = None
        handler.harness = harness
        frames: list[str] = []

        class Buffer:
            async def append(self, _kind, data):
                frames.append(data["sse"])

            async def close(self, _status):
                pass

        with (
            patch("cowork.handlers.responses.ConversationService") as conversations,
            patch("cowork.handlers.responses.get_harness", return_value=harness),
        ):
            conversations.return_value.get_conversation.return_value = MagicMock()
            conversations.return_value.save_user_message.return_value = SimpleNamespace(id=uuid4())
            conversations.return_value.save_assistant_turn.return_value = SimpleNamespace(id=uuid4())
            await handler._produce(
                conv_id=uuid4(), harness_input=[{"type": "text", "text": "How many at S1?"}],
                original_content="How many at S1?", model="anton", disabled=None,
                harness_name="anton", harness_id="anton", buffer=Buffer(),
            )
            [saved_call] = conversations.return_value.save_assistant_turn.call_args_list
        return frames, saved_call.args[2]

    async def test_the_saved_answer_carries_its_figure_sources(self):
        frames, saved = await self._produce([
            TurnContext(question="How many at S1?"), *_read_stock_cell(), StreamTextDelta(text="S1 holds 250 units."),
        ])

        [figure_sources] = [event for event in saved if event.get("type") == "response.figure_sources"]
        assert [(f["text"], f["sources"]) for f in figure_sources["figures"]] == [
            ("250", [{"tool_use_id": "toolu_read", "page": None}]),
        ]
        assert any(frame.startswith("event: response.figure_sources") for frame in frames)

    async def test_failed_native_output_stays_unsourced_in_the_saved_answer(self):
        tool_input = {"url": "https://example.com/rates"}
        frames, saved = await self._produce([
            TurnContext(question="What is the rate?"),
            *_tool_call("native", "web_fetch", tool_input),
            *_tool_done("native", None),
            StreamTextDelta(text="The rate is 404."),
            TurnHistory(_tool_rows("native", "web_fetch", tool_input, "404")),
        ])

        [persisted] = [event for event in saved if event.get("type") == "response.figure_sources"]
        assert persisted == _figures(_parse_sse(frames))
        assert _by_text(persisted)["404"]["origin"] == "none"
        assert persisted["steps"] == []

    async def test_the_saved_answer_carries_a_repeated_figures_mark(self):
        frames, saved = await self._produce([
            TurnContext(question="And S2?", earlier_answers=(_earlier_answer(("4,600", FigureOrigin.step)),)),
            *_read_stock_cell(),
            StreamTextDelta(text="S2 holds 90 units, against 4,600 last year."),
        ])

        [persisted] = [event for event in saved if event.get("type") == "response.figure_sources"]
        assert persisted == _figures(_parse_sse(frames))
        assert {text: figure["origin"] for text, figure in _by_text(persisted).items()} == {
            "90": "step", "4,600": "earlier_answer",
        }
        assert _by_text(persisted)["4,600"]["sources"] == []

    async def test_the_answer_carries_them_with_the_setting_off(self):
        from cowork.common.settings.user_settings import UserSettings, use_turn_settings
        from cowork.db.scoped import LOCAL_SCOPE

        with use_turn_settings(LOCAL_SCOPE, UserSettings(show_figure_sources=False)):
            frames, saved = await self._produce([
                TurnContext(question="How many at S1?"), *_read_stock_cell(), StreamTextDelta(text="S1 holds 250 units."),
            ])

        assert [event.get("type") for event in saved].count("response.figure_sources") == 1
        assert any(frame.startswith("event: response.figure_sources") for frame in frames)


# The figure_sources frames of the turn _every_input_shape builds, compacted
# with no tool rows and then whole.
_EVERY_INPUT_SHAPE_FRAMES = Path(__file__).parent / "fixtures" / "figure_sources_every_input_shape.sse"


async def _every_input_shape(*, compacted: bool) -> str:
    """The response.figure_sources frame of one turn that crosses every shape
    the formatter reads a step from: a streamed cell, a streamed cell whose
    arguments were cut short, a streamed cell with no arguments, a cell that
    calls a client an earlier cell built, web and file calls whose results
    only the rows hold, a file read with no path, a web call with odd
    arguments whose query states a number its results echo, an is_error
    row, a web call whose tool_done said it failed, a cell only the rows
    hold, junk rows, earlier turns' cells (one printed a figure the answer
    repeats) beside junk history rows, and an evidence call no row holds.

    The compacted turn is the one a producer whose anton predates
    ChatSession.turn_messages sends: a compaction and no tool rows, so only
    the figures a streamed cell holds keep a mark. The whole turn sends its
    rows."""
    load = "import pandas as pd\ndf = pd.read_csv('stock.csv')"
    total = "print(df.qty.sum())"
    history = [
        {"role": "user", "content": "Load the stock."},
        *_cell_rows("toolu_load", code=load, stdout=""),
        *_cell_rows("toolu_old", code="print(80 * 101)", stdout="8080\n"),
        {"role": "assistant", "content": "Loaded."},
        {"content": 5},
        {"role": "assistant", "content": [{"type": "tool_use", "name": "scratchpad", "input": {}}]},
    ]
    rows = [
        *_tool_rows("toolu_web", "web_fetch", {"url": "https://example.com/rates"}, "Overtime rate: $42 per hour"),
        *_tool_rows(
            "toolu_file", "read_text_file", {"path": "/workspace/stock.csv"},
            [{"type": "text", "text": "part,site,qty\nP100,S1,250\n"}, {"type": "image"}],
        ),
        *_tool_rows("toolu_nopath", "read_text_file", {"path": 5}, "notes: 33 lines"),
        *_tool_rows(
            "toolu_search", "web_search", {"url": None, "query": "overtime rates 2024"}, "18 results found for 2024",
        ),
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_bad", "name": "web_fetch", "input": {"url": "https://x.example/"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_bad", "content": "rate 91", "is_error": True},
        ]},
        *_tool_rows("toolu_failed", "web_fetch", {"url": "https://down.example/"}, "Rate 64 per hour"),
        *_cell_rows("toolu_rowcell", code="print(5000 + 150)", stdout="5150\n", logs="ok\n", description="Row cell"),
    ]
    read = [
        *rows,
        "junk",
        {"role": "user", "content": [{"type": "tool_result"}]},
        {"content": [{"type": "tool_use", "id": "z", "name": "web_fetch", "input": "nope"}]},
    ]
    sum_cell = asdict(Cell(code=total, stdout="340\n", stderr="", error=None, description="Sum stock"))
    cut_cell = asdict(Cell(code="print(77 * 101)", stdout="7777\n", stderr="", error=None))
    bare_cell = asdict(Cell(code="print(6 * 7 * 100)", stdout="4200\n", stderr="", error=None))
    setup = "client = httpx.Client(base_url='https://erp.example.com')"
    fetch = "print(client.get('/stock?token=synthetic-secret').text)"
    setup_cell = asdict(Cell(code=setup, stdout="", stderr="", error=None, description="Open ERP client"))
    fetch_cell = asdict(Cell(code=fetch, stdout="612\n", stderr="", error=None, description="Fetch stock"))
    frames = [
        chunk async for chunk in format_responses_stream(_events(
            TurnContext(question="What is the total, the rate and S1?", history=history),
            *_tool_call("toolu_sum", "scratchpad", {"action": "exec", "name": "main", "code": total}),
            StreamToolResult(name="scratchpad", content=json.dumps(sum_cell), action="exec", id="toolu_sum"),
            StreamToolUseStart(id="toolu_cut", name="scratchpad"),
            StreamToolUseDelta(id="toolu_cut", json_delta='{"action": "exec", "name": "main", "co'),
            StreamToolUseEnd(id="toolu_cut"),
            StreamToolResult(name="scratchpad", content=json.dumps(cut_cell), action="exec", id="toolu_cut"),
            StreamToolResult(name="scratchpad", content=json.dumps(bare_cell), action="exec", id="toolu_bare"),
            *_tool_call("toolu_setup", "scratchpad", {"action": "exec", "name": "main", "code": setup}),
            StreamToolResult(name="scratchpad", content=json.dumps(setup_cell), action="exec", id="toolu_setup"),
            *_tool_call("toolu_fetch", "scratchpad", {"action": "exec", "name": "main", "code": fetch}),
            StreamToolResult(name="scratchpad", content=json.dumps(fetch_cell), action="exec", id="toolu_fetch"),
            *_tool_call("toolu_web", "web_fetch", {"url": "https://example.com/rates"}),
            *_tool_done("toolu_web", True),
            *_tool_call("toolu_file", "read_text_file", {"path": "/workspace/stock.csv"}),
            *_tool_done("toolu_file", True),
            *_tool_call("toolu_nopath", "read_text_file", {"path": 5}),
            *_tool_done("toolu_nopath", True),
            *_tool_call("toolu_search", "web_search", {"url": None, "query": "overtime rates 2024"}),
            *_tool_done("toolu_search", True),
            *_tool_call("toolu_bad", "web_fetch", {"url": "https://x.example/"}),
            *_tool_done("toolu_bad", False),
            *_tool_call("toolu_failed", "web_fetch", {"url": "https://down.example/"}),
            *_tool_done("toolu_failed", False),
            *_tool_call("toolu_lost", "web_fetch", {"url": "https://lost.example/"}),
            *_tool_done("toolu_lost", True),
            *([StreamContextCompacted(message="Context compacted")] if compacted else []),
            StreamTextDelta(text=(
                "340 units in all, 7777 checked, 4200 billed, 612 in the ERP, overtime $42 an hour, S1 holds 250, "
                "5150 rows, 18 results for 2024, 33 lines, 64 on the down page, and 91 and 8080 elsewhere."
            )),
            *([] if compacted else [TurnHistory(rows, read=read)]),
        ), model="test-model")
    ]
    return "".join(frame for frame in frames if frame.startswith("event: response.figure_sources\n"))


class TestTheFigureSourcesEventIsUnchanged:
    async def test_every_input_shape_gives_the_same_frames(self, monkeypatch):
        """The frames are pinned byte for byte, so a change to how the
        formatter reads its steps can't move a figure, a source or a step."""
        from uuid import UUID

        monkeypatch.setattr(stream_formatter_mod, "time", SimpleNamespace(time=lambda: 1_760_000_000.0))
        monkeypatch.setattr(stream_formatter_mod, "uuid", SimpleNamespace(uuid4=lambda: UUID(int=0xABCDEF)))

        frames = await _every_input_shape(compacted=True) + await _every_input_shape(compacted=False)

        assert frames == _EVERY_INPUT_SHAPE_FRAMES.read_text()
