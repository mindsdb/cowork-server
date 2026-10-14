from __future__ import annotations

import json

from anton.core.llm.provider import (
    StreamTextDelta,
    StreamToolResult,
    StreamToolUseDelta,
    StreamToolUseEnd,
    StreamToolUseStart,
)
from cowork.harnesses.anton_harness.stream_formatter import TurnQuestion, format_responses_stream


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


async def _run(*items, question: str = "", sink: list | None = None) -> list[tuple[str, dict]]:
    def event_sink(event_type: str, data: dict) -> None:
        if sink is not None:
            sink.append((event_type, data))

    return _parse_sse([
        chunk async for chunk in format_responses_stream(
            _events(TurnQuestion(text=question), *items), model="test-model", event_sink=event_sink,
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
        by_text = {f["text"]: f for f in payload["figures"]}
        assert by_text["250"]["origin"] == "step"
        assert by_text["250"]["sources"] == [
            {"tool_use_id": "toolu_read", "kind": "file", "label": "stock.csv", "page": None},
        ]
        answer = "Site S1 holds 250 units; overtime costs $30."
        assert by_text["$30"] == {
            "start": answer.index("$30"), "end": answer.index("$30") + 3, "text": "$30", "origin": "none", "sources": [],
        }
        # Recorded like every other event, so it saves with the answer and
        # comes back on reload.
        assert ("response.figure_sources", payload) in sink

    async def test_the_question_event_streams_nothing_and_marks_figures_it_states(self):
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
            StreamTextDelta(text="Site S1 holds 250 units; "),
            StreamTextDelta(text="overtime costs $30."),
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
        assert [content[f["start"]:f["end"]] for f in saved[0]["figures"]] == ["250", "$30"]
