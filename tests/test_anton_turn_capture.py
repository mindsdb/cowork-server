"""The in-process harness's tool rows for a turn, through a mid-turn compaction.

anton compacts or hard-truncates history inside a turn under context
pressure, replacing `session.history` with a list that no longer holds the
turn's earlier tool rows. `ChatSession.turn_messages()` still returns every
message the turn appended, so the harness reads the turn's rows from it. A
turn that compacted ran near or over the context window, and rows this
process captures have no size cap, so its rows reach the formatter
(TurnHistory.read) and none are stored: the next turn replays it as text only.

An anton without `turn_messages` (a local editable anton older than
mindsdb/anton#585) gets the slice of `history` past the seed, and nothing
after a compaction, where that slice could drop rows or keep a tool_result
whose tool_use was folded away.

The real-session turns run a tool round AFTER the compaction: a turn that
compacts on its final round appends only text afterwards, which
split_turn_into_rows drops, so such a turn cannot tell a read of every list
the turn wrote to from a read of the final one.
"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from anton.core.llm.provider import (
    ContextOverflowError,
    LLMResponse,
    ProviderConnectionInfo,
    StreamComplete,
    StreamTextDelta,
    StreamToolUseDelta,
    StreamToolUseEnd,
    StreamToolUseStart,
    ToolCall,
    Usage,
)
from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.tools import web_tools
from anton.core.tools.registry import ToolOutcome

import cowork.services.artifact_autopublish as autopublish
import cowork.services.task_objects as task_objects
from cowork.harnesses.anton_harness.harness import AntonHarness, BuiltChatSession
from cowork.harnesses.anton_harness import harness as harness_mod
from cowork.harnesses.anton_harness.agent_files import AgentFiles, Attachment
from cowork.harnesses.anton_harness.stream_formatter import TurnContext, TurnHistory, format_responses_stream
from cowork.db.scoped import LOCAL_SCOPE
from cowork.schemas.figure_sources import EarlierAnswer, Figure, FigureOrigin


class StreamContextCompacted:
    """Name matches anton's event class — the harness detects it by class name."""


class _StreamTextDelta:
    pass


class _FakeSession:
    """An anton ChatSession without `turn_messages`: turn_stream appends this
    turn's block-rows to `history`; when `compact` is set it also reassigns
    `history` mid-turn (as _summarize_history does) and yields a compaction
    event."""

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


# The page each web_fetch call reads, by tool_use_id, in call order.
PAGES = {
    "toolu_1": "Wage table: overtime rate is $42 per hour.",
    "toolu_2": "Shift calendar: three shifts.",
    "toolu_3": "Holiday list: none.",
    "toolu_4": "Line roster: the plant runs 9 lines.",
}
TOOL_IDS = list(PAGES)
QUESTION = "What does overtime cost?"
# $42 is on the first page, which the compaction folds into its summary; 9 is
# on the page the round after the compaction reads; no page holds 17.
ANSWER = "Overtime costs $42 an hour, the plant runs 9 lines, and 17 shifts."
SEED = [
    {"role": "user", "content": "seed question one"},
    {"role": "assistant", "content": "seed answer one"},
]
# The planning call that returns toolu_4. The compaction or the overflow lands
# on it, so toolu_4's round runs after the swap.
SWAP_ROUND = 3
SWAP_MODES = ["proactive", "overflow", "hard-truncate"]
# StreamContextCompacted events each mode yields.
COMPACTIONS = {"none": 0, "proactive": 1, "overflow": 1, "hard-truncate": 2}
EXPECTED_ROW_IDS = [(role, tool_use_id) for tool_use_id in TOOL_IDS for role in ("assistant", "user")]


def _scripted_llm(*, mode: str):
    """A planner that calls web_fetch four times, then answers.

    `mode` picks what happens on SWAP_ROUND: "proactive" reports a context
    pressure over the threshold, so the tool loop compacts; "overflow" raises
    ContextOverflowError once (summarize and retry); "hard-truncate" raises it
    twice (summarize, then hard-truncate and retry); "none" does neither.
    """
    llm = AsyncMock()
    llm.coding_provider = MagicMock()
    llm.coding_provider.export_connection_info = MagicMock(
        return_value=ProviderConnectionInfo(provider="anthropic", api_key="test"),
    )
    llm.coding_model = "claude-sonnet-4-6"
    # The real client defaults the router role to the coding model; mirror it,
    # and give the planning role the same id, so the budgets anton derives
    # from a model's window (hard truncation's result stubs among them) read
    # a real model id, not an AsyncMock attribute.
    llm.router_model = "claude-sonnet-4-6"
    llm.planning_model = "claude-sonnet-4-6"
    llm.max_tokens = 8192
    llm.planning_provider = MagicMock()
    llm.planning_provider.native_web_tools = MagicMock(return_value=set())
    state = {"round": 0, "overflowed": 0}
    overflow_budget = {"overflow": 1, "hard-truncate": 2}.get(mode, 0)

    def plan_stream(messages, **kwargs):
        async def gen():
            r = state["round"]
            if r == SWAP_ROUND and state["overflowed"] < overflow_budget:
                state["overflowed"] += 1
                raise ContextOverflowError("too long")
            state["round"] += 1
            pressure = 0.99 if mode == "proactive" and r == SWAP_ROUND else 0.0
            if r < len(TOOL_IDS):
                tool_use_id = TOOL_IDS[r]
                arguments = {"url": f"https://example.com/{tool_use_id}"}
                yield StreamToolUseStart(id=tool_use_id, name="web_fetch")
                yield StreamToolUseDelta(id=tool_use_id, json_delta=json.dumps(arguments))
                yield StreamToolUseEnd(id=tool_use_id)
                yield StreamComplete(response=LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id=tool_use_id, name="web_fetch", input=arguments)],
                    usage=Usage(input_tokens=100, context_pressure=pressure),
                    stop_reason="tool_use",
                ))
            else:
                yield StreamTextDelta(text=ANSWER)
                yield StreamComplete(response=LLMResponse(
                    content=ANSWER,
                    usage=Usage(input_tokens=100, context_pressure=pressure),
                    stop_reason="end_turn",
                ))

        return gen()

    llm.plan_stream = plan_stream
    llm.summarize = AsyncMock(return_value=LLMResponse(content="SUMMARY of earlier turns"))
    return llm


async def _fetch(session, tc_input):
    """anton's web_fetch, without the network: a page read whole, with the
    verdict anton's own handler gives a 2xx response."""
    return ToolOutcome(content=PAGES[tc_input["url"].rsplit("/", 1)[1]], ok=True)


def _real_session(monkeypatch, *, mode: str) -> ChatSession:
    monkeypatch.setattr(web_tools.WEB_FETCH_FALLBACK_TOOL, "handler", _fetch)
    session = ChatSession(ChatSessionConfig(llm_client=_scripted_llm(mode=mode), initial_history=list(SEED)))
    session._verify_min_tool_rounds = 99  # no verifier LLM call
    return session


def _live_result_ids(session: ChatSession) -> list[str]:
    return [
        block["tool_use_id"]
        for message in session.history
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]


def _row_ids(rows: list) -> list[tuple[str, str]]:
    return [(row["role"], block.get("id") or block.get("tool_use_id")) for row in rows for block in row["content"]]


async def _no_earlier_answers(*, conversation_id, scope):
    return ()


def _stream(monkeypatch, session, *, question: str = "hi", read_earlier_answers=_no_earlier_answers):
    """The turn's events, as AntonHarness.stream_response yields them for
    `session`. The earlier answers' read is replaced too, by
    ``read_earlier_answers``, so the turn opens no database unit for it."""
    monkeypatch.setattr(harness_mod, "read_earlier_answers", read_earlier_answers)
    monkeypatch.setattr(task_objects, "snapshot_artifact_state", lambda *_a, **_k: (set(), {}))
    monkeypatch.setattr(task_objects, "index_turn_artifacts", lambda *_a, **_k: ([], set(), None))
    monkeypatch.setattr(task_objects, "cards_for_slugs", lambda *_a, **_k: [])

    async def _no_autopublish(*_a, **_k):
        return set()

    monkeypatch.setattr(autopublish, "autopublish_project_artifacts", _no_autopublish)

    async def _fake_build(
        self, conversation, model=None, reasoning_effort=None, disabled_connections=None, channel_context=None,
        tool_messages=False,
    ):
        return BuiltChatSession(session=session, turn_scratch_dir=None, seed_info=None, learns=True)

    monkeypatch.setattr(AntonHarness, "_build_chat_session", _fake_build)

    conversation = SimpleNamespace(
        id="c1", project_id="p1", project=SimpleNamespace(path="/tmp", name="tmp"),
    )
    return AntonHarness().stream_response(conversation=conversation, input=[{"type": "text", "text": question}])


def _drive(monkeypatch, session, *, question: str = "hi", read_earlier_answers=_no_earlier_answers) -> list:
    async def _drain():
        return [
            ev async for ev in _stream(
                monkeypatch, session, question=question, read_earlier_answers=read_earlier_answers,
            )
        ]

    return asyncio.run(_drain())


def test_an_anton_without_turn_messages_captures_the_rows_past_the_seed(monkeypatch):
    events = _drive(monkeypatch, _FakeSession(compact=False))
    histories = [e for e in events if isinstance(e, TurnHistory)]
    assert len(histories) == 1
    assert [r["role"] for r in histories[0].rows] == ["assistant", "user"]
    assert histories[0].read is None


def test_an_anton_without_turn_messages_captures_nothing_after_a_compaction(monkeypatch):
    events = _drive(monkeypatch, _FakeSession(compact=True))
    assert not any(isinstance(e, TurnHistory) for e in events)


@pytest.mark.parametrize("mode", SWAP_MODES)
def test_a_turn_that_compacts_hands_every_tool_row_to_the_formatter_and_stores_none(monkeypatch, mode):
    session = _real_session(monkeypatch, mode=mode)

    events = _drive(monkeypatch, session, question=QUESTION)

    assert sum(type(e).__name__ == "StreamContextCompacted" for e in events) == COMPACTIONS[mode]
    # Precondition: the swap folded the turn's first round out of the live list.
    assert "toolu_1" not in _live_result_ids(session)
    [history] = [e for e in events if isinstance(e, TurnHistory)]
    assert history.rows == []
    assert _row_ids(history.read) == EXPECTED_ROW_IDS
    assert {
        block["tool_use_id"]: block["content"]
        for row in history.read
        for block in row["content"]
        if block["type"] == "tool_result"
    } == PAGES


def test_a_turn_that_does_not_compact_stores_its_tool_rows(monkeypatch):
    session = _real_session(monkeypatch, mode="none")

    events = _drive(monkeypatch, session, question=QUESTION)

    assert not any(type(e).__name__ == "StreamContextCompacted" for e in events)
    [history] = [e for e in events if isinstance(e, TurnHistory)]
    assert _row_ids(history.rows) == EXPECTED_ROW_IDS
    assert history.read is None


@pytest.mark.parametrize("mode", SWAP_MODES)
def test_a_compacted_turns_figures_come_from_the_rounds_on_both_sides_of_the_compaction(monkeypatch, mode):
    """$42 is on the page the compaction folded away, 9 on the page the round
    after it read. Each is that page's figure, and 17, which no page holds,
    has no source. The turn's rows are read, not stored."""
    session = _real_session(monkeypatch, mode=mode)
    sink: list[tuple[str, dict]] = []

    async def _format():
        return [
            chunk async for chunk in format_responses_stream(
                _stream(monkeypatch, session, question=QUESTION), model="test-model",
                event_sink=lambda event_type, data: sink.append((event_type, data)),
            )
        ]

    asyncio.run(_format())

    [sources] = [data for event_type, data in sink if event_type == "response.figure_sources"]
    figures = {figure["text"]: figure for figure in sources["figures"]}
    assert figures["$42"]["origin"] == "step"
    assert figures["$42"]["sources"] == [{"tool_use_id": "toolu_1", "page": None}]
    assert figures["9"]["origin"] == "step"
    assert figures["9"]["sources"] == [{"tool_use_id": "toolu_4", "page": None}]
    assert figures["17"]["origin"] == "none"
    assert [data["rows"] for event_type, data in sink if event_type == "response.turn_history"] == [[]]


def test_the_turn_opens_with_its_question_its_project_and_the_files_attached_to_it(monkeypatch):
    attachment = Attachment(file_id="f1", stored_path="/files/f1/stock.csv")

    async def attached(*, conversation_id, scope):
        return (attachment,) if conversation_id == "c1" else ()

    monkeypatch.setattr(harness_mod, "read_conversation_attachments", attached)

    events = _drive(monkeypatch, _FakeSession(compact=False))

    assert events[0] == TurnContext(
        question="hi", files=AgentFiles.on_this_server("/tmp", project_id="p1", attachments=[attachment]),
        history=[{"role": "user", "content": "seed"}],
    )
    assert not any(isinstance(e, TurnContext) for e in events[1:])


def test_the_turn_opens_with_the_earlier_answers_it_read(monkeypatch):
    earlier = EarlierAnswer(figures=[Figure(start=0, end=5, text="4,600", origin=FigureOrigin.step)])
    reads = []

    async def read_earlier_answers(*, conversation_id, scope):
        reads.append((conversation_id, scope))
        return (earlier,)

    events = _drive(monkeypatch, _FakeSession(compact=False), read_earlier_answers=read_earlier_answers)

    assert events[0].earlier_answers == (earlier,)
    assert reads == [("c1", LOCAL_SCOPE)]


def test_the_turns_context_holds_the_history_as_seeded_not_as_the_turn_grows_it(monkeypatch):
    """The formatter reads earlier turns' tool results from it once the turn
    ends, by which time the session's own history holds this turn too."""
    events = _drive(monkeypatch, _FakeSession(compact=False))

    assert events[0].history == [{"role": "user", "content": "seed"}]
