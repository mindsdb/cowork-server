"""ENG-735's acceptance criterion, end to end against the real code.

A fact the summary dropped must be recoverable through the agent's own tool
call, without replaying the whole history. Nothing is mocked except the LLM:
real anton `ChatSession` (tool dispatch included), real cowork compaction
replay/persistence, real sqlite `Conversation`/`Message` rows.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlmodel import Session

from anton.core.llm.provider import LLMResponse, ProviderConnectionInfo, ToolCall, Usage
from anton.core.session import ChatSession, ChatSessionConfig

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_engine
from cowork.harnesses.anton_harness.harness import AntonHarness
from cowork.harnesses.anton_harness.tools import build_cowork_recall_history_tool
from cowork.models.message import Message
from cowork.services.conversations import ConversationService
from cowork.services.projects import GENERAL_PROJECT_ID

pytestmark = pytest.mark.skipif(
    not hasattr(ChatSession, "last_compaction"),
    reason="requires anton's ChatSession.last_compaction (ENG-664) — pin not bumped yet",
)

FAKE_WINDOW_TOKENS = 300
_BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)
# The detail the summary throws away, mentioned once in an early turn.
SECRET = "hunter2-staging"


def _base_llm():
    llm = AsyncMock()
    llm.coding_provider = MagicMock()
    llm.coding_provider.export_connection_info = MagicMock(
        return_value=ProviderConnectionInfo(provider="anthropic", api_key="test")
    )
    llm.coding_model = "claude-sonnet-4-6"
    llm.planning_provider = MagicMock()
    llm.planning_provider.native_web_tools = MagicMock(return_value=set())
    return llm


def _forgetful_llm():
    """Answers "ack", and summarizes by dropping everything but the goal.

    This is the risk ENG-735 exists for: a real STATE RECORD flattens detail,
    so the password mentioned in turn 1 is simply not in the summary.
    """
    llm = _base_llm()

    def _plan(messages, **kwargs):
        chars = sum(len(json.dumps(m)) for m in messages)
        input_tokens = chars // 4
        return LLMResponse(
            content="ack",
            usage=Usage(
                input_tokens=input_tokens,
                context_pressure=min(input_tokens / FAKE_WINDOW_TOKENS, 1.0),
            ),
        )

    llm.plan = AsyncMock(side_effect=_plan)
    llm.summarize = AsyncMock(
        side_effect=lambda *a, **kw: LLMResponse(content="## Goal\nSet up the staging deploy.")
    )
    return llm


def _recalling_llm(query: str):
    """Calls `recall_history` once, then answers with whatever it got back.

    Mirrors what a real model does with a tool result, so the assertion at the
    end is on the turn's actual reply — proof the tool is reachable through
    anton's dispatcher, not just callable in isolation.
    """
    llm = _base_llm()
    calls: list[int] = []

    def _plan(messages, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(id="call-1", name="recall_history", input={"query": query})],
            )
        # The tool result is the last thing in the history anton hands back.
        return LLMResponse(content=f"From the archive: {json.dumps(messages[-1]['content'])}")

    llm.plan = AsyncMock(side_effect=_plan)
    return llm


@pytest.fixture
def svc():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield ConversationService(ScopedSession(s, LOCAL_SCOPE))


def _save_turn(svc, conv_id, turn: int, question: str, answer: str) -> None:
    # Explicit strictly-increasing timestamps: many turns land in the same
    # wall-clock second, and canonical order only disambiguates a same-turn
    # user/assistant pair.
    svc.session.add(Message(
        conversation_id=conv_id, role="user", content=question,
        created_at=_BASE_TIME + timedelta(seconds=2 * turn),
    ))
    svc.session.add(Message(
        conversation_id=conv_id, role="assistant", content=answer,
        created_at=_BASE_TIME + timedelta(seconds=2 * turn + 1),
    ))
    svc.session.commit()


def _seed(svc, conv):
    replayable = [
        m for m in svc.get_ordered_messages(conv.id) if m.role in {"user", "assistant"}
    ]
    return AntonHarness._seed_history(
        replayable, conv.history_summary, conv.history_summary_cutoff_id,
        AntonHarness._stamp_message,
    )


class TestDroppedFactIsRecoverable:
    async def test_agent_recovers_it_with_a_tool_call(self, svc):
        conv = svc.create_conversation("recall-test", project_id=GENERAL_PROJECT_ID)

        # 1. A conversation long enough to compact. The secret is said once, in
        #    the first turn, and never repeated.
        turns = [(f"The staging password is {SECRET}.", "noted")]
        turns += [(f"Turn {i}: unrelated chatter about the deploy.", "ack") for i in range(2, 9)]

        compacted = False
        for turn, (question, answer) in enumerate(turns, start=1):
            initial_history, seed_info = _seed(svc, conv)
            session = ChatSession(ChatSessionConfig(
                llm_client=_forgetful_llm(), initial_history=initial_history,
            ))
            await session.turn(question)
            if session.last_compaction is not None:
                AntonHarness._persist_history_compaction(conv, session, seed_info)
                compacted = True
            _save_turn(svc, conv.id, turn, question, answer)
            conv = svc.get_conversation(conv.id)

        assert compacted, "the conversation never compacted — nothing to recall from"

        # 2. The summary dropped it, and the replayed history no longer carries
        #    it: without the tool the fact is unreachable this turn.
        initial_history, _ = _seed(svc, conv)
        assert SECRET not in (conv.history_summary or "")
        assert SECRET not in json.dumps(initial_history)

        # 3. The agent asks for it and gets it back.
        tool = AntonHarness._recall_history_tool(
            conv, MagicMock(history_compaction_enabled=True),
        )
        assert tool is not None, "the tool should be offered once a summary exists"

        session = ChatSession(ChatSessionConfig(
            llm_client=_recalling_llm("staging password"),
            initial_history=initial_history,
            tools=[tool],
        ))
        reply = await session.turn("What was the staging password again?")

        assert SECRET in reply

    async def test_whole_history_is_not_replayed_to_answer(self, svc):
        """The recall reply must stay far smaller than the full history — the
        point of the backstop is a targeted lookup, not undoing compaction."""
        conv = svc.create_conversation("recall-size", project_id=GENERAL_PROJECT_ID)
        for turn in range(1, 13):
            question = f"Turn {turn}: " + ("filler " * 40)
            _save_turn(svc, conv.id, turn, question, "ack " * 40)
        ordered = svc.get_ordered_messages(conv.id)
        svc.update_history_compaction(conv.id, "## Goal\nsomething", ordered[-1].id)
        conv = svc.get_conversation(conv.id)

        archive_size = len(json.dumps(svc.archived_messages(conv.id)))
        tool = AntonHarness._recall_history_tool(
            conv, MagicMock(history_compaction_enabled=True),
        )
        answer = await tool.handler(None, {"query": "filler"})

        assert len(answer) < archive_size / 3
