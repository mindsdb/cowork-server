"""End-to-end demonstration of ENG-664's core claim, run against the real
code on both sides of the boundary (anton's `ChatSession._summarize_history`
+ cowork-server's `AntonHarness._seed_history` / `_persist_history_compaction`
+ a real sqlite `Conversation`/`Message` DB) — no mocked business logic, only
the LLM calls are faked.

Simulates a 30-turn conversation and checks the Done-when criteria from the
ticket:
  * tokens/turn level off instead of growing every turn
  * one summary is created, then reused and extended — not recreated
  * a fact from an early turn is still recoverable after compaction
  * the `history_compaction_enabled=False` switch reverts to unbounded growth
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlmodel import Session

from anton.core.llm.provider import LLMResponse, ProviderConnectionInfo, Usage
from anton.core.session import ChatSession, ChatSessionConfig

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.harnesses.anton_harness.harness import AntonHarness
from cowork.models.message import Message
from cowork.services.conversations import ConversationService
from cowork.services.projects import GENERAL_PROJECT_ID

# Deliberately tiny so a demo-sized conversation crosses the compaction
# threshold multiple times without needing huge synthetic strings.
FAKE_WINDOW_TOKENS = 300
N_TURNS = 30
_BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _stamp(m):
    """Mirrors AntonHarness._build_chat_session's `_stamped` closure exactly."""
    om = m.to_openai_message().model_dump()
    ts = m.created_at.strftime("%Y-%m-%d %H:%M") if getattr(m, "created_at", None) else None
    if ts and isinstance(om.get("content"), str) and om["content"]:
        om["content"] = f"[{ts}] {om['content']}"
    return om


def _mock_llm():
    # Mirrors anton's tests/conftest.py::make_mock_llm — ChatSession.__init__
    # reads coding_provider/planning_provider synchronously, which a bare
    # AsyncMock would otherwise turn into unawaited coroutines.
    llm = AsyncMock()
    llm.coding_provider = MagicMock()
    llm.coding_provider.export_connection_info = MagicMock(
        return_value=ProviderConnectionInfo(provider="anthropic", api_key="test")
    )
    llm.coding_model = "claude-sonnet-4-6"
    llm.planning_provider = MagicMock()
    llm.planning_provider.native_web_tools = MagicMock(return_value=set())

    def _plan(messages, **kwargs):
        chars = sum(len(json.dumps(m)) for m in messages)
        input_tokens = chars // 4
        pressure = min(input_tokens / FAKE_WINDOW_TOKENS, 1.0)
        return LLMResponse(content="ack", usage=Usage(input_tokens=input_tokens, context_pressure=pressure))

    def _summarize(system, messages, max_tokens=None):
        # A believable fake summarizer: carries forward every FACT_n marker
        # it's shown (from old turns and/or a prior summary being updated),
        # like a real STATE RECORD would.
        text = messages[0]["content"]
        facts = sorted(set(re.findall(r"FACT_\d+", text)))
        return LLMResponse(content="## Remaining facts\n" + ", ".join(facts))

    llm.plan = AsyncMock(side_effect=_plan)
    llm.summarize = AsyncMock(side_effect=_summarize)
    return llm


@pytest.fixture
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


async def _run_conversation(session, *, compaction_enabled: bool) -> list[int]:
    svc = ConversationService(session)
    conv = svc.create_conversation("plateau-test", project_id=GENERAL_PROJECT_ID)
    chars_per_turn: list[int] = []
    compactions = 0

    for turn in range(1, N_TURNS + 1):
        ordered_messages = svc.get_ordered_messages(conv.id)
        history_summary = conv.history_summary if compaction_enabled else None
        initial_history, seed_info = AntonHarness._seed_history(
            ordered_messages, history_summary, conv.history_summary_cutoff_id, _stamp,
        )
        chars_per_turn.append(len(json.dumps(initial_history)))

        anton_session = ChatSession(ChatSessionConfig(llm_client=_mock_llm(), initial_history=initial_history))
        reply = await anton_session.turn(f"Turn {turn}: remember FACT_{turn} = value{turn}.")

        if compaction_enabled and anton_session.last_compaction is not None:
            compactions += 1
            AntonHarness._persist_history_compaction(conv, anton_session, seed_info)

        # Explicit, strictly-increasing timestamps: this loop runs many turns
        # within the same wall-clock second, but _MESSAGE_ORDER's tie-break
        # only disambiguates a same-turn user/assistant pair, not different
        # turns — without distinct created_at values every user message
        # would sort before every assistant message, conversation-wide.
        session.add(Message(
            conversation_id=conv.id, role="user",
            content=f"Turn {turn}: remember FACT_{turn} = value{turn}.",
            created_at=_BASE_TIME + timedelta(seconds=2 * turn),
        ))
        session.add(Message(
            conversation_id=conv.id, role="assistant", content=reply,
            created_at=_BASE_TIME + timedelta(seconds=2 * turn + 1),
        ))
        session.commit()
        conv = svc.get_conversation(conv.id)

    return chars_per_turn, compactions, conv


class TestHistoryCompactionPlateau:
    async def test_chars_per_turn_level_off_when_enabled(self, session, capsys):
        chars_per_turn, compactions, conv = await _run_conversation(session, compaction_enabled=True)

        with capsys.disabled():
            print(f"\n[enabled]  compactions={compactions}  chars/turn={chars_per_turn}")

        assert compactions >= 2, "expected multiple compactions across 30 turns"
        # The tail keeps growing between compactions but never anywhere near
        # unbounded — the max stays close to the window budget instead of
        # climbing turn after turn like the old full-history replay did.
        first_half_max = max(chars_per_turn[:10])
        second_half_max = max(chars_per_turn[20:])
        assert second_half_max < first_half_max * 2, (
            "chars/turn should plateau, not keep growing turn over turn"
        )

        # One summary, reused+extended — not recreated from scratch each time
        # (the fake summarizer only ever sees FACT markers it's asked to fold
        # in; if it were resummarizing from scratch every pass instead of
        # updating in place, early FACTs would eventually fall out of the
        # window it's shown).
        assert "FACT_1" in conv.history_summary, "an early-turn fact must still be recoverable"

    async def test_disabled_reverts_to_unbounded_full_history(self, session, capsys):
        chars_per_turn, compactions, conv = await _run_conversation(session, compaction_enabled=False)

        with capsys.disabled():
            print(f"\n[disabled] compactions={compactions}  chars/turn={chars_per_turn}")

        assert compactions == 0
        assert conv.history_summary is None
        # Full history every turn -> strictly grows (old, pre-ENG-664 behavior).
        assert chars_per_turn == sorted(chars_per_turn)
        assert chars_per_turn[-1] > chars_per_turn[0]
