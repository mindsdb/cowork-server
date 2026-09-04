"""Tests for the archive-search tool wired into the anton harness (ENG-735).

Covers the three pieces the pure search (tests/test_history_recall.py) does
not: which messages count as the archive, when the tool is offered to the
agent at all, and what the handler answers on each path.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_engine
from cowork.harnesses.anton_harness.harness import AntonHarness
from cowork.harnesses.anton_harness.tools import build_cowork_recall_history_tool
from cowork.services.conversations import ConversationService
from cowork.services.projects import GENERAL_PROJECT_ID


@pytest.fixture
def svc():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield ConversationService(ScopedSession(s, LOCAL_SCOPE))


def _conversation_with_turns(svc, turns: list[tuple[str, str]]):
    """A saved conversation whose messages are the given (question, answer) pairs."""
    conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)
    for question, answer in turns:
        svc.save_user_message(conv.id, question)
        svc.save_assistant_turn(conv.id, answer, [], harness="anton")
    return conv


def _tool(svc, conv):
    """The tool as the harness builds it — reading the archive per call."""
    return build_cowork_recall_history_tool(lambda: svc.archived_messages(conv.id))


def _call(tool, **tc_input) -> str:
    # The handler ignores anton's session (it holds the archive reader instead).
    return asyncio.run(tool.handler(None, tc_input))


class TestArchivedMessages:
    def test_empty_without_a_summary(self, svc):
        conv = _conversation_with_turns(svc, [("q1", "a1"), ("q2", "a2")])

        assert svc.archived_messages(conv.id) == []

    def test_stops_at_the_cutoff(self, svc):
        conv = _conversation_with_turns(svc, [("q1", "a1"), ("q2", "a2"), ("q3", "a3")])
        ordered = svc.get_ordered_messages(conv.id)
        svc.update_history_compaction(conv.id, "SUMMARY", ordered[1].id)

        archived = svc.archived_messages(conv.id)

        assert [m["content"] for m in archived] == ["q1", "a1"]

    def test_empty_when_the_cutoff_row_is_gone(self, svc):
        # Same staleness rule the replay applies: an unknown cutoff means the
        # saved summary no longer describes this history.
        conv = _conversation_with_turns(svc, [("q1", "a1")])
        other = _conversation_with_turns(svc, [("elsewhere", "a")])
        foreign_id = svc.get_ordered_messages(other.id)[0].id
        svc.update_history_compaction(conv.id, "SUMMARY", foreign_id)

        assert svc.archived_messages(conv.id) == []


class TestToolRegistration:
    def test_offered_once_a_summary_exists(self, svc):
        conv = _conversation_with_turns(svc, [("q1", "a1")])
        svc.update_history_compaction(
            conv.id, "SUMMARY", svc.get_ordered_messages(conv.id)[0].id,
        )
        user = SimpleNamespace(history_compaction_enabled=True)

        tool = AntonHarness._recall_history_tool(conv, user)

        assert tool is not None and tool.name == "recall_history"

    def test_withheld_before_the_first_compaction(self, svc):
        conv = _conversation_with_turns(svc, [("q1", "a1")])
        user = SimpleNamespace(history_compaction_enabled=True)

        assert AntonHarness._recall_history_tool(conv, user) is None

    def test_withheld_when_compaction_is_off(self, svc):
        conv = _conversation_with_turns(svc, [("q1", "a1")])
        svc.update_history_compaction(
            conv.id, "SUMMARY", svc.get_ordered_messages(conv.id)[0].id,
        )
        user = SimpleNamespace(history_compaction_enabled=False)

        assert AntonHarness._recall_history_tool(conv, user) is None


class TestHandler:
    def test_returns_the_matching_turn(self, svc):
        conv = _conversation_with_turns(svc, [
            ("what is the staging password", "it is hunter2"),
            ("rename the file", "renamed"),
            ("what is next", "nothing"),
        ])
        ordered = svc.get_ordered_messages(conv.id)
        svc.update_history_compaction(conv.id, "SUMMARY", ordered[3].id)

        answer = _call(_tool(svc, conv), query="staging password")

        assert "hunter2" in answer

    def test_does_not_search_past_the_cutoff(self, svc):
        # The tail is already in the agent's context; returning it again would
        # spend context on turns it can already read.
        conv = _conversation_with_turns(svc, [("old news", "ok"), ("hunter2 lives here", "ok")])
        ordered = svc.get_ordered_messages(conv.id)
        svc.update_history_compaction(conv.id, "SUMMARY", ordered[1].id)

        answer = _call(_tool(svc, conv), query="hunter2")

        # The no-match reply echoes the query, so the turn's own words are what
        # proves it was not returned.
        assert "lives here" not in answer
        assert "no earlier turn matches" in answer

    def test_no_match_asks_for_a_rephrase_once(self, svc):
        conv = _conversation_with_turns(svc, [("q1", "a1"), ("q2", "a2")])
        ordered = svc.get_ordered_messages(conv.id)
        svc.update_history_compaction(conv.id, "SUMMARY", ordered[1].id)

        answer = _call(_tool(svc, conv), query="quarterly forecast")

        # The reply has to teach the retry, not just refuse: the match rule is
        # prefix-based, which the model cannot infer from an empty result.
        assert "word beginnings" in answer
        assert "Do not repeat this query" in answer

    def test_reports_an_empty_archive(self, svc):
        conv = _conversation_with_turns(svc, [("q1", "a1")])

        answer = _call(_tool(svc, conv), query="anything")

        assert "nothing is archived" in answer

    def test_missing_query_is_rejected(self, svc):
        conv = _conversation_with_turns(svc, [("q1", "a1")])

        assert "required" in _call(_tool(svc, conv), query="  ")

    def test_limit_is_clamped(self, svc):
        conv = _conversation_with_turns(svc, [(f"backup {i}", "ok") for i in range(14)])
        ordered = svc.get_ordered_messages(conv.id)
        svc.update_history_compaction(conv.id, "SUMMARY", ordered[-1].id)

        answer = _call(_tool(svc, conv), query="backup", limit=99)

        assert answer.count("--- turn ") == 10

    def test_junk_limit_falls_back_to_the_default(self, svc):
        conv = _conversation_with_turns(svc, [(f"backup {i}", "ok") for i in range(9)])
        ordered = svc.get_ordered_messages(conv.id)
        svc.update_history_compaction(conv.id, "SUMMARY", ordered[-1].id)

        answer = _call(_tool(svc, conv), query="backup", limit="lots")

        assert answer.count("--- turn ") == 3
