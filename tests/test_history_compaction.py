"""Tests for replaying anton's persisted history summary instead of full
history (ENG-664): `AntonHarness._seed_history` (build initial_history from
summary + tail, or fall back to full history),
`AntonHarness._persist_history_compaction` (save the result after a turn),
and `AntonHarness._stamp_message` (per-message timestamp prefix).
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.common.settings.user_settings import UserSettings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_engine
from cowork.harnesses.anton_harness.harness import AntonHarness
from cowork.services.conversations import ConversationService
from cowork.services.projects import GENERAL_PROJECT_ID


def _stamp(m):
    return {"role": m.role, "content": m.id}


def _fake_messages(n: int, role: str = "user") -> list[SimpleNamespace]:
    return [SimpleNamespace(id=uuid4(), role=role) for _ in range(n)]


@pytest.fixture
def svc():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield ConversationService(ScopedSession(s, LOCAL_SCOPE))


class TestHistoryCompactionSetting:
    def test_enabled_by_default(self):
        assert UserSettings().history_compaction_enabled is True

    def test_can_be_disabled(self):
        assert UserSettings(history_compaction_enabled=False).history_compaction_enabled is False


class TestSeedHistory:
    def test_no_summary_uses_full_history(self):
        messages = _fake_messages(4)
        initial_history, seed_info = AntonHarness._seed_history(messages, None, None, _stamp)

        assert initial_history == [_stamp(m) for m in messages]
        assert seed_info["tail_start"] == 0
        assert seed_info["synthetic_prefix_len"] == 0
        assert seed_info["ordered_messages"] == messages

    def test_valid_cutoff_replays_summary_plus_tail(self):
        # alternating roles so the tail (messages[3:]) starts on "assistant" —
        # no separator needed.
        messages = [
            SimpleNamespace(id=uuid4(), role="user" if i % 2 == 0 else "assistant")
            for i in range(5)
        ]
        cutoff_id = messages[2].id  # summary covers messages[0:3]

        initial_history, seed_info = AntonHarness._seed_history(
            messages, "SUMMARY TEXT", cutoff_id, _stamp,
        )

        assert initial_history[0] == {"role": "user", "content": "SUMMARY TEXT"}
        assert initial_history[1:] == [_stamp(m) for m in messages[3:]]
        assert seed_info["tail_start"] == 3
        assert seed_info["synthetic_prefix_len"] == 1

    def test_tail_starting_with_user_gets_assistant_separator(self):
        """Two consecutive `user`-role messages break/degrade most providers —
        if the tail starts with `user`, insert the same assistant separator
        anton's own _summarize_history uses."""
        messages = _fake_messages(3, role="user")  # tail will all be "user"
        cutoff_id = messages[0].id

        initial_history, seed_info = AntonHarness._seed_history(
            messages, "SUMMARY TEXT", cutoff_id, _stamp,
        )

        assert [m["role"] for m in initial_history[:2]] == ["user", "assistant"]
        assert initial_history[2:] == [_stamp(m) for m in messages[1:]]
        assert seed_info["synthetic_prefix_len"] == 2

    def test_stale_cutoff_falls_back_to_full_history(self):
        """The cutoff message isn't in the current message list (e.g. it was
        deleted) — treat the summary as stale and replay everything."""
        messages = _fake_messages(4)
        missing_cutoff_id = uuid4()

        initial_history, seed_info = AntonHarness._seed_history(
            messages, "SUMMARY TEXT", missing_cutoff_id, _stamp,
        )

        assert initial_history == [_stamp(m) for m in messages]
        assert seed_info["tail_start"] == 0
        assert seed_info["synthetic_prefix_len"] == 0


class TestStampMessage:
    """A leaked timestamp bug: an earlier version of `_stamp_message`
    (formerly an inline `_stamped` closure in `_build_chat_session`)
    stamped BOTH roles, so Anton's own replayed assistant replies came back
    to it prefixed with `[YYYY-MM-DD HH:MM] `, and it started imitating that
    visible convention in its own new output — most noticeable on short
    turns ("hi", "who are you?") with little else to anchor generation.
    User-only stamping matches anton's own live-turn stamping
    (core_agent/anton/core/session.py's _stamp_user_content)."""

    @staticmethod
    def _msg(role, content, created_at):
        return SimpleNamespace(
            role=role,
            created_at=created_at,
            to_openai_message=lambda: SimpleNamespace(
                model_dump=lambda: {"role": role, "content": content}
            ),
        )

    def test_stamps_a_user_message(self):
        ts = datetime(2026, 8, 18, 17, 41)
        m = self._msg("user", "who are you?", ts)

        assert AntonHarness._stamp_message(m) == {
            "role": "user", "content": "[2026-08-18 17:41] who are you?",
        }

    def test_does_not_stamp_an_assistant_message(self):
        """The regression: an assistant reply replayed back to Anton must
        stay byte-identical to what was actually said, or the model starts
        echoing the bracketed-timestamp convention in new replies."""
        ts = datetime(2026, 8, 18, 17, 41)
        m = self._msg("assistant", "I'm Anton, your AI coworker.", ts)

        assert AntonHarness._stamp_message(m) == {
            "role": "assistant", "content": "I'm Anton, your AI coworker.",
        }

    def test_no_created_at_leaves_content_untouched(self):
        m = self._msg("user", "hi", None)
        assert AntonHarness._stamp_message(m) == {"role": "user", "content": "hi"}


class TestPersistHistoryCompaction:
    def test_noop_when_session_did_not_compact(self, svc):
        conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)
        fake_anton_session = SimpleNamespace(last_compaction=None)

        AntonHarness._persist_history_compaction(
            conv, fake_anton_session, {"ordered_messages": [], "tail_start": 0, "synthetic_prefix_len": 0},
        )

        assert svc.get_conversation(conv.id).history_summary is None

    def test_noop_on_anton_predating_last_compaction(self, svc):
        """cowork-server and anton deploy independently — an older anton
        `ChatSession` with no `last_compaction` property must no-op here,
        not raise."""
        conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)

        class _OldChatSession:
            pass  # no `last_compaction` attribute at all

        AntonHarness._persist_history_compaction(
            conv, _OldChatSession(), {"ordered_messages": [], "tail_start": 0, "synthetic_prefix_len": 0},
        )

        assert svc.get_conversation(conv.id).history_summary is None

    def test_persists_cutoff_on_first_compaction(self, svc):
        conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)
        ordered = _fake_messages(6)

        # No prior summary was replayed this turn (synthetic_prefix_len=0), so
        # covered_through indexes directly into ordered_messages.
        fake_anton_session = SimpleNamespace(
            last_compaction={"summary": "[COMPACTED] state record", "covered_through": 4}
        )
        seed_info = {"ordered_messages": ordered, "tail_start": 0, "synthetic_prefix_len": 0}

        AntonHarness._persist_history_compaction(conv, fake_anton_session, seed_info)

        refreshed = svc.get_conversation(conv.id)
        assert refreshed.history_summary == "[COMPACTED] state record"
        assert refreshed.history_summary_cutoff_id == ordered[3].id

    def test_persists_cutoff_accounting_for_separator_offset(self, svc):
        """A summary + separator were prepended this turn (synthetic_prefix_len
        = 2); covered_through counts both synthetic entries, so they must be
        subtracted before mapping onto ordered_messages."""
        conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)
        ordered = _fake_messages(6)

        # initial_history was [summary, separator, m2, m3, m4, m5] (tail_start=2);
        # covered_through=4 counts summary+separator+m2+m3 -> new cutoff m3.
        fake_anton_session = SimpleNamespace(
            last_compaction={"summary": "[COMPACTED] updated state", "covered_through": 4}
        )
        seed_info = {"ordered_messages": ordered, "tail_start": 2, "synthetic_prefix_len": 2}

        AntonHarness._persist_history_compaction(conv, fake_anton_session, seed_info)

        assert svc.get_conversation(conv.id).history_summary_cutoff_id == ordered[3].id

    def test_no_new_material_covered_does_not_persist(self, svc):
        """covered_through only reaches the synthetic summary entry itself
        (nothing real folded in) — nothing to persist."""
        conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)
        ordered = _fake_messages(3)

        fake_anton_session = SimpleNamespace(
            last_compaction={"summary": "irrelevant", "covered_through": 1}
        )
        seed_info = {"ordered_messages": ordered, "tail_start": 0, "synthetic_prefix_len": 1}

        AntonHarness._persist_history_compaction(conv, fake_anton_session, seed_info)

        assert svc.get_conversation(conv.id).history_summary is None
