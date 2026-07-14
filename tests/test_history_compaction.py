"""Tests for replaying anton's persisted history summary instead of full
history (ENG-664): `AntonHarness._seed_history` (build initial_history from
summary + tail, or fall back to full history) and
`AntonHarness._persist_history_compaction` (save the result after a turn).
"""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.common.settings.user_settings import UserSettings
from cowork.db.session import get_engine
from cowork.harnesses.anton_harness.harness import AntonHarness
from cowork.models.message import Message
from cowork.services.conversations import ConversationService
from cowork.services.projects import GENERAL_PROJECT_ID


def _stamp(m):
    return {"role": m.role, "content": m.id}


def _fake_messages(n: int, role: str = "user") -> list[SimpleNamespace]:
    return [SimpleNamespace(id=uuid4(), role=role) for _ in range(n)]


@pytest.fixture
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


class TestHistoryCompactionSetting:
    def test_enabled_by_default(self):
        assert UserSettings().history_compaction_enabled is True

    def test_can_be_disabled(self):
        assert UserSettings(history_compaction_enabled=False).history_compaction_enabled is False
        # `_build_chat_session` passes `None` for history_summary when this is
        # False, which is exactly `TestSeedHistory.test_no_summary_uses_full_history`.


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
        """Two consecutive `user`-role messages break/degrade most
        providers — if the tail starts with `user`, insert the same
        assistant separator anton's own _summarize_history uses."""
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


class TestPersistHistoryCompaction:
    def test_noop_on_anton_predating_last_compaction(self, session):
        """cowork-server and anton deploy independently — an older anton
        `ChatSession` with no `last_compaction` property must no-op here,
        not raise."""
        svc = ConversationService(session)
        conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)

        class _OldChatSession:
            pass  # no `last_compaction` attribute at all

        AntonHarness._persist_history_compaction(
            conv, _OldChatSession(), {"ordered_messages": [], "tail_start": 0, "synthetic_prefix_len": 0},
        )

        assert svc.get_conversation(conv.id).history_summary is None

    def test_noop_when_session_did_not_compact(self, session):
        svc = ConversationService(session)
        conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)
        fake_anton_session = SimpleNamespace(last_compaction=None)

        AntonHarness._persist_history_compaction(
            conv, fake_anton_session, {"ordered_messages": [], "tail_start": 0, "synthetic_prefix_len": 0},
        )

        assert svc.get_conversation(conv.id).history_summary is None

    def test_persists_cutoff_on_first_compaction(self, session):
        svc = ConversationService(session)
        conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)
        db_messages = [
            Message(conversation_id=conv.id, role="user" if i % 2 == 0 else "assistant", content=f"m{i}")
            for i in range(6)
        ]
        session.add_all(db_messages)
        session.commit()
        ordered_messages = svc.get_ordered_messages(conv.id)

        # No prior summary was replayed this turn (synthetic_prefix_len=0), so
        # covered_through indexes directly into ordered_messages.
        fake_anton_session = SimpleNamespace(
            last_compaction={"summary": "[COMPACTED] state record", "covered_through": 4}
        )
        seed_info = {"ordered_messages": ordered_messages, "tail_start": 0, "synthetic_prefix_len": 0}

        AntonHarness._persist_history_compaction(conv, fake_anton_session, seed_info)

        refreshed = svc.get_conversation(conv.id)
        assert refreshed.history_summary == "[COMPACTED] state record"
        assert refreshed.history_summary_cutoff_id == ordered_messages[3].id

    def test_persists_cutoff_accounting_for_replayed_summary_offset(self, session):
        """When a summary was already replayed as initial_history[0] this
        turn, `covered_through` counts that synthetic entry too — the cutoff
        must land on ordered_messages[tail_start + covered - offset - 1]."""
        svc = ConversationService(session)
        conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)
        db_messages = [
            Message(conversation_id=conv.id, role="user" if i % 2 == 0 else "assistant", content=f"m{i}")
            for i in range(6)
        ]
        session.add_all(db_messages)
        session.commit()
        ordered_messages = svc.get_ordered_messages(conv.id)

        # Previous cutoff was after message 1 (tail_start=2); this turn's
        # initial_history was [summary, m2, m3, m4, m5] (no separator needed,
        # synthetic_prefix_len=1) and compaction folded the summary + m2 + m3
        # in (covered_through=3 counting the synthetic summary entry),
        # leaving m4, m5 verbatim — new cutoff is m3.
        fake_anton_session = SimpleNamespace(
            last_compaction={"summary": "[COMPACTED] updated state", "covered_through": 3}
        )
        seed_info = {"ordered_messages": ordered_messages, "tail_start": 2, "synthetic_prefix_len": 1}

        AntonHarness._persist_history_compaction(conv, fake_anton_session, seed_info)

        refreshed = svc.get_conversation(conv.id)
        assert refreshed.history_summary == "[COMPACTED] updated state"
        assert refreshed.history_summary_cutoff_id == ordered_messages[3].id

    def test_persists_cutoff_accounting_for_separator_offset(self, session):
        """Same as above, but the tail started with `user` this turn, so
        `_seed_history` also inserted the assistant separator — two
        synthetic entries to subtract, not one."""
        svc = ConversationService(session)
        conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)
        db_messages = [
            Message(conversation_id=conv.id, role="user" if i % 2 == 0 else "assistant", content=f"m{i}")
            for i in range(6)
        ]
        session.add_all(db_messages)
        session.commit()
        ordered_messages = svc.get_ordered_messages(conv.id)

        # initial_history was [summary, separator, m2, m3, m4, m5];
        # covered_through=4 counts summary+separator+m2+m3 -> new cutoff m3.
        fake_anton_session = SimpleNamespace(
            last_compaction={"summary": "[COMPACTED] updated state", "covered_through": 4}
        )
        seed_info = {"ordered_messages": ordered_messages, "tail_start": 2, "synthetic_prefix_len": 2}

        AntonHarness._persist_history_compaction(conv, fake_anton_session, seed_info)

        refreshed = svc.get_conversation(conv.id)
        assert refreshed.history_summary_cutoff_id == ordered_messages[3].id

    def test_no_new_material_covered_does_not_persist(self, session):
        """covered_through only reaches the synthetic summary entry itself
        (nothing new folded in) — nothing to persist."""
        svc = ConversationService(session)
        conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)
        db_messages = [Message(conversation_id=conv.id, role="user", content="m0")]
        session.add_all(db_messages)
        session.commit()
        ordered_messages = svc.get_ordered_messages(conv.id)

        fake_anton_session = SimpleNamespace(
            last_compaction={"summary": "irrelevant", "covered_through": 1}
        )
        seed_info = {"ordered_messages": ordered_messages, "tail_start": 0, "synthetic_prefix_len": 1}

        AntonHarness._persist_history_compaction(conv, fake_anton_session, seed_info)

        assert svc.get_conversation(conv.id).history_summary is None
