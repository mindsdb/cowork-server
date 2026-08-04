"""ENG-1231: the current turn's user message is persisted immediately, marked
pending, so a refresh mid-turn still shows the question — but it's kept out of
replayed LLM history until the turn ends.

  - get_messages (the UI /items view) INCLUDES pending rows → the question shows.
  - get_ordered_messages (LLM-history replay) EXCLUDES pending rows → the harness
    doesn't double-feed the current input.
  - finalize_pending clears the flag → the finished turn rejoins history.
"""
from __future__ import annotations

import pytest
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_engine
from cowork.services.conversations import ConversationService
from cowork.services.projects import ProjectService


@pytest.fixture
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


@pytest.fixture
def svc(session):
    return ConversationService(ScopedSession(session, LOCAL_SCOPE))


@pytest.fixture
def conv(session):
    scoped = ScopedSession(session, LOCAL_SCOPE)
    project = ProjectService(scoped).create_project("pending-test")
    return ConversationService(scoped).create_conversation("t", project_id=project.id)


def test_pending_user_message_shows_in_ui_but_not_in_llm_history(svc, conv):
    svc.save_user_message(conv.id, "hello?", pending=True)

    # UI view shows the question immediately — this is what a mid-turn reload reads.
    ui = svc.get_messages(conv.id)
    assert [m["role"] for m in ui] == ["user"]
    assert ui[0]["content"] == "hello?"

    # LLM-history replay excludes it, so the harness won't double-feed the input.
    assert svc.get_ordered_messages(conv.id) == []
    # ...unless a caller explicitly opts in.
    assert len(svc.get_ordered_messages(conv.id, include_pending=True)) == 1


def test_finalize_pending_returns_the_turn_to_history(svc, conv):
    svc.save_user_message(conv.id, "hello?", pending=True)
    assert svc.get_ordered_messages(conv.id) == []

    svc.finalize_pending(conv.id)

    assert [m.role for m in svc.get_ordered_messages(conv.id)] == ["user"]
    # Still visible in the UI, now finalized.
    assert [m["role"] for m in svc.get_messages(conv.id)] == ["user"]


def test_finalize_pending_scoped_to_message_id_leaves_other_pending_rows(svc, conv):
    # A hard crash (killed between the pending persist and finalize) can strand a
    # pending row. A later turn must finalize ONLY its own row — else it folds the
    # orphaned, unanswered question into replayed LLM history. The producers pass
    # message_id for exactly this reason.
    stranded = svc.save_user_message(conv.id, "orphaned?", pending=True)
    current = svc.save_user_message(conv.id, "answered?", pending=True)

    svc.finalize_pending(conv.id, current.id)

    # Only the current turn's row rejoined history; the orphan stays pending.
    hist_contents = [m.content for m in svc.get_ordered_messages(conv.id)]
    assert hist_contents == ["answered?"]
    pending_only = svc.get_ordered_messages(conv.id, include_pending=True)
    assert {m.content: m.pending for m in pending_only} == {
        "orphaned?": True,
        "answered?": False,
    }
    # ...and both remain visible in the UI regardless of pending state.
    assert {m["content"] for m in svc.get_messages(conv.id)} == {"orphaned?", "answered?"}
    assert stranded.id != current.id


def test_finalize_pending_is_idempotent_and_safe_when_nothing_pending(svc, conv):
    svc.finalize_pending(conv.id)  # nothing pending → no-op, no error
    svc.save_user_message(conv.id, "done", pending=False)  # already final
    svc.finalize_pending(conv.id)
    assert len(svc.get_ordered_messages(conv.id)) == 1


def test_non_pending_user_message_is_in_history_immediately(svc, conv):
    # The non-streaming / channels path persists without the flag.
    svc.save_user_message(conv.id, "hi", pending=False)
    assert len(svc.get_ordered_messages(conv.id)) == 1


def test_finalize_before_empty_assistant_turn_still_finalizes(svc, conv):
    # The producers call finalize_pending BEFORE save_assistant_turn. On an empty
    # turn, save_assistant_turn early-returns (no text/events/tool_rows) and writes
    # nothing — but the pending flag must already be cleared, so the question isn't
    # stranded out of history.
    svc.save_user_message(conv.id, "hello?", pending=True)
    svc.finalize_pending(conv.id)
    svc.save_assistant_turn(conv.id, "", [], tool_rows=None)  # early-returns, writes nothing

    hist = svc.get_ordered_messages(conv.id)
    assert [m.role for m in hist] == ["user"]  # user finalized; no assistant row written
