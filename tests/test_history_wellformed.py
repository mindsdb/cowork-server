"""History stays well-formed even when a turn produces no text.

Regression for the "consecutive Role.user" defect: a failed/empty
assistant turn used to be dropped entirely, leaving the already-committed
user message unpaired, so the next user message produced two consecutive
user rows. save_assistant_turn now persists a placeholder assistant row
instead, keeping (user, assistant) pairing intact.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.message import Message
from cowork.services.conversations import ConversationService
from cowork.services.projects import GENERAL_PROJECT_ID


@pytest.fixture()
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


# get_messages orders by (created_at, id). In production a user message is
# committed seconds before its assistant turn finishes; the tests below
# stamp explicit, monotonically increasing timestamps so ordering is
# deterministic rather than relying on same-instant tie-breaking by UUID.
# Well in the past so user rows always precede the assistant row, which
# save_assistant_turn stamps with the real wall-clock now().
_T0 = datetime(2020, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _add_user(session, conv_id, content):
    session.add(Message(conversation_id=conv_id, role="user", content=content,
                        created_at=_T0))
    session.commit()


def _role_counts(svc: ConversationService, conv_id):
    counts = {"user": 0, "assistant": 0}
    for m in svc.get_messages(conv_id):
        r = m["role"].value if hasattr(m["role"], "value") else m["role"]
        counts[str(r)] = counts.get(str(r), 0) + 1
    return counts


def test_empty_turn_with_events_still_pairs(session):
    svc = ConversationService(session)
    conv = svc.create_conversation(topic="t", project_id=GENERAL_PROJECT_ID)
    _add_user(session, conv.id, "hi")

    # A turn that yielded events but no text (e.g. a failed tool loop) —
    # previously dropped, now persisted as a placeholder assistant row.
    svc.save_assistant_turn(conv.id, "", [{"type": "response.completed"}], harness="anton")

    # Invariant: every user turn is answered by an assistant row, so a
    # second user turn can never sit directly after the first (the
    # "consecutive Role.user" defect). Assert by pairing, not by fragile
    # timestamp ordering (save_assistant_turn stamps its own now()).
    assert _role_counts(svc, conv.id) == {"user": 1, "assistant": 1}

    _add_user(session, conv.id, "again")
    svc.save_assistant_turn(conv.id, "real answer", [], harness="anton")
    assert _role_counts(svc, conv.id) == {"user": 2, "assistant": 2}


def test_truly_empty_turn_is_skipped(session):
    # No text AND no events == nothing happened; still skipped so we don't
    # spam placeholder rows for genuine no-ops.
    svc = ConversationService(session)
    conv = svc.create_conversation(topic="t", project_id=GENERAL_PROJECT_ID)
    _add_user(session, conv.id, "hi")
    svc.save_assistant_turn(conv.id, "", [], harness="anton")
    assert _role_counts(svc, conv.id) == {"user": 1, "assistant": 0}
