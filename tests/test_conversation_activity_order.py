"""A conversation's "recent" position must track its last *activity*, not its
creation time.

Regression (ENG-961): conversation.modified_at only moves on rename/move, never
on a turn, so ordering by it (or by created_at) sent actively-used older
conversations to the bottom after an app restart. Last-activity is derived from
MAX(message.created_at) instead. These tests pin that derivation + ordering.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_engine
from cowork.models.message import Message
from cowork.services.conversations import ConversationService
from cowork.services.projects import ProjectService

# Naive UTC: SQLite's DateTime storage drops tzinfo, so readback is naive.
_BASE = datetime(2026, 1, 1, 12, 0, 0)


@pytest.fixture
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


@pytest.fixture
def svc(session):
    return ConversationService(ScopedSession(session, LOCAL_SCOPE))


def _project(session, name):
    return ProjectService(ScopedSession(session, LOCAL_SCOPE)).create_project(name)


def _conv_created_at(session, svc, project_id, topic, created_at):
    conv = svc.create_conversation(topic, project_id=project_id)
    conv.created_at = created_at
    session.add(conv)
    session.commit()
    return conv


def _add_message(session, conversation_id, created_at, seq=0):
    session.add(Message(
        conversation_id=conversation_id,
        role="user",
        content='"hi"',
        created_at=created_at,
        seq=seq,
    ))
    session.commit()


def test_recent_activity_outranks_recent_creation(session, svc):
    """An OLD conversation with a fresh message sorts above a NEWER one that's
    been idle — the exact case that broke after a restart."""
    proj = _project(session, "eng961-order")
    old_active = _conv_created_at(session, svc, proj.id, "old-active", _BASE)
    new_idle = _conv_created_at(session, svc, proj.id, "new-idle", _BASE + timedelta(days=5))

    # old_active gets a message AFTER new_idle was even created.
    _add_message(session, old_active.id, _BASE + timedelta(days=10))
    _add_message(session, new_idle.id, _BASE + timedelta(days=5))

    ordered = svc.list_conversations_with_activity(project_id=proj.id)
    ids = [c.id for c, _ in ordered]
    assert ids.index(old_active.id) < ids.index(new_idle.id)


def test_activity_is_latest_message_time(session, svc):
    proj = _project(session, "eng961-latest")
    conv = _conv_created_at(session, svc, proj.id, "conv", _BASE)
    _add_message(session, conv.id, _BASE + timedelta(hours=1), seq=0)
    _add_message(session, conv.id, _BASE + timedelta(hours=3), seq=1)  # latest
    _add_message(session, conv.id, _BASE + timedelta(hours=2), seq=2)

    (returned_conv, activity), = svc.list_conversations_with_activity(project_id=proj.id)
    assert returned_conv.id == conv.id
    assert activity == _BASE + timedelta(hours=3)
    assert svc.last_message_at(conv.id) == _BASE + timedelta(hours=3)


def test_empty_conversation_falls_back_to_created_at(session, svc):
    proj = _project(session, "eng961-empty")
    conv = _conv_created_at(session, svc, proj.id, "empty", _BASE)

    # No messages: derived activity coalesces to created_at, not NULL.
    (_, activity), = svc.list_conversations_with_activity(project_id=proj.id)
    assert activity == _BASE
    assert svc.last_message_at(conv.id) is None


def test_endpoint_serializes_activity_and_orders_by_it(session, svc):
    """Full chain over HTTP: the list JSON is ordered by activity and each
    row's `updated_at` reflects the latest message, not creation time."""
    from fastapi.testclient import TestClient

    from cowork.server import app

    proj = _project(session, "eng961-endpoint")
    old_active = _conv_created_at(session, svc, proj.id, "old-active", _BASE)
    new_idle = _conv_created_at(session, svc, proj.id, "new-idle", _BASE + timedelta(days=5))
    _add_message(session, old_active.id, _BASE + timedelta(days=10))

    client = TestClient(app, client=("127.0.0.1", 50001))
    r = client.get(f"/api/v1/conversations/?project_id={proj.id}")
    assert r.status_code == 200
    convs = r.json()["conversations"]
    ids = [c["id"] for c in convs]
    assert ids.index(str(old_active.id)) < ids.index(str(new_idle.id))
    active_row = next(c for c in convs if c["id"] == str(old_active.id))
    assert active_row["updatedAt"].startswith("2026-01-11")  # _BASE + 10 days

    # Single-get reflects the same derived activity.
    g = client.get(f"/api/v1/conversations/{old_active.id}")
    assert g.status_code == 200
    assert g.json()["updatedAt"].startswith("2026-01-11")
