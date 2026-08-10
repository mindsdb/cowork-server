"""save_user_message honours an explicit `created_at` (the turn's send time)
instead of the DB `func.now()` default, which — with deferred persistence —
would stamp the message with the turn's END time. Keeps the stored stamp
aligned with the agent's live-turn timestamp (ENG-1092 follow-up)."""
from __future__ import annotations

from datetime import datetime, timezone

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


def _conversation(session):
    project = ProjectService(ScopedSession(session, LOCAL_SCOPE)).create_project("p")
    return svc_conv(session, project.id)


def svc_conv(session, project_id):
    return ConversationService(
        ScopedSession(session, LOCAL_SCOPE)
    ).create_conversation("t", project_id=project_id)


def test_explicit_created_at_is_stored(session, svc):
    conv = _conversation(session)
    sent = datetime(2026, 8, 3, 15, 14, tzinfo=timezone.utc)

    msg = svc.save_user_message(conv.id, '"hi"', created_at=sent)

    # SQLite drops tzinfo on readback; compare the UTC wall-clock the stamp uses.
    assert msg.created_at.strftime("%Y-%m-%d %H:%M") == "2026-08-03 15:14"


def test_omitted_created_at_falls_back_to_db_default(session, svc):
    conv = _conversation(session)

    msg = svc.save_user_message(conv.id, '"hi"')

    assert msg.created_at is not None  # server_default func.now() populated it
