"""Deleting a conversation (or clearing all its history) must drop the
conversation's object index too.

Regression: a stale `task_objects` row outlived the chat that produced it,
so a cleared conversation kept resurfacing an artifact it no longer owned —
even after the artifact file itself was gone.
"""
from __future__ import annotations

import pytest
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.task_object import TaskObject
from cowork.services.conversations import ConversationService
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.task_objects import TaskObjectService


@pytest.fixture
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


def _rows_for(session, conversation_id) -> list[TaskObject]:
    return list(
        session.exec(
            select(TaskObject).where(TaskObject.conversation_id == conversation_id)
        ).all()
    )


def test_delete_conversation_drops_task_objects(session):
    svc = ConversationService(session)
    conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)
    TaskObjectService(session).index_artifact(conv.id, GENERAL_PROJECT_ID, "my-artifact")
    assert _rows_for(session, conv.id), "precondition: artifact indexed"

    assert svc.delete_conversation(conv.id) is True
    assert _rows_for(session, conv.id) == [], "index rows must be gone with the conversation"


def test_clear_all_history_drops_task_objects(session):
    """Truncating from turn 0 (the UI's 'delete chat history') leaves the
    conversation but removes everything — its object index goes too."""
    svc = ConversationService(session)
    conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)
    # One full turn: user message + assistant turn (mirrors a real exchange).
    from cowork.models.message import Message

    session.add(Message(conversation_id=conv.id, role="user", content='"make a plan"'))
    session.commit()
    svc.save_assistant_turn(conv.id, "done", events=[])
    TaskObjectService(session).index_artifact(conv.id, GENERAL_PROJECT_ID, "my-artifact")
    assert _rows_for(session, conv.id), "precondition: artifact indexed"

    svc.delete_turn(conv.id, 0)  # clear from the first turn = clear all history
    assert _rows_for(session, conv.id) == [], "cleared history must drop the index"


def test_partial_turn_delete_keeps_task_objects(session):
    """Deleting a later turn (not turn 0) is a partial truncation — the
    surviving turns may still reference the artifact, so leave the index."""
    svc = ConversationService(session)
    conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)
    from cowork.models.message import Message

    session.add(Message(conversation_id=conv.id, role="user", content='"first"'))
    session.commit()
    svc.save_assistant_turn(conv.id, "a1", events=[])
    session.add(Message(conversation_id=conv.id, role="user", content='"second"'))
    session.commit()
    svc.save_assistant_turn(conv.id, "a2", events=[])
    TaskObjectService(session).index_artifact(conv.id, GENERAL_PROJECT_ID, "my-artifact")

    svc.delete_turn(conv.id, 1)  # drop only the second turn
    assert _rows_for(session, conv.id), "partial delete must keep the index"
