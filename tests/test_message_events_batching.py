"""ENG-2768: get_messages fetched every message's MessageEvents in its own
query (an N+1). _hydrate_message_items batches them into one
`WHERE message_id IN (...)` query, grouped by message_id in Python — this
only regression-tests that grouping preserves each message's own event
order, since the batched query drops the per-message ORDER BY the old
per-message query relied on.
"""
from __future__ import annotations

import pytest
from sqlmodel import Session, SQLModel, create_engine

from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.models.conversation import Conversation
from cowork.models.message import Message
from cowork.models.message_event import MessageEvent
from cowork.models.project import Project
from cowork.schemas.responses import Role
from cowork.services.conversations import ConversationService


@pytest.fixture
def engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def session(engine):
    with Session(engine) as raw:
        yield ScopedSession(raw, LOCAL_SCOPE)


@pytest.fixture
def conversation(tmp_path, session):
    project = Project(name="P", path=str(tmp_path / "p"))
    session.add(project)
    session.commit()
    session.refresh(project)
    conv = Conversation(project_id=project.id, topic="t")
    session.add(conv)
    session.commit()
    session.refresh(conv)
    return conv


def _add_message(session, conversation, *, role, content, seq, events):
    message = Message(
        conversation_id=conversation.id, role=role, content=content, seq=seq
    )
    session.add(message)
    session.commit()
    session.refresh(message)
    for i, event_data in enumerate(events):
        session.add(
            MessageEvent(message_id=message.id, sequence_number=i, event_data=event_data)
        )
    session.commit()
    return message


def test_batched_events_stay_in_order_per_message(session, conversation):
    # Two messages, each with several events, inserted so the events are NOT
    # already grouped/ordered by (message_id, sequence_number) accidentally —
    # message A's events are added out of natural id order relative to B's.
    a = _add_message(
        session, conversation, role=Role.assistant, content="a", seq=0,
        events=[{"step": 0}, {"step": 1}, {"step": 2}],
    )
    b = _add_message(
        session, conversation, role=Role.assistant, content="b", seq=1,
        events=[{"step": 0}, {"step": 1}],
    )

    items = ConversationService(session).get_messages(conversation.id)

    by_id = {item["id"]: item for item in items}
    assert [e["step"] for e in by_id[a.id]["events"]] == [0, 1, 2]
    assert [e["step"] for e in by_id[b.id]["events"]] == [0, 1]


def test_message_with_no_events_gets_empty_list(session, conversation):
    m = _add_message(
        session, conversation, role=Role.user, content="hi", seq=0, events=[]
    )
    items = ConversationService(session).get_messages(conversation.id)
    assert items[0]["id"] == m.id
    assert items[0]["events"] == []
