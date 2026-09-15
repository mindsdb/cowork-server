"""ENG-2768: delete_turn is anchored by message id instead of a positional
turn_index. The old index was resolved against whatever the CLIENT
happened to have loaded — on a lazily-paginated long conversation that's
only ever a partial view, so a client-computed index could name the wrong
absolute turn server-side and delete the wrong range of history. Anchoring
by the actual message id removes that class of bug: the server looks the
row up directly, there is nothing for a client's partial view to get wrong.
"""
from __future__ import annotations

import uuid

import pytest
from sqlmodel import Session, SQLModel, create_engine

from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.models.conversation import Conversation
from cowork.models.project import Project
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


def _turn(svc, conv, user_text, answer):
    """A plain (no tool rows) user+assistant turn via the real service path,
    returning the assistant Message so tests can anchor delete_turn on it."""
    svc.save_user_message(conv.id, user_text)
    svc.save_assistant_turn(conv.id, answer, [])
    return next(
        m for m in svc.get_ordered_messages(conv.id)
        if m.role.value == "assistant" and m.content == answer
    )


def _visible_pairs(svc, conv):
    return [(m["role"].value, m["content"]) for m in svc.get_messages(conv.id)]


def test_delete_by_assistant_id_removes_that_turn_and_everything_after(session, conversation):
    svc = ConversationService(session)
    a1 = _turn(svc, conversation, "q1", "a1")
    _turn(svc, conversation, "q2", "a2")
    _turn(svc, conversation, "q3", "a3")

    a2 = next(m for m in svc.get_ordered_messages(conversation.id) if m.content == "a2")
    deleted = svc.delete_turn(conversation.id, a2.id)

    assert deleted == 4  # q2, a2, and everything after (q3, a3)
    assert _visible_pairs(svc, conversation) == [("user", "q1"), ("assistant", "a1")]


def test_delete_an_early_turn_of_a_long_conversation_cuts_exactly_there(session, conversation):
    """The server's cut point comes from the message's real stored position,
    not from anything a client believed about array length — proving a
    partially-loaded client can never cause the wrong range to be deleted."""
    svc = ConversationService(session)
    turns = [_turn(svc, conversation, f"q{i}", f"a{i}") for i in range(10)]

    early = turns[2]  # 3rd turn out of 10
    svc.delete_turn(conversation.id, early.id)

    remaining = _visible_pairs(svc, conversation)
    assert remaining == [("user", "q0"), ("assistant", "a0"), ("user", "q1"), ("assistant", "a1")]


def test_delete_orphan_turn_by_user_message_id(session, conversation):
    svc = ConversationService(session)
    _turn(svc, conversation, "q1", "a1")
    orphan = svc.save_user_message(conversation.id, "q2-never-answered")

    svc.delete_turn(conversation.id, orphan.id)

    assert _visible_pairs(svc, conversation) == [("user", "q1"), ("assistant", "a1")]


def test_delete_rejects_a_user_message_that_already_has_a_reply(session, conversation):
    svc = ConversationService(session)
    _turn(svc, conversation, "q1", "a1")
    answered_user = next(m for m in svc.get_ordered_messages(conversation.id) if m.role.value == "user")

    with pytest.raises(ValueError):
        svc.delete_turn(conversation.id, answered_user.id)
    # nothing was removed
    assert _visible_pairs(svc, conversation) == [("user", "q1"), ("assistant", "a1")]


def test_delete_rejects_a_hidden_tool_row_id(session, conversation):
    svc = ConversationService(session)
    svc.save_user_message(conversation.id, "q1")
    svc.save_assistant_turn(
        conversation.id, "a1", [],
        tool_rows=[
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "x", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "r"}]},
        ],
    )
    tool_row = next(
        m for m in svc.get_ordered_messages(conversation.id)
        if isinstance(m.content, list) and m.content and m.content[0].get("type") == "tool_use"
    )
    with pytest.raises(ValueError):
        svc.delete_turn(conversation.id, tool_row.id)


def test_unknown_and_foreign_conversation_message_ids_fail_the_same_way(tmp_path, engine):
    org_a = ScopedSession(Session(engine), TenantScope(org_mode=True, org_id="org-a", user_id="user-a"))
    org_b = ScopedSession(Session(engine), TenantScope(org_mode=True, org_id="org-b", user_id="user-b"))

    project = Project(name="P", path=str(tmp_path / "p"))
    org_a.add(project)
    org_a.commit()
    org_a.refresh(project)
    conv = Conversation(project_id=project.id, topic="t")
    org_a.add(conv)
    org_a.commit()
    org_a.refresh(conv)
    svc_a = ConversationService(org_a)
    a1 = _turn(svc_a, conv, "q1", "a1")

    with pytest.raises(ValueError) as foreign_exc:
        ConversationService(org_b).delete_turn(conv.id, a1.id)
    with pytest.raises(ValueError) as unknown_exc:
        ConversationService(org_b).delete_turn(uuid.uuid4(), uuid.uuid4())
    assert str(foreign_exc.value) == str(unknown_exc.value)
    # nothing was actually removed by the rejected foreign attempt
    assert _visible_pairs(svc_a, conv) == [("user", "q1"), ("assistant", "a1")]
