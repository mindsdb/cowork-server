"""sidekick_contexts: a conversation's Chimera sidekick, kept between turns.

anton hands the sidekick out as a dict (ChatSession.sidekick_state) and takes
the same dict back on the next session. The service maps it onto one row per
conversation and back, reaches the row only through a conversation the caller
owns, scrubs what goes back to a model, and the row goes with its conversation
or with the turns a rewind deletes.
"""
from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest
from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine

from anton.utils.datasources import _DS_KNOWN_VARS, _DS_SECRET_VARS
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.models.sidekick_context import SidekickContext
from cowork.services.conversations import ConversationService
from cowork.services.sidekick_contexts import SidekickContextService, scrubbed_state


@pytest.fixture
def engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'sidekick.db'}")

    # Foreign keys on, as PostgreSQL has them: the delete order is checked.
    @event.listens_for(engine, "connect")
    def _foreign_keys(dbapi_connection, _record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

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


@pytest.fixture
def clean_ds_state():
    def _clean():
        _DS_SECRET_VARS.clear()
        _DS_KNOWN_VARS.clear()
        for key in list(os.environ):
            if key.startswith("DS_"):
                del os.environ[key]

    _clean()
    yield
    _clean()


def _state(**overrides) -> dict:
    return {
        "sidekick_id": "sk-1",
        "model": "gpt-sol",
        "messages": [
            {"role": "user", "content": "Count the rows in sales.csv"},
            {"role": "assistant", "content": [{"type": "text", "text": "12,034 rows"}]},
        ],
        "summary": "Counted rows.",
        "status": "open",
        "last_call_at": "2026-11-04T10:00:00+00:00",
        "signals": 1,
        **overrides,
    }


def test_a_saved_state_loads_back_as_anton_left_it(session, conversation):
    service = SidekickContextService(session)
    assert service.load(conversation.id) is None

    assert service.save(conversation.id, _state()) is True

    assert service.load(conversation.id) == _state()


def test_a_second_save_replaces_the_row(session, conversation):
    service = SidekickContextService(session)
    service.save(conversation.id, _state())
    service.save(conversation.id, _state(status="idle", messages=[], summary=None, signals=0))

    loaded = service.load(conversation.id)
    assert loaded["status"] == "idle"
    assert loaded["messages"] == []
    assert loaded["summary"] is None
    assert len(session.exec(session.select(SidekickContext)).all()) == 1


def test_last_call_at_comes_back_as_utc_iso(session, conversation):
    # SQLite keeps no offset; the instant is UTC whichever way it went in.
    when = datetime(2026, 11, 4, 12, 30, tzinfo=UTC)
    service = SidekickContextService(session)
    service.save(conversation.id, _state(last_call_at="2026-11-04T13:30:00+01:00"))

    assert service.load(conversation.id)["last_call_at"] == when.isoformat()


def test_an_unknown_status_is_kept_as_idle(session, conversation):
    service = SidekickContextService(session)
    service.save(conversation.id, _state(status="thinking"))

    assert service.load(conversation.id)["status"] == "idle"
    assert service.is_open(conversation) is False


def test_open_means_the_sidekick_handed_its_work_back(session, conversation):
    service = SidekickContextService(session)
    assert service.is_open(conversation) is False
    service.save(conversation.id, _state(status="open"))
    assert service.is_open(conversation) is True


@pytest.mark.parametrize("state", [None, "state", {"messages": "not a list"}])
def test_a_malformed_state_writes_nothing(session, conversation, state):
    assert SidekickContextService(session).save(conversation.id, state) is False
    assert SidekickContextService(session).load(conversation.id) is None


def test_a_conversation_out_of_reach_has_no_sidekick(session):
    import uuid

    missing = uuid.uuid4()
    service = SidekickContextService(session)
    assert service.save(missing, _state()) is False
    assert service.load(missing) is None


def test_replay_scrubs_a_secret_vaulted_since_the_state_was_saved(clean_ds_state, monkeypatch):
    secret = "SuperSecret123!"
    state = _state(
        messages=[
            {"role": "user", "content": f"the password is {secret}"},
            {"role": "assistant", "content": [{"type": "text", "text": f"using {secret}"}]},
        ],
        summary=f"logged in with {secret}",
    )
    _DS_SECRET_VARS.add("DS_POSTGRES_ABC12__PASSWORD")
    monkeypatch.setenv("DS_POSTGRES_ABC12__PASSWORD", secret)

    scrubbed = scrubbed_state(state)

    assert secret not in str(scrubbed["messages"])
    assert secret not in scrubbed["summary"]
    # A copy: the stored state is left as it was.
    assert secret in state["messages"][0]["content"]


def test_the_row_goes_with_its_conversation(session, conversation):
    SidekickContextService(session).save(conversation.id, _state())

    assert ConversationService(session).delete_conversation(conversation.id) is True

    assert session.get(SidekickContext, conversation.id) is None


def test_a_history_rewind_starts_the_sidekick_fresh(session, conversation):
    conversations = ConversationService(session)
    conversations.save_user_message(conversation.id, "q1")
    conversations.save_assistant_turn(conversation.id, "a1", [])
    SidekickContextService(session).save(conversation.id, _state())
    answer = next(m for m in conversations.get_ordered_messages(conversation.id) if m.content == "a1")

    conversations.delete_turn(conversation.id, answer.id)

    assert SidekickContextService(session).load(conversation.id) is None
