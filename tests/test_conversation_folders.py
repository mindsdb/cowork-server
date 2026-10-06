"""Working folders attached to a desktop chat."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, select

from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.models.conversation_folder import ConversationFolder
from cowork.models.message import Message
from cowork.models.project import Project
from cowork.services.conversations import ConversationService
from cowork.services.projects import ProjectService


def _fk_enforcing_engine():
    """Foreign keys on, as on Postgres, so a folder row left behind refuses the
    conversation delete instead of silently orphaning."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    SQLModel.metadata.create_all(engine)
    return engine


def _folders(session: Session, conversation_id) -> list[ConversationFolder]:
    return list(
        session.exec(
            select(ConversationFolder).where(ConversationFolder.conversation_id == conversation_id)
        ).all()
    )


def _chat_with_folder(session: Session, tmp_path: Path, name: str):
    project = Project(name=name, path=str(tmp_path / name))
    session.add(project)
    session.commit()
    scoped = ScopedSession(session, LOCAL_SCOPE)
    conversation = ConversationService(scoped).create_conversation("topic", project_id=project.id)
    scoped.add(ConversationFolder(conversation_id=conversation.id, path=str(tmp_path / "docs")))
    session.commit()
    return project, conversation


def test_deleting_a_chat_removes_its_folders(tmp_path):
    with Session(_fk_enforcing_engine(), expire_on_commit=False) as session:
        _project, conversation = _chat_with_folder(session, tmp_path, "folders-chat")
        assert _folders(session, conversation.id)

        scoped = ScopedSession(session, LOCAL_SCOPE)
        assert ConversationService(scoped).delete_conversation(conversation.id) is True

        assert _folders(session, conversation.id) == []


def test_deleting_a_project_removes_its_chats_folders(tmp_path):
    with Session(_fk_enforcing_engine(), expire_on_commit=False) as session:
        project, conversation = _chat_with_folder(session, tmp_path, "folders-project")

        assert ProjectService(ScopedSession(session, LOCAL_SCOPE)).delete_project(project.id) is True

        assert _folders(session, conversation.id) == []


def test_clearing_a_chats_history_keeps_its_folders(tmp_path):
    """Folders are the user's setting for the chat, not something a turn made."""
    with Session(_fk_enforcing_engine(), expire_on_commit=False) as session:
        _project, conversation = _chat_with_folder(session, tmp_path, "folders-history")
        scoped = ScopedSession(session, LOCAL_SCOPE)
        service = ConversationService(scoped)
        session.add(Message(conversation_id=conversation.id, role="user", content='"hi"'))
        session.commit()
        service.save_assistant_turn(conversation.id, "hello", events=[])
        first_answer = next(
            m.id
            for m in session.exec(
                select(Message).where(
                    Message.conversation_id == conversation.id, Message.role == "assistant"
                )
            ).all()
        )

        service.delete_turn(conversation.id, first_answer)

        assert len(_folders(session, conversation.id)) == 1
