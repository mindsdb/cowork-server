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


# ── ENG-701: attachment cleanup on conversation / project delete ──────────
from pathlib import Path  # noqa: E402

from cowork.models.conversation import Conversation  # noqa: E402
from cowork.models.project import Project  # noqa: E402
from cowork.services.files import FileService, attachment_purpose, unlink_file_dirs  # noqa: E402
from cowork.services.projects import ProjectService  # noqa: E402


def _attach(session, conversation_id, name="doc.txt"):
    return FileService(session).create_file_from_bytes(
        filename=name, content_type="text/plain", data=b"hello",
        purpose=attachment_purpose(str(conversation_id)),
    )


def _attachment_rows(session, conversation_id):
    return FileService(session).list_file_rows(attachment_purpose(str(conversation_id)))


def test_delete_conversation_removes_attachment_rows_and_bytes(session):
    svc = ConversationService(session)
    conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)
    f = _attach(session, conv.id)
    path = Path(f.path)
    assert path.exists() and _attachment_rows(session, conv.id), "precondition"

    assert svc.delete_conversation(conv.id) is True
    assert _attachment_rows(session, conv.id) == [], "rows gone with the conversation"
    assert not path.exists(), "bytes unlinked"


def test_delete_conversation_leaves_other_conversations_and_purposes(session):
    svc = ConversationService(session)
    keep = svc.create_conversation("keep", project_id=GENERAL_PROJECT_ID)
    doomed = svc.create_conversation("doomed", project_id=GENERAL_PROJECT_ID)
    keep_file = _attach(session, keep.id)
    _attach(session, doomed.id)
    # A non-attachment purpose must never be touched.
    other = FileService(session).create_file_from_bytes(
        filename="c.bin", content_type="application/octet-stream",
        data=b"x", purpose="channel:some-channel",
    )

    svc.delete_conversation(doomed.id)

    assert _attachment_rows(session, keep.id) and Path(keep_file.path).exists()
    assert FileService(session).list_file_rows("channel:some-channel"), "non-attachment untouched"
    assert Path(other.path).exists()


def test_delete_project_cascades_conversations_and_attachments(session):
    proj = ProjectService(session).create_project("eng701-cascade-test")
    svc = ConversationService(session)
    conv = svc.create_conversation("topic", project_id=proj.id)
    f = _attach(session, conv.id)
    path = Path(f.path)
    assert path.exists()

    assert ProjectService(session).delete_project(proj.id) is True
    # The conversation itself is gone (no more orphaned rows) …
    assert session.get(Conversation, conv.id) is None
    # … along with its attachment rows + bytes.
    assert _attachment_rows(session, conv.id) == []
    assert not path.exists()


def test_delete_project_survives_one_conversation_delete_failure(session, monkeypatch):
    """Fault isolation: if one conversation fails to delete, the project delete
    must still complete and clean up the rest — not abort half-cascaded."""
    proj = ProjectService(session).create_project("eng701-fault-test")
    svc = ConversationService(session)
    bad = svc.create_conversation("bad", project_id=proj.id)
    good = svc.create_conversation("good", project_id=proj.id)
    _attach(session, bad.id)
    good_file = _attach(session, good.id)

    real = ConversationService.delete_conversation

    def flaky(self, cid):
        if str(cid) == str(bad.id):
            raise RuntimeError("boom")
        return real(self, cid)

    monkeypatch.setattr(ConversationService, "delete_conversation", flaky)

    assert ProjectService(session).delete_project(proj.id) is True
    assert session.get(Project, proj.id) is None, "project deleted despite one failure"
    # The good conversation + its attachment were still cleaned …
    assert session.get(Conversation, good.id) is None
    assert _attachment_rows(session, good.id) == []
    assert not Path(good_file.path).exists()
    # … the failed one is skipped (logged), left as it was — not fatal.
    assert session.get(Conversation, bad.id) is not None


def test_delete_by_purpose_stages_without_committing(session):
    """The attachment-row delete must land in the CALLER's transaction, not its
    own — otherwise a crash between it and the conversation-row delete leaves a
    'ghost' conversation (row present, contents gone). Proof: after
    delete_by_purpose, a rollback brings the rows back, and the bytes are still
    on disk (unlink is the caller's post-commit step)."""
    svc = ConversationService(session)
    conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)
    _attach(session, conv.id)

    dirs = FileService(session).delete_by_purpose(attachment_purpose(str(conv.id)))
    session.rollback()

    assert _attachment_rows(session, conv.id), "delete_by_purpose must not commit on its own"
    assert dirs and all(d.exists() for d in dirs), "bytes must survive until the caller commits"

    # And the helper only removes bytes once called explicitly (post-commit).
    FileService(session).delete_by_purpose(attachment_purpose(str(conv.id)))
    session.commit()
    unlink_file_dirs(dirs)
    assert not any(d.exists() for d in dirs), "bytes removed after commit + unlink"
