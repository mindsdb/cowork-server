"""Attachment ↔ conversation linkage (ENG-264).

The composer uploads attachments against a client-allocated conversation
id before the first stream. The responses handler must either adopt that
id (valid UUID) or re-link the uploads to the conversation it creates
(legacy non-UUID ids) — otherwise the Task Uploads rail, which queries by
the live conversation id, comes back empty.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.file import File
from cowork.services.conversations import ConversationService
from cowork.services.files import FileService, attachment_purpose


@pytest.fixture()
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


def _add_file(session: Session, purpose: str) -> File:
    file = File(
        filename="report.csv",
        content_type="text/csv",
        size=12,
        purpose=purpose,
        path="",
    )
    session.add(file)
    session.commit()
    session.refresh(file)
    return file


def test_create_conversation_adopts_client_allocated_id(session):
    allocated = uuid4()
    svc = ConversationService(session)
    conversation = svc.create_conversation(topic="t", conversation_id=allocated)
    assert conversation.id == allocated
    # And it is fetchable under that id — the handler's get-or-adopt path.
    assert svc.get_conversation(allocated).id == allocated


def test_create_conversation_without_id_still_generates_one(session):
    conversation = ConversationService(session).create_conversation(topic="t")
    assert isinstance(conversation.id, UUID)


def test_relink_purpose_moves_attachments(session):
    legacy_id = "20260612_134542_a1b2c3"
    new_id = str(uuid4())
    old = attachment_purpose("general", legacy_id)
    new = attachment_purpose("general", new_id)
    _add_file(session, old)
    _add_file(session, old)

    svc = FileService(session)
    assert svc.relink_purpose(old, new) == 2
    assert [f.filename for f in svc.list_files(purpose=new)] == ["report.csv", "report.csv"]
    assert svc.list_files(purpose=old) == []


def test_relink_purpose_noop_when_nothing_matches(session):
    svc = FileService(session)
    assert svc.relink_purpose(
        attachment_purpose("general", "nope"),
        attachment_purpose("general", str(uuid4())),
    ) == 0


def test_list_attachments_returns_legacy_row_shape(session):
    """The Task Uploads rail renders item.name / item.mime / item.size and
    parses ISO timestamps — the OpenAI FileResponse shape (filename /
    bytes / epoch-seconds) made rows show raw ids and a 1970s age."""
    from cowork.api.v1.endpoints.compat.stubs import list_attachments

    sid = str(uuid4())
    _add_file(session, attachment_purpose("general", sid))
    rows = list_attachments("general", sid, session, ids=None)
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "report.csv"
    assert row["mime"] == "text/csv"
    assert row["size"] == 12
    # ISO string a JS `new Date(...)` parses correctly (not epoch seconds).
    assert isinstance(row["created_at"], str) and "T" in row["created_at"]

    # The client's ?ids= filter narrows the listing.
    assert list_attachments("general", sid, session, ids=["nonexistent"]) == []
    assert len(list_attachments("general", sid, session, ids=[row["id"]])) == 1


def test_attachment_raw_serves_inline(session, tmp_path):
    """Row click opens the raw URL in a browser tab expecting the file to
    render — attachment disposition silently downloads instead."""
    from cowork.api.v1.endpoints.compat.stubs import attachment_raw

    payload = tmp_path / "photo.png"
    payload.write_bytes(b"\x89PNG fake")
    file = File(
        filename="photo.png",
        content_type="image/png",
        size=9,
        purpose=attachment_purpose("general", str(uuid4())),
        path=str(payload),
    )
    session.add(file)
    session.commit()
    session.refresh(file)

    response = attachment_raw("general", "ignored", file.id, session)
    assert response.headers["content-disposition"].startswith("inline")
    assert "photo.png" in response.headers["content-disposition"]


def test_stubs_purpose_matches_canonical_helper():
    """The upload endpoint and the rail's list endpoint both tag through
    the same helper — pin the format so they can't drift apart."""
    from cowork.api.v1.endpoints.compat.stubs import _attachment_purpose

    assert _attachment_purpose("proj", "abc") == attachment_purpose("proj", "abc")
    assert attachment_purpose("proj", "abc") == "attachment:proj:abc"
