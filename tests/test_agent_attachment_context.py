"""Regression: the agent must be told where conversation-attached files live.

Uploads land under `.cowork/files/<uuid>/<name>` — OUTSIDE the project dir.
The harness injects their absolute paths into the agent's context so it can
read them on any turn instead of scanning only the project root and wrongly
reporting "no files uploaded" (the Cyberdeck bug).
"""
from __future__ import annotations

from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.harnesses.anton_harness.harness import _conversation_attachment_context
from cowork.models.conversation import Conversation
from cowork.models.file import File
from cowork.models.project import Project
from cowork.services.files import attachment_purpose


def _session() -> Session:
    return Session(get_engine(get_app_settings().database.uri))


def _make_conversation(session: Session, project_name: str) -> Conversation:
    project = Project(name=project_name, path=f"/tmp/{project_name}")
    session.add(project)
    session.commit()
    session.refresh(project)
    conv = Conversation(topic="t", project_id=project.id)
    session.add(conv)
    session.commit()
    session.refresh(conv)
    return conv


def test_context_lists_attached_file_paths():
    with _session() as session:
        conv = _make_conversation(session, "Cyberdeck-ctx-1")
        purpose = attachment_purpose("Cyberdeck-ctx-1", str(conv.id))
        files = {
            "README.md": "/home/anton/.cowork/files/51d15fc7/README.md",
            "Kyle_Logo.png": "/home/anton/.cowork/files/28a83013/Kyle_Logo.png",
        }
        for name, path in files.items():
            session.add(File(filename=name, content_type="text/plain", size=1, purpose=purpose, path=path))
        session.commit()
        session.refresh(conv)

        ctx = _conversation_attachment_context(conv)

        # Every attached file's absolute path AND name must be surfaced.
        for name, path in files.items():
            assert path in ctx
            assert name in ctx
        # And it must tell the agent these live outside the project dir.
        assert "OUTSIDE the project" in ctx


def test_context_empty_when_no_attachments():
    with _session() as session:
        conv = _make_conversation(session, "Cyberdeck-ctx-2")
        assert _conversation_attachment_context(conv) == ""


def test_context_safe_when_conversation_detached():
    # A conversation with no bound session must not raise — just yields "".
    with _session() as session:
        conv = _make_conversation(session, "Cyberdeck-ctx-3")
    session.close()  # detaches conv from its session
    assert _conversation_attachment_context(conv) == ""
