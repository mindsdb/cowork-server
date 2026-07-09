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


def test_context_lists_attached_file_paths(tmp_path):
    with _session() as session:
        conv = _make_conversation(session, "Cyberdeck-ctx-1")
        purpose = attachment_purpose(str(conv.id))
        # Real files on disk — the helper only lists paths that exist.
        files = {}
        for name in ("README.md", "Kyle_Logo.png"):
            p = tmp_path / name
            p.write_text("x")
            files[name] = str(p)
            session.add(File(filename=name, content_type="text/plain", size=1, purpose=purpose, path=str(p)))
        session.commit()
        session.refresh(conv)

        ctx = _conversation_attachment_context(conv)

        # Every attached file's absolute path AND name must be surfaced.
        for name, path in files.items():
            assert path in ctx
            assert name in ctx
        # And it must tell the agent these live outside the project dir.
        assert "OUTSIDE the project" in ctx


def test_context_skips_files_missing_from_disk(tmp_path):
    with _session() as session:
        conv = _make_conversation(session, "ctx-miss")
        purpose = attachment_purpose(str(conv.id))
        present = tmp_path / "here.md"
        present.write_text("x")
        session.add(File(filename="here.md", content_type="text/plain", size=1, purpose=purpose, path=str(present)))
        # Row whose file was deleted from disk — must not be listed.
        session.add(File(filename="gone.md", content_type="text/plain", size=1, purpose=purpose, path=str(tmp_path / "gone.md")))
        session.commit()
        session.refresh(conv)

        ctx = _conversation_attachment_context(conv)
        assert "here.md" in ctx
        assert "gone.md" not in ctx


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


def test_context_logs_and_returns_empty_on_error(monkeypatch):
    # An unexpected failure (e.g. the DB query blows up) must NOT crash the
    # turn — but it must also not fail silently, because a swallowed error
    # looks identical to "no attachments" and reintroduces the Cyberdeck bug.
    # The helper must return "" AND log a warning so the failure is visible.
    from unittest.mock import MagicMock

    import cowork.harnesses.anton_harness.harness as harness_mod
    from cowork.services import files as files_module

    def _boom(self, *args, **kwargs):
        raise RuntimeError("simulated DB failure")

    monkeypatch.setattr(files_module.FileService, "list_file_rows", _boom)
    spy_logger = MagicMock()
    monkeypatch.setattr(harness_mod, "logger", spy_logger)

    with _session() as session:
        conv = _make_conversation(session, "Cyberdeck-ctx-err")
        ctx = _conversation_attachment_context(conv)

    assert ctx == ""  # degrades gracefully
    assert spy_logger.warning.called, "a failure must be logged, not swallowed silently"


def test_context_skips_one_corrupt_row_keeps_others(tmp_path, monkeypatch):
    # A single unresolvable row (here, a path the OS rejects on stat) must not
    # abort the whole list — every other attachment must still be surfaced.
    import cowork.harnesses.anton_harness.harness as harness_mod

    good = tmp_path / "good.md"
    good.write_text("x")
    bad_path = str(tmp_path / "bad.md")

    real_exists = harness_mod.Path.exists

    def flaky_exists(self):
        if str(self) == bad_path:
            raise OSError("simulated unreadable path")
        return real_exists(self)

    monkeypatch.setattr(harness_mod.Path, "exists", flaky_exists)

    with _session() as session:
        conv = _make_conversation(session, "ctx-corrupt")
        purpose = attachment_purpose(str(conv.id))
        session.add(File(filename="good.md", content_type="text/plain", size=1, purpose=purpose, path=str(good)))
        session.add(File(filename="bad.md", content_type="text/plain", size=1, purpose=purpose, path=bad_path))
        session.commit()
        session.refresh(conv)

        ctx = _conversation_attachment_context(conv)

    assert "good.md" in ctx   # surviving file still listed
    assert "bad.md" not in ctx  # corrupt row skipped, not fatal
