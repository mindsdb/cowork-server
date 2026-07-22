"""Cross-tenant behaviour of the swept FileService + harness scope recovery.

Files are a root table (own org_id): direct org filtering on every query,
stamping on writes. The harness attachment listing recovers the ORIGINAL
scope from the session (never derives one from the conversation row).
"""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.harnesses.anton_harness.harness import _conversation_attachment_context
from cowork.models.conversation import Conversation
from cowork.models.file import File
from cowork.models.project import Project
from cowork.services.files import FileService, attachment_purpose

ORG_A = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ORG_B = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"


def _scope(org: str, user: str = "user-1") -> TenantScope:
    return TenantScope(org_mode=True, org_id=org, user_id=user)


@pytest.fixture()
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_FILES_DIR", str(tmp_path / "files"))
    get_app_settings.cache_clear()
    import cowork.models.message, cowork.models.message_event  # noqa: F401  mappers
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(eng)
    yield eng
    get_app_settings.cache_clear()


def _svc(engine, scope: TenantScope) -> FileService:
    return FileService(ScopedSession(Session(engine), scope))


def _mkfile(svc: FileService, purpose: str = "assistants") -> File:
    return svc.create_file_from_bytes(
        filename="report.csv", content_type="text/csv", data=b"a,b\n1,2\n", purpose=purpose
    )


def test_upload_stamps_org_and_creator(engine):
    row = _mkfile(_svc(engine, _scope(ORG_A, "alice")))
    assert row.org_id == ORG_A
    assert row.created_by == "alice"


def test_other_org_cannot_list_get_or_delete(engine):
    a = _svc(engine, _scope(ORG_A))
    b = _svc(engine, _scope(ORG_B))
    row = _mkfile(a)

    assert b.list_files() == []
    with pytest.raises(ValueError, match="not found"):
        b.get_file(row.id)
    with pytest.raises(ValueError, match="not found"):
        b.get_file_content(row.id)
    assert b.delete_file(row.id) is False  # same answer as nonexistent


def test_cross_org_delete_touches_no_bytes(engine):
    a = _svc(engine, _scope(ORG_A))
    b = _svc(engine, _scope(ORG_B))
    row = _mkfile(a)
    path = Path(row.path)
    assert path.exists()

    assert b.delete_file(row.id) is False
    assert path.exists(), "cross-org delete must not touch the filesystem"


def test_purpose_operations_stay_in_org(engine):
    # Same purpose string in two orgs — relink/delete must not cross over.
    a = _svc(engine, _scope(ORG_A))
    b = _svc(engine, _scope(ORG_B))
    shared_purpose = attachment_purpose(str(uuid4()))
    _mkfile(a, purpose=shared_purpose)
    _mkfile(b, purpose=shared_purpose)

    assert a.relink_purpose(shared_purpose, "moved") == 1  # only A's row
    assert len(b.list_file_rows(shared_purpose)) == 1      # B's untouched

    dirs = b.delete_by_purpose(shared_purpose)
    assert len(dirs) == 1  # only B's row staged


def test_local_scope_sees_everything(engine):
    _mkfile(_svc(engine, _scope(ORG_A)))
    local = _svc(engine, LOCAL_SCOPE)
    assert len(local.list_files()) == 1


def test_upload_fail_closed_writes_no_bytes(engine, tmp_path):
    from cowork.db.scoped import MissingTenantScopeError
    files_root = Path(get_app_settings().file.root_dir)
    before = set(files_root.iterdir()) if files_root.exists() else set()
    # org mode without an org in scope (audit gap) must fail BEFORE disk I/O
    svc = _svc(engine, TenantScope(org_mode=True, org_id=None))
    with pytest.raises(MissingTenantScopeError):
        _mkfile(svc)
    after = set(files_root.iterdir()) if files_root.exists() else set()
    assert before == after, "no orphaned bytes on scope failure"


def test_compat_upload_scope_failure_is_401_not_500(monkeypatch):
    # Org mode, audit, no identity: the upload's scope failure must surface
    # as the app-level 401, not be swallowed into the handler's generic 500.
    from fastapi.testclient import TestClient

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "audit")
    get_app_settings.cache_clear()
    try:
        from cowork.server import create_app
        client = TestClient(create_app())
        res = client.post(
            "/api/v1/attachments/general/some-session/upload",
            files={"files": ("a.txt", b"hi", "text/plain")},
        )
        assert res.status_code == 401
        assert res.json() == {"detail": "Unauthorized"}
    finally:
        get_app_settings.cache_clear()


def test_index_artifact_rejects_foreign_roots(engine):
    from cowork.services.task_objects import TaskObjectService
    from cowork.models.task_object import TaskObject
    # org B's roots...
    raw_b = Session(engine)
    b = ScopedSession(raw_b, _scope(ORG_B))
    project_b = Project(name=f"pb-{uuid4().hex[:6]}", path="/tmp/pb")
    b.add(project_b)
    b.commit()
    conv_b = Conversation(topic="b", project_id=project_b.id)
    b.add(conv_b)
    b.commit()
    b.refresh(conv_b)
    # ...indexed under org A's scope: anchoring must refuse, no row created
    a = ScopedSession(Session(engine), _scope(ORG_A))
    with pytest.raises(ValueError, match="not found"):
        TaskObjectService(a).index_artifact(conv_b.id, project_b.id, "stolen-slug")
    raw = Session(engine)
    assert raw.exec(select(TaskObject).where(TaskObject.ref == "stolen-slug")).first() is None


def test_index_artifact_works_for_own_roots(engine):
    from cowork.services.task_objects import TaskObjectService
    from cowork.models.task_object import TaskObject
    raw = Session(engine)
    a = ScopedSession(raw, _scope(ORG_A))
    project = Project(name=f"pa-{uuid4().hex[:6]}", path="/tmp/pa")
    a.add(project)
    a.commit()
    conv = Conversation(topic="a", project_id=project.id)
    a.add(conv)
    a.commit()
    a.refresh(conv)
    TaskObjectService(a).index_artifact(conv.id, project.id, "own-slug")
    assert raw.exec(select(TaskObject).where(TaskObject.ref == "own-slug")).first() is not None


# ── harness attachment listing: scope recovery, never derivation ────────────

def _conversation_with_file(engine, scope: TenantScope):
    """A conversation + attached file created under `scope`, returned attached
    to a scope-wrapped session (like the handler paths produce)."""
    raw = Session(engine)
    scoped = ScopedSession(raw, scope)
    project = Project(name=f"p-{uuid4().hex[:6]}", path="/tmp/x")
    scoped.add(project)
    scoped.commit()
    conv = Conversation(topic="t", project_id=project.id)
    scoped.add(conv)
    scoped.commit()
    scoped.refresh(conv)
    FileService(scoped).create_file_from_bytes(
        filename="doc.txt", content_type="text/plain", data=b"hi",
        purpose=attachment_purpose(str(conv.id)),
    )
    return raw, conv


def test_harness_lists_attachments_with_recovered_scope(engine, monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    _raw, conv = _conversation_with_file(engine, _scope(ORG_A))
    ctx = _conversation_attachment_context(conv)
    assert "doc.txt" in ctx


def test_harness_fails_closed_on_scope_mismatch(engine, monkeypatch, caplog):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    # a conversation genuinely owned by org B...
    _rawb, conv_b = _conversation_with_file(engine, _scope(ORG_B))
    # ...reached through a session whose recorded scope is org A (wrong routing)
    raw = Session(engine)
    ScopedSession(raw, _scope(ORG_A))
    stray = raw.get(Conversation, conv_b.id)
    with caplog.at_level("WARNING"):
        ctx = _conversation_attachment_context(stray)
    assert ctx == ""
    assert "does not match scope org" in caplog.text


def test_harness_fails_closed_without_scope_in_org_mode(engine, monkeypatch, caplog):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    # conversation loaded through a RAW session — no scope ever recorded
    raw = Session(engine)
    project = Project(name="raw-proj", path="/tmp/x", org_id=ORG_A)
    raw.add(project)
    raw.commit()
    conv = Conversation(topic="t", project_id=project.id, org_id=ORG_A)
    raw.add(conv)
    raw.commit()
    raw.refresh(conv)
    with caplog.at_level("WARNING"):
        ctx = _conversation_attachment_context(conv)
    assert ctx == ""
    assert "no tenant scope" in caplog.text


def test_harness_works_in_local_mode(engine, monkeypatch):
    monkeypatch.delenv("COWORK_TENANCY_MODE", raising=False)
    get_app_settings.cache_clear()
    _raw, conv = _conversation_with_file(engine, LOCAL_SCOPE)
    assert "doc.txt" in _conversation_attachment_context(conv)
