"""Cross-tenant behaviour of the swept FileService + harness scope recovery.

Files are a root table (own org_id): direct org filtering on every query,
stamping on writes. The harness attachment listing recovers the ORIGINAL
scope from the session (never derives one from the conversation row).
"""
from __future__ import annotations

import io
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi import UploadFile
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
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path))
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
    # Watch the whole isolated tree — a leak could land in either layout.
    before = set(tmp_path.rglob("*"))
    # org mode without an org in scope (audit gap) must fail BEFORE disk I/O
    svc = _svc(engine, TenantScope(org_mode=True, org_id=None))
    with pytest.raises(MissingTenantScopeError):
        _mkfile(svc)
    assert set(tmp_path.rglob("*")) == before, "no orphaned bytes on scope failure"


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
    # Fail closed means: leak NOTHING about the other org's files. Assert that
    # property directly rather than `ctx == ""` — since ENG-1357 the helper
    # always returns the generic attachment affordance (org-agnostic constant
    # text, no filenames or paths), so an empty-string assertion would fail
    # for a reason that has nothing to do with tenancy.
    assert "doc.txt" not in ctx
    assert str(ORG_B) not in ctx
    # Nor may it claim nothing is attached — a file IS attached, we just
    # refused to look. See test_agent_attachment_context.py.
    assert "No files are currently attached" not in ctx
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
    # Same as above: the guarantee is "no attachment listing", not "empty
    # string". Nothing was looked up, so it must not assert emptiness either.
    assert "doc.txt" not in ctx
    assert "No files are currently attached" not in ctx
    assert "no tenant scope" in caplog.text


def test_harness_works_in_local_mode(engine, monkeypatch):
    monkeypatch.delenv("COWORK_TENANCY_MODE", raising=False)
    get_app_settings.cache_clear()
    _raw, conv = _conversation_with_file(engine, LOCAL_SCOPE)
    assert "doc.txt" in _conversation_attachment_context(conv)


# ── upload safety: untrusted filename + size cap (not tenancy, but this is the
# FileService test home and reuses `engine`/`_svc`) ──────────────────────────

def _upload(name: str, data: bytes = b"x") -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=name)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["../../../../etc/pwned", "/etc/cron.d/pwned", ".."])
async def test_upload_filename_cannot_escape_root(engine, tmp_path, name):
    svc = _svc(engine, _scope(ORG_A))
    res = await svc.create_file(_upload(name), purpose="assistants")
    stored = Path(svc._get_file_model(UUID(res.id)).path).resolve()
    # org-first layout: bytes contained under <shared>/<org>/files
    assert (tmp_path / ORG_A / "files").resolve() in stored.parents
    assert stored.name in ("pwned", "upload")                # basename or fallback
    assert not (tmp_path / "etc").exists()


def test_delete_never_rmtrees_an_escaped_legacy_path(engine, tmp_path):
    # A legacy row whose stored path escaped the root must not let delete rmtree it.
    svc = _svc(engine, _scope(ORG_A))
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("x")
    f = File(filename="x", content_type="text/plain", size=1,
             purpose="assistants", path=str(victim / "x"))
    svc.session.add(f)
    svc.session.commit()
    svc.session.refresh(f)
    assert svc.delete_file(f.id) is True
    assert (victim / "keep.txt").exists()  # untouched


def test_delete_by_purpose_never_rmtrees_an_escaped_legacy_path(engine, tmp_path):
    # Same escape as delete_file, via the conversation/project cleanup path.
    from cowork.services.files import unlink_file_dirs
    svc = _svc(engine, _scope(ORG_A))
    victim = tmp_path / "victim2"
    victim.mkdir()
    (victim / "keep.txt").write_text("x")
    f = File(filename="x", content_type="text/plain", size=1,
             purpose="attachment:legacy", path=str(victim / "x"))
    svc.session.add(f)
    svc.session.commit()

    dirs = svc.delete_by_purpose("attachment:legacy")
    svc.session.commit()
    unlink_file_dirs(dirs)  # the caller unlinks after committing
    assert (victim / "keep.txt").exists()  # untouched


# ── org-first files layout ───────────────────────────────────────────────────

def test_files_land_in_separate_org_subtrees(engine, tmp_path):
    # Disk-level separation, not just row filtering.
    row_a = _mkfile(_svc(engine, _scope(ORG_A)))
    row_b = _mkfile(_svc(engine, _scope(ORG_B)))
    assert Path(row_a.path) == tmp_path / ORG_A / "files" / str(row_a.id) / "report.csv"
    assert Path(row_b.path) == tmp_path / ORG_B / "files" / str(row_b.id) / "report.csv"


def test_same_org_delete_removes_bytes(engine):
    # Write and delete must resolve the same root, or bytes silently orphan.
    svc = _svc(engine, _scope(ORG_A))
    row = _mkfile(svc)
    path = Path(row.path)
    assert path.exists()
    assert svc.delete_file(row.id) is True
    assert not path.parent.exists()


def test_local_mode_delete_removes_bytes(engine, tmp_path):
    svc = _svc(engine, LOCAL_SCOPE)
    row = _mkfile(svc)
    path = Path(row.path)
    assert (tmp_path / "files").resolve() in path.resolve().parents  # unkeyed base
    assert svc.delete_file(row.id) is True
    assert not path.parent.exists()


def test_delete_by_purpose_removes_bytes_under_current_root(engine):
    # Staged dirs must be the real on-disk dirs.
    from cowork.services.files import unlink_file_dirs
    svc = _svc(engine, _scope(ORG_A))
    purpose = attachment_purpose(str(uuid4()))
    rows = [_mkfile(svc, purpose=purpose) for _ in range(2)]
    paths = [Path(r.path) for r in rows]
    assert all(p.exists() for p in paths)

    dirs = svc.delete_by_purpose(purpose)
    svc.session.commit()
    unlink_file_dirs(dirs)
    assert all(not p.parent.exists() for p in paths)


def test_delete_unlinks_stored_dir_after_root_move(engine, tmp_path, monkeypatch):
    # If the root moves between write and delete, the stored dir must still go.
    svc = _svc(engine, _scope(ORG_A))
    row = _mkfile(svc)
    old_path = Path(row.path)
    assert old_path.exists()

    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "moved-root"))
    get_app_settings.cache_clear()
    try:
        assert svc.delete_file(row.id) is True
        assert not old_path.parent.exists(), "bytes at the old root must not orphan"
    finally:
        get_app_settings.cache_clear()


def test_delete_after_root_move_still_ignores_escaped_paths(engine, tmp_path, monkeypatch):
    # Stored dir only counts when its resolved parent is named <file.id>.
    svc = _svc(engine, _scope(ORG_A))
    victim = tmp_path / "victim-moved"
    victim.mkdir()
    (victim / "keep.txt").write_text("x")
    f = File(filename="x", content_type="text/plain", size=1,
             purpose="assistants", path=str(victim / "x"))
    svc.session.add(f)
    svc.session.commit()
    svc.session.refresh(f)

    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "moved-root2"))
    get_app_settings.cache_clear()
    try:
        assert svc.delete_file(f.id) is True
        assert (victim / "keep.txt").exists()  # untouched
    finally:
        get_app_settings.cache_clear()


# ── cloud attachment staging into the pod workspace ──────────────────────────

def test_stage_conversation_attachments_copies_into_workspace(engine, tmp_path):
    svc = _svc(engine, _scope(ORG_A))
    conv = str(uuid4())
    a = svc.create_file_from_bytes(filename="shot.png", content_type="image/png",
                                   data=b"img", purpose=attachment_purpose(conv))
    b = svc.create_file_from_bytes(filename="notes.txt", content_type="text/plain",
                                   data=b"hi", purpose=attachment_purpose(conv))
    proj = tmp_path / "proj"
    assert svc.stage_conversation_attachments(conv, proj) == 2
    base = proj / "conversations" / conv / "attachments"
    assert (base / str(a.id) / "shot.png").read_bytes() == b"img"
    assert (base / str(b.id) / "notes.txt").read_bytes() == b"hi"


def test_stage_is_idempotent_and_conversation_scoped(engine, tmp_path):
    svc = _svc(engine, _scope(ORG_A))
    conv, other = str(uuid4()), str(uuid4())
    svc.create_file_from_bytes(filename="a.txt", content_type="text/plain",
                               data=b"x", purpose=attachment_purpose(conv))
    svc.create_file_from_bytes(filename="b.txt", content_type="text/plain",
                               data=b"y", purpose=attachment_purpose(other))
    proj = tmp_path / "proj"
    assert svc.stage_conversation_attachments(conv, proj) == 1      # only this conv's
    assert svc.stage_conversation_attachments(conv, proj) == 1      # idempotent
    assert not (proj / "conversations" / other).exists()           # other conv untouched


def test_stage_project_instructions_copies_anton_md_into_workspace(tmp_path):
    from cowork.services.files import stage_project_instructions
    conv = str(uuid4())
    proj = tmp_path / "proj"
    (proj / ".anton").mkdir(parents=True)
    (proj / ".anton" / "anton.md").write_text("# Project rules\nBe concise.")

    assert stage_project_instructions(proj, conv) is True
    dest = proj / "conversations" / conv / ".anton" / "anton.md"
    assert dest.read_text() == "# Project rules\nBe concise."
    # idempotent: no error, still in place
    assert stage_project_instructions(proj, conv) is True
    assert dest.is_file()


def test_stage_project_instructions_noop_without_anton_md(tmp_path):
    from cowork.services.files import stage_project_instructions
    proj = tmp_path / "proj"
    proj.mkdir()
    assert stage_project_instructions(proj, str(uuid4())) is False


def test_stage_prunes_a_deleted_attachment(engine, tmp_path):
    svc = _svc(engine, _scope(ORG_A))
    conv = str(uuid4())
    a = svc.create_file_from_bytes(filename="keep.txt", content_type="text/plain",
                                   data=b"k", purpose=attachment_purpose(conv))
    b = svc.create_file_from_bytes(filename="gone.txt", content_type="text/plain",
                                   data=b"g", purpose=attachment_purpose(conv))
    proj = tmp_path / "proj"
    assert svc.stage_conversation_attachments(conv, proj) == 2
    base = proj / "conversations" / conv / "attachments"
    assert (base / str(b.id)).is_dir()

    svc.delete_file(b.id)                       # user removes one attachment
    assert svc.stage_conversation_attachments(conv, proj) == 1
    assert (base / str(a.id)).is_dir()          # kept one remains
    assert not (base / str(b.id)).exists()      # deleted one pruned → agent stops seeing it


def test_stage_rejects_non_uuid_conversation_segment(engine, tmp_path):
    from cowork.services.files import stage_project_instructions
    svc = _svc(engine, _scope(ORG_A))
    assert svc.stage_conversation_attachments("../evil", tmp_path / "proj") == 0
    assert stage_project_instructions(tmp_path / "proj", "../evil") is False


def test_remove_conversation_workspace_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")   # cloud-only cleanup
    get_app_settings.cache_clear()
    from cowork.services.files import remove_conversation_workspace_dir
    conv = str(uuid4())
    proj = tmp_path / "proj"
    ws = proj / "conversations" / conv
    (ws / "attachments" / "x").mkdir(parents=True)
    (ws / ".anton").mkdir(parents=True)
    remove_conversation_workspace_dir(proj, conv)
    assert not ws.exists()
    remove_conversation_workspace_dir(proj, conv)   # idempotent, no error
    remove_conversation_workspace_dir(None, conv)   # no project → no-op


def test_remove_conversation_workspace_dir_is_noop_on_desktop(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    from cowork.services.files import remove_conversation_workspace_dir
    conv = str(uuid4())
    proj = tmp_path / "proj"
    ws = proj / "conversations" / conv
    (ws / "notes").mkdir(parents=True)               # a user's own dir, coincidental name
    remove_conversation_workspace_dir(proj, conv)
    assert ws.exists(), "desktop conversation delete must not rmtree a project subdir"


def test_stage_instructions_restages_a_same_length_same_mtime_edit(tmp_path):
    """A typo-fix edit (same length, same mtime second) must still re-stage —
    the old size+mtime skip could serve stale instructions."""
    import os
    from cowork.services.files import stage_project_instructions
    conv = str(uuid4())
    proj = tmp_path / "proj"
    (proj / ".anton").mkdir(parents=True)
    src = proj / ".anton" / "anton.md"
    src.write_text("be terse")
    assert stage_project_instructions(proj, conv) is True
    dest = proj / "conversations" / conv / ".anton" / "anton.md"
    assert dest.read_text() == "be terse"

    # same-length edit, pinned to the same mtime as the staged copy
    fixed_mtime = dest.stat().st_mtime
    src.write_text("be funny")                       # same length (8), different content
    os.utime(src, (fixed_mtime, fixed_mtime))
    assert stage_project_instructions(proj, conv) is True
    assert dest.read_text() == "be funny"            # re-staged despite the tie
