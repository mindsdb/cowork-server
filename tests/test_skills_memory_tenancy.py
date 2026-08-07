"""Org keying of the filesystem stores: skills and global memory.

Both are `<root>/<org_id>/` in org mode (skills and rules/lessons are injected
into agent turns — a cross-org write would be prompt injection into another
org's agent). Local mode uses the shared root unchanged; org mode without an
org fails closed.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import (
    LOCAL_SCOPE,
    MissingTenantScopeError,
    ScopedSession,
    TenantScope,
    scoped_storage_root,
)
from cowork.services.skills import SkillService

ORG_A = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ORG_B = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"


def _org(org: str | None) -> TenantScope:
    return TenantScope(org_mode=True, org_id=org, user_id="u")


# scoped_storage_root

def test_storage_root_local_is_base():
    assert scoped_storage_root(Path("/x"), None) == Path("/x")
    assert scoped_storage_root(Path("/x"), LOCAL_SCOPE) == Path("/x")


def test_storage_root_org_keyed_and_fail_closed():
    assert scoped_storage_root(Path("/x"), _org(ORG_A)) == Path("/x") / ORG_A
    with pytest.raises(MissingTenantScopeError):
        scoped_storage_root(Path("/x"), _org(None))


# skills

@pytest.fixture()
def skills_root(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "skills"))
    get_app_settings.cache_clear()
    yield tmp_path / "skills"
    get_app_settings.cache_clear()


def test_skills_isolated_per_org(skills_root):
    a = SkillService(_org(ORG_A))
    b = SkillService(_org(ORG_B))
    a.create_skill(label="Report", name="report", instructions="write reports")

    assert [s.name for s in a.list_skills()] == ["report"]
    assert b.list_skills() == []                      # invisible cross-org
    with pytest.raises(ValueError):
        b.get_skill("report")                         # unreadable cross-org
    assert b.delete_skill("report") is False          # undeletable cross-org
    assert [s.name for s in a.list_skills()] == ["report"]  # untouched

    # same name in another org is a fresh skill, not a collision/overwrite
    b.create_skill(label="Report", name="report", instructions="B's own")
    assert a.get_skill("report").instructions != "B's own"


def test_skills_local_mode_uses_shared_root(skills_root):
    local = SkillService()
    local.create_skill(label="X", name="x", instructions="i")
    assert (skills_root / "x").is_dir()               # no org segment


def test_skills_org_mode_without_org_fails_closed(skills_root):
    with pytest.raises(MissingTenantScopeError):
        SkillService(_org(None))


# global memory

@pytest.fixture()
def memory_env(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_MEMORY_DIR", str(tmp_path / "memory"))
    get_app_settings.cache_clear()
    import cowork.models.project, cowork.models.conversation  # noqa: F401
    import cowork.models.message, cowork.models.message_event  # noqa: F401
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(eng)
    yield eng, tmp_path / "memory"
    get_app_settings.cache_clear()


def _memory_service(engine, scope: TenantScope):
    from cowork.services.memory import MemoryService
    return MemoryService(ScopedSession(Session(engine), scope))


@pytest.mark.asyncio
async def test_global_memory_isolated_per_org(memory_env):
    engine, root = memory_env
    a = _memory_service(engine, _org(ORG_A))
    b = _memory_service(engine, _org(ORG_B))

    await a.update_memory(scope="global", category="rules", content="A's rules")
    assert (await a.get_memory(scope="global", category="rules")).content.strip() == "A's rules"
    # org B sees empty, and its write doesn't touch A's file
    assert (await b.get_memory(scope="global", category="rules")).content.strip() == ""
    await b.update_memory(scope="global", category="rules", content="B's rules")
    assert (await a.get_memory(scope="global", category="rules")).content.strip() == "A's rules"
    # on disk: two distinct org-keyed files
    assert (root / ORG_A / "rules.md").is_file() and (root / ORG_B / "rules.md").is_file()


@pytest.mark.asyncio
async def test_global_memory_local_mode_unchanged(memory_env):
    engine, root = memory_env
    local = _memory_service(engine, LOCAL_SCOPE)
    await local.update_memory(scope="global", category="rules", content="local rules")
    assert (root / "rules.md").is_file()              # no org segment


# review fixes: link distribution is desktop-only; zip caps; harness memory keyed

def test_org_mode_creates_no_project_symlinks(skills_root, tmp_path, monkeypatch):
    # The UUID-slug escape: org A names a skill with org B's org id. In org
    # mode no symlink reconciliation may run at all — nothing outside the
    # org's own root may be created or referenced.
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    proj = tmp_path / "projects" / "victim-proj"
    proj.mkdir(parents=True)
    (tmp_path / "skills" / ORG_B).mkdir(parents=True)  # org B's skill root
    (tmp_path / "skills" / ORG_B / "secret").mkdir()

    SkillService(_org(ORG_A)).create_skill(label="X", name=ORG_B, instructions="i")
    assert not (proj / "skills").exists(), "org mode must not touch project dirs"


def test_local_mode_still_reconciles_links(skills_root, tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    proj = tmp_path / "projects" / "p1"
    proj.mkdir(parents=True)
    SkillService().create_skill(label="Y", name="y", instructions="i")
    assert (proj / "skills" / "y").exists(), "desktop link distribution unchanged"


def _zip_bytes(members: dict[str, str]) -> bytes:
    import io as _io
    import zipfile as _zf
    buf = _io.BytesIO()
    with _zf.ZipFile(buf, "w", _zf.ZIP_DEFLATED) as z:
        for name, content in members.items():
            z.writestr(name, content)
    return buf.getvalue()


def test_zip_import_rejects_oversized_archives(skills_root, monkeypatch):
    svc = SkillService(_org(ORG_A))
    monkeypatch.setattr(SkillService, "_ZIP_MAX_UNCOMPRESSED", 16)
    with pytest.raises(ValueError, match="expand beyond"):
        svc.import_skill(_zip_bytes({"SKILL.md": "x" * 100}), filename="bomb.zip")


def test_inprocess_harness_memory_root_is_org_keyed(memory_env):
    from cowork.common.settings.user_settings import use_settings_scope, current_settings_scope
    engine, root = memory_env
    with use_settings_scope(_org(ORG_A)):
        keyed = scoped_storage_root(root, current_settings_scope())
    assert keyed == root / ORG_A
    assert scoped_storage_root(root, current_settings_scope()) == root  # reset outside
