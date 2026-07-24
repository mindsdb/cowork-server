"""Cross-tenant behaviour of the swept ProjectService.

Two orgs against one database: everything org A creates must be invisible
and untouchable for org B — and indistinguishable from nonexistent (404-shaped
None/ValueError, never a "forbidden"). Also covers the GENERAL project
bootstrap contract: fixed row only, atomic claim, no created_by attribution.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from sqlmodel import Session, select

from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.models.project import Project
from cowork.services.projects import GENERAL_PROJECT, GENERAL_PROJECT_ID, ProjectService

ORG_A = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ORG_B = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"


def _scope(org: str, user: str = "user-1") -> TenantScope:
    return TenantScope(org_mode=True, org_id=org, user_id=user)


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """Isolated engine + projects root, seeded with the GENERAL row (NULL org)."""
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    from cowork.common.settings.app_settings import get_app_settings
    get_app_settings.cache_clear()

    from sqlalchemy.pool import StaticPool
    from sqlmodel import SQLModel, create_engine

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as seed:
        seed.add(Project(id=GENERAL_PROJECT_ID, name=GENERAL_PROJECT, path=str(tmp_path / "projects" / "general"), is_active=True))
        seed.commit()
    yield engine
    get_app_settings.cache_clear()


def _svc(engine, scope: TenantScope) -> ProjectService:
    # One session per scope, like production (one request = one session).
    return ProjectService(ScopedSession(Session(engine), scope))


def _raw(engine) -> Session:
    return Session(engine)


def test_creation_stamps_org_and_creator(db):
    svc = _svc(db, _scope(ORG_A, "alice"))
    project = svc.create_project("reports")
    assert project.org_id == ORG_A
    assert project.created_by == "alice"


def test_other_org_cannot_see_list_get_rename_or_delete(db):
    a = _svc(db, _scope(ORG_A))
    b = _svc(db, _scope(ORG_B))
    project = a.create_project("secret-plans")

    assert "secret-plans" not in {p.name for p in b.list_projects()}
    with pytest.raises(ValueError, match="not found"):
        b.get_project(project.id)
    with pytest.raises(ValueError, match="not found"):
        b.get_project_by_name("secret-plans")
    with pytest.raises(ValueError, match="not found"):
        b.update_project(project.id, name="stolen")
    assert b.delete_project(project.id) is False  # same answer as nonexistent

    # and org A still has it, untouched
    assert a.get_project(project.id).name == "secret-plans"


def test_project_names_are_per_org(db):
    a = _svc(db, _scope(ORG_A))
    b = _svc(db, _scope(ORG_B))
    assert a.create_project("reports").name == "reports"
    # Name uniqueness is org-scoped: org B is blind to A's "reports", so no
    # -2 suffix. (Directory separation comes from per-org runtimes — one org
    # per deployment — so the shared-root mkdir collision can't happen in
    # production; asserted on the query, not create, for that reason.)
    assert b._unique_name("reports") == "reports"


@pytest.mark.parametrize("evil", ["../evil", "a/../../b", "/etc/passwd", "..", "a/b"])
def test_project_paths_cannot_escape_the_root(db, tmp_path, evil):
    svc = _svc(db, _scope(ORG_A))
    project = svc.create_project(evil)
    # The created dir must sit directly under the real projects root, not
    # wherever the (sanitized) name happened to point.
    projects_root = (tmp_path / "projects").resolve()
    assert Path(project.path).resolve().parent == projects_root
    assert "/" not in project.name and ".." != project.name


def test_project_path_guard_rejects_escapes_directly(db):
    svc = _svc(db, _scope(ORG_A))
    with pytest.raises(ValueError):
        svc._project_path("../evil")
    with pytest.raises(ValueError):
        svc._project_path(".")


def test_local_mode_sees_everything(db):
    a = _svc(db, _scope(ORG_A))
    a.create_project("cloud-thing")
    local = _svc(db, LOCAL_SCOPE)
    assert "cloud-thing" in {p.name for p in local.list_projects()}


# ── GENERAL project bootstrap ───────────────────────────────────────────────

def test_general_claim_is_idempotent_within_an_org(db):
    a = _svc(db, _scope(ORG_A))
    first = a.ensure_general_for_scope()
    second = a.ensure_general_for_scope()
    assert first is not None and second is not None
    assert first.id == second.id == GENERAL_PROJECT_ID
    assert first.org_id == ORG_A
    assert first.created_by is None  # system-created, never attributed


def test_general_claimed_by_a_is_gone_for_b(db):
    a = _svc(db, _scope(ORG_A))
    b = _svc(db, _scope(ORG_B))
    assert a.ensure_general_for_scope() is not None

    assert b.ensure_general_for_scope() is None  # no restamp, no access
    assert GENERAL_PROJECT not in {p.name for p in b.list_projects()}
    # A's claim untouched
    row = _raw(db).exec(select(Project).where(Project.id == GENERAL_PROJECT_ID)).one()
    assert row.org_id == ORG_A


def test_general_claim_only_touches_the_fixed_row(db):
    # A legacy NULL-org project must NOT be adopted by the bootstrap.
    raw = _raw(db)
    raw.add(Project(name="legacy", path="/tmp/legacy", is_active=False))
    raw.commit()
    a = _svc(db, _scope(ORG_A))
    a.ensure_general_for_scope()

    legacy = _raw(db).exec(select(Project).where(Project.name == "legacy")).one()
    assert legacy.org_id is None


def test_general_in_local_mode_needs_no_claim(db):
    local = _svc(db, LOCAL_SCOPE)
    general = local.ensure_general_for_scope()
    assert general is not None
    row = _raw(db).exec(select(Project).where(Project.id == GENERAL_PROJECT_ID)).one()
    assert row.org_id is None  # local mode never stamps
