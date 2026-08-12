"""Cross-tenant behaviour of the swept ProjectService.

Two orgs against one database: everything org A creates must be invisible
and untouchable for org B — and indistinguishable from nonexistent (404-shaped
None/ValueError, never a "forbidden"). Also covers the default-project
contract: one per org, no duplicates, no created_by attribution.
"""
from __future__ import annotations

import shutil
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
    # Name uniqueness is org-scoped: B is blind to A's "reports", so no -2
    # suffix. Directories are separated by the org-keyed root now, not by
    # "one org per deployment" (that died with the shared-instance decision).
    assert b._unique_name("reports") == "reports"


@pytest.mark.parametrize("evil", ["../evil", "a/../../b", "/etc/passwd", "..", "a/b"])
def test_project_paths_cannot_escape_the_root(db, tmp_path, evil):
    svc = _svc(db, _scope(ORG_A))
    project = svc.create_project(evil)
    # The created dir must sit directly under this ORG's projects root (the root
    # is org-keyed, like the skills/memory stores), not wherever the (sanitized)
    # name happened to point.
    projects_root = (tmp_path / "projects" / str(ORG_A)).resolve()
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

def test_general_is_idempotent_within_an_org(db):
    a = _svc(db, _scope(ORG_A))
    first = a.ensure_general_for_scope()
    second = a.ensure_general_for_scope()
    assert first is not None and second is not None
    assert first.id == second.id
    assert first.org_id == ORG_A
    assert first.created_by is None  # system-created, never attributed


def test_every_org_gets_its_own_general(db):
    """One seeded id can't serve N tenants: the first org claimed it and every
    later org got None → 404."""
    a = _svc(db, _scope(ORG_A))
    b = _svc(db, _scope(ORG_B))

    ga = a.ensure_general_for_scope()
    gb = b.ensure_general_for_scope()

    assert ga is not None and gb is not None
    assert ga.id != gb.id  # separate rows, not a contested one
    assert (ga.org_id, gb.org_id) == (ORG_A, ORG_B)
    assert Path(ga.path) != Path(gb.path)  # and separate directories
    # Each org sees exactly its own.
    assert GENERAL_PROJECT in {p.name for p in a.list_projects()}
    assert GENERAL_PROJECT in {p.name for p in b.list_projects()}


def test_general_does_not_adopt_a_legacy_null_org_row(db):
    # A legacy NULL-org project must NOT be swept into an org.
    raw = _raw(db)
    raw.add(Project(name="legacy", path="/tmp/legacy", is_active=False))
    raw.commit()
    a = _svc(db, _scope(ORG_A))
    a.ensure_general_for_scope()

    legacy = _raw(db).exec(select(Project).where(Project.name == "legacy")).one()
    assert legacy.org_id is None


def test_general_never_duplicates_across_concurrent_sessions(db):
    """No unique constraint + 2 replicas + called on every request: two sessions
    racing first load must converge on ONE row."""
    a1 = _svc(db, _scope(ORG_A, "alice"))
    a2 = _svc(db, _scope(ORG_A, "bob"))  # a second replica, same org

    # Both past the pre-check before either writes. Calling
    # ensure_general_for_scope twice would short-circuit and miss the race.
    path = a1._project_path(GENERAL_PROJECT)
    a1._insert_general_if_absent(path)
    a2._insert_general_if_absent(path)

    rows = _raw(db).exec(
        select(Project).where(Project.org_id == ORG_A, Project.name == GENERAL_PROJECT)
    ).all()
    assert len(rows) == 1
    assert a1.ensure_general_for_scope().id == rows[0].id


def test_general_recreates_a_missing_directory(db):
    """The live 404: row resolved, directory did not exist."""
    a = _svc(db, _scope(ORG_A))
    general = a.ensure_general_for_scope()
    assert general is not None
    shutil.rmtree(general.path)
    assert not Path(general.path).is_dir()

    again = a.ensure_general_for_scope()

    assert again is not None and Path(again.path).is_dir()


def test_general_in_local_mode_needs_no_claim(db):
    local = _svc(db, LOCAL_SCOPE)
    general = local.ensure_general_for_scope()
    assert general is not None
    row = _raw(db).exec(select(Project).where(Project.id == GENERAL_PROJECT_ID)).one()
    assert row.org_id is None  # local mode never stamps


def test_general_repoints_a_legacy_unkeyed_path_with_no_content(db, tmp_path):
    """Rows predating the org-keyed root point outside it. With no content there,
    re-point them — otherwise the org stays 404'd."""
    a = _svc(db, _scope(ORG_A))
    legacy = tmp_path / "projects" / GENERAL_PROJECT  # un-keyed, never created
    raw = _raw(db)
    raw.add(Project(name=GENERAL_PROJECT, path=str(legacy), is_active=False, org_id=ORG_A))
    raw.commit()

    general = a.ensure_general_for_scope()

    assert general is not None
    assert Path(general.path).parent == (tmp_path / "projects" / str(ORG_A))
    assert Path(general.path).is_dir()


def test_general_keeps_a_legacy_path_that_still_has_content(db, tmp_path):
    """The mirror case: a dir with real content must not be abandoned."""
    a = _svc(db, _scope(ORG_A))
    legacy = tmp_path / "projects" / GENERAL_PROJECT
    legacy.mkdir(parents=True)
    (legacy / "notes.md").write_text("real work")
    raw = _raw(db)
    raw.add(Project(name=GENERAL_PROJECT, path=str(legacy), is_active=False, org_id=ORG_A))
    raw.commit()

    general = a.ensure_general_for_scope()

    assert general is not None
    assert Path(general.path) == legacy
    assert (Path(general.path) / "notes.md").read_text() == "real work"


def test_same_project_name_in_two_orgs_gets_separate_directories(db):
    """Rows were always keyed by org_id; the filesystem was not — two orgs using
    `reports` shared one directory."""
    a = _svc(db, _scope(ORG_A))
    b = _svc(db, _scope(ORG_B))

    pa = a.create_project("reports")
    pb = b.create_project("reports")

    assert pa.name == pb.name == "reports"  # no -2 suffix: names are per org
    assert Path(pa.path) != Path(pb.path)
    (Path(pa.path) / "a.md").write_text("org A")
    (Path(pb.path) / "b.md").write_text("org B")
    assert not (Path(pa.path) / "b.md").exists()  # no shared directory


def test_general_id_round_trips_through_get_project(db):
    """Review: sa.literal(str(uuid4())) binds as String and skips the Uuid bind
    processor, so the stored id doesn't match how every other lookup binds.

    Must use a FRESH session: the writing session's identity map would serve the
    row from memory and never round-trip through the column type.
    """
    general = _svc(db, _scope(ORG_A)).ensure_general_for_scope()
    assert general is not None

    fresh = _svc(db, _scope(ORG_A))
    assert fresh.get_project(general.id).id == general.id


def test_unique_index_rejects_a_second_default_for_one_org(db):
    """What arbitrates on Postgres, where two replicas can both pass NOT EXISTS."""
    import sqlalchemy as sa
    raw = _raw(db)
    raw.exec(sa.text(
        "CREATE UNIQUE INDEX uq_projects_default_per_org ON projects "
        "(coalesce(org_id, '')) WHERE name = 'general'"
    ))
    _svc(db, _scope(ORG_A)).ensure_general_for_scope()

    with pytest.raises(sa.exc.IntegrityError):
        dup = _raw(db)
        dup.add(Project(name=GENERAL_PROJECT, path="/tmp/dup", is_active=False, org_id=ORG_A))
        dup.commit()


def test_a_losing_insert_adopts_the_winners_row(db, monkeypatch):
    """The loser must return the winner's row, not assume its own insert landed."""
    import sqlalchemy as sa
    winner = _svc(db, _scope(ORG_A, "alice")).ensure_general_for_scope()
    assert winner is not None

    loser = _svc(db, _scope(ORG_A, "bob"))
    monkeypatch.setattr(
        type(loser),
        "get_project_by_name_or_none",
        lambda self, name: None,  # pretend the pre-check saw nothing
        raising=True,
    )
    monkeypatch.setattr(
        type(loser),
        "_execute_general_insert",
        lambda self, raw, path, org_id: (_ for _ in ()).throw(
            sa.exc.IntegrityError("insert", {}, Exception("duplicate"))
        ),
        raising=True,
    )

    loser._insert_general_if_absent(loser._project_path(GENERAL_PROJECT))  # must not raise

    rows = _raw(db).exec(
        select(Project).where(Project.org_id == ORG_A, Project.name == GENERAL_PROJECT)
    ).all()
    assert len(rows) == 1 and rows[0].id == winner.id


def test_fresh_org_has_an_active_project(db):
    """Review: provisioned with is_active=False, so a fresh org had zero active
    projects and get_active_project raised on the first request."""
    a = _svc(db, _scope(ORG_A))
    a.ensure_general_for_scope()
    assert a.get_active_project().name == GENERAL_PROJECT


def test_a_member_cannot_take_the_default_projects_name(db):
    """Review: a member could POST /projects named `general` before the system row
    exists — it became user-attributed and undeletable, and the name-based lookup
    then blocked the real default forever."""
    a = _svc(db, _scope(ORG_A, "alice"))
    hijack = a.create_project(GENERAL_PROJECT)

    assert hijack.name != GENERAL_PROJECT  # the name is reserved
    general = a.ensure_general_for_scope()
    assert general is not None
    assert general.created_by is None  # the real system row, not the member's
    assert general.id != hijack.id


def test_repointing_does_not_attribute_the_system_project(db, tmp_path):
    """Review: ScopedSession.add stamps created_by when it's None, so healing a
    stale path attributed the org's default to whoever triggered the heal."""
    legacy = tmp_path / "projects" / GENERAL_PROJECT
    raw = _raw(db)
    raw.add(Project(name=GENERAL_PROJECT, path=str(legacy), is_active=False, org_id=ORG_A))
    raw.commit()

    general = _svc(db, _scope(ORG_A, "alice")).ensure_general_for_scope()

    assert general is not None
    assert general.created_by is None  # still system-created
