"""The user-facing project label (ENG-1676).

`name` is the slug: the on-disk directory, the URL segment and the lookup key.
It is produced by an ASCII allowlist, so a Cyrillic/CJK/Arabic name sanitized
to nothing and the project was created as `untitled-project`, `-2`, `-3` — a
whole project list collapsing into an unnamed sequence.

`display_name` holds what the user typed. It is purely additive: nothing here
may change a slug, a path, or how a project is addressed.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlmodel import Session, SQLModel, create_engine
from sqlalchemy.pool import StaticPool

from cowork.db.scoped import ScopedSession, TenantScope
from cowork.models.project import Project
from cowork.services.projects import (
    GENERAL_PROJECT,
    GENERAL_PROJECT_ID,
    ProjectService,
    display_label,
)

ORG = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path))
    from cowork.common.settings.app_settings import get_app_settings
    get_app_settings.cache_clear()
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as seed:
        seed.add(Project(id=GENERAL_PROJECT_ID, name=GENERAL_PROJECT,
                         path=str(tmp_path / "projects" / "general"), is_active=True))
        seed.commit()
    yield engine
    get_app_settings.cache_clear()


def _svc(engine) -> ProjectService:
    return ProjectService(ScopedSession(Session(engine), TenantScope(org_mode=True, org_id=ORG, user_id="u1")))


# ── the resolver ────────────────────────────────────────────────────────────

@pytest.fixture()
def api(tmp_path, monkeypatch):
    """A TestClient over the real app.

    Two things here are deliberate, and both are load-bearing:

    * NOT stacked on the `db` fixture. That one builds its own in-memory
      engine, so schema setup would run twice against different engines.
    * The client is yielded bare rather than entered as a context manager.
      `with TestClient(...)` runs the app lifespan, which re-creates the schema
      conftest's session-scoped `db_schema` already built -- `table
      channel_installations already exists`. Every other endpoint test in this
      suite yields it bare for the same reason.
    """
    from fastapi.testclient import TestClient
    from cowork.common.settings.app_settings import get_app_settings
    from cowork.server import create_app

    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    yield TestClient(create_app())
    get_app_settings.cache_clear()


def _post_project(client, name: str) -> dict:
    res = client.post("/api/v1/projects/", json={"name": name})
    assert res.status_code == 201, res.text
    return res.json()


def test_a_row_without_a_display_name_reads_as_its_slug():
    """Every project that predates the column. No backfill ran, so this is the
    state of all of them, and they must render exactly as they always have."""
    assert display_label(Project(name="reports", display_name=None, path="/x")) == "reports"


def test_a_row_with_a_display_name_reads_as_that():
    assert display_label(Project(name="untitled-project", display_name="Мій проєкт", path="/x")) == "Мій проєкт"


# ── creation ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("typed", [
    "Мій тестовий проєкт",   # Ukrainian — the reported case
    "我的项目",                # Chinese
    "テストプロジェクト",        # Japanese
    "Δοκιμαστικό έργο",      # Greek
    "مشروع تجريبي",           # Arabic
    "פרויקט בדיקה",           # Hebrew
    "Mon Projet Café",       # accented Latin — was silently mangled, not destroyed
    "My Test Project",       # plain ASCII with spaces
])
def test_the_typed_name_survives_verbatim(db, typed):
    project = _svc(db).create_project(typed)
    assert display_label(project) == typed


def test_the_slug_is_untouched_by_any_of_this(db):
    """The whole safety argument: `name` still comes out of the same ASCII
    sanitizer, so the directory, the routes and the lookup key are unchanged."""
    project = _svc(db).create_project("Мій тестовий проєкт")
    assert project.name == "untitled-project"
    assert project.name in project.path


def test_whitespace_only_input_stores_the_fallback_label_not_null(db):
    """NULL is reserved for "predates the column". A project created now always
    carries an explicit label, even when the user typed nothing meaningful —
    otherwise the slug would leak through the resolver on a brand-new row."""
    project = _svc(db).create_project("   ")
    assert project.display_name == "untitled-project"
    assert display_label(project) == "untitled-project"


# ── duplicates ──────────────────────────────────────────────────────────────

def test_two_projects_typed_the_same_do_not_display_the_same(db):
    """Slugs already differ (`untitled-project`, `-2`), but the labels would
    have been identical in the sidebar — the reason we dedupe on the label."""
    svc = _svc(db)
    first = svc.create_project("Мій проєкт")
    second = svc.create_project("Мій проєкт")
    third = svc.create_project("Мій проєкт")
    assert [display_label(p) for p in (first, second, third)] == [
        "Мій проєкт", "Мій проєкт 2", "Мій проєкт 3",
    ]
    assert first.name != second.name != third.name


def test_a_literal_name_collides_with_an_older_slug_labelled_project(db):
    """Dedupe compares the RESOLVED label, so a legacy row (display_name NULL,
    labelled by its slug) still blocks that exact string."""
    svc = _svc(db)
    legacy = svc.create_project("reports")
    assert legacy.name == "reports"
    typed_the_slug = svc.create_project("reports")
    assert display_label(typed_the_slug) == "reports 2"


def test_the_human_suffix_is_a_space_not_the_slug_hyphen(db):
    svc = _svc(db)
    svc.create_project("Quarterly Review")
    second = svc.create_project("Quarterly Review")
    assert display_label(second) == "Quarterly Review 2"
    assert display_label(second) != "Quarterly-Review-2"


# ── rename ──────────────────────────────────────────────────────────────────

def test_rename_updates_the_label_even_when_the_slug_cannot_move(db):
    """Two different Cyrillic names sanitize to the same slug, so the rename
    branch that moves the directory never fires — and the label still has to
    follow, because the user did rename the project."""
    svc = _svc(db)
    project = svc.create_project("Перший проєкт")
    slug_before = project.name
    renamed = svc.update_project(project.id, name="Другий проєкт")
    assert display_label(renamed) == "Другий проєкт"
    assert renamed.name == slug_before  # nothing moved on disk


def test_rename_still_moves_the_directory_when_the_slug_changes(db):
    """Today's mechanics are deliberately untouched by this ticket."""
    from pathlib import Path
    svc = _svc(db)
    project = svc.create_project("alpha")
    old_path = Path(project.path)
    assert old_path.exists()
    renamed = svc.update_project(project.id, name="beta")
    assert renamed.name == "beta"
    assert Path(renamed.path).exists()
    assert Path(renamed.path).name == "beta"
    assert not old_path.exists()
    assert display_label(renamed) == "beta"


def test_the_label_reaches_the_wire(api):
    """The renderer reads `display_name` off the project payload.

    The projects endpoints return the model directly with no response_model, so
    the field is exposed by adding it — but that is exactly the kind of thing a
    later `response_model=` would silently strip, taking the whole feature with
    it and leaving every project rendering its slug again.

    Asserted against the real HTTP response, not `model_dump()` on the instance
    the service returns. Those are not the same probe: `create_project` commits
    without refreshing, so the returned instance's attributes are expired and
    `model_dump()` omits every one of them. It happened to agree with the wire
    until ENG-1911 dropped the `session.refresh`, at which point the probe went
    red while the endpoint was still perfectly correct. The wire is the contract
    this test is named after, so the wire is what it now reads.
    """
    created = _post_project(api, "Мій тестовий проєкт")
    assert created["display_name"] == "Мій тестовий проєкт"
    # The slug collapsed, which is the bug; the exact suffix depends on what
    # else this session's DB already holds, so only the collapse is asserted.
    assert created["name"].startswith("untitled-project")

    # And on the read path, not just the create response.
    listed = api.get("/api/v1/projects/")
    assert listed.status_code == 200, listed.text
    row = next(p for p in listed.json() if p["id"] == created["id"])
    assert row["display_name"] == "Мій тестовий проєкт"


def test_a_same_slug_rename_still_moves_the_label(api):
    """Renaming Cyrillic -> Cyrillic changes nothing about the slug.

    `_sanitize_name` maps every non-Latin name to `untitled-project`, so the
    rename leaves `name`, the directory and the URL segment untouched while the
    label the user sees must still change. The endpoint has a fast path for
    "resolved name == current name" that treats the request as an unprotected
    active-selection update; this is the case that must NOT take it.
    """
    created = _post_project(api, "Мій тестовий проєкт")
    assert created["name"].startswith("untitled-project")

    renamed = api.patch(
        f"/api/v1/projects/{created['id']}",
        json={"name": "Інший проєкт"},
    )
    assert renamed.status_code == 200, renamed.text
    body = renamed.json()
    assert body["display_name"] == "Інший проєкт"
    # The identity half is genuinely untouched -- that is the whole design.
    assert body["name"] == created["name"]
    assert body["path"] == created["path"]


# --------------------------------------------------------------------------
# Org mode (web). The PATCH endpoint branches on tenancy before it branches on
# anything else: `not session.scope.org_mode` calls the service straight
# through, so the local-mode tests above never execute the shared-resource
# path at all. The label-only rename has to be proven on the org path too,
# because that is where the permission gate lives.
# --------------------------------------------------------------------------

ORG_ID = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ALICE = "11111111-1111-4111-8111-111111111111"
BOB = "22222222-2222-4222-8222-222222222222"


def _org_principal(user_id: str):
    from cowork.principal import Principal

    return Principal(
        user_id=user_id,
        org_id=ORG_ID,
        email=f"{user_id[0]}@example.com",
        roles=frozenset(),
    )


def _org_scoped(engine, user_id: str) -> ScopedSession:
    return ScopedSession(
        Session(engine),
        TenantScope(org_mode=True, org_id=ORG_ID, user_id=user_id),
    )


@pytest.fixture()
def org_engine(tmp_path, monkeypatch):
    from sqlalchemy.pool import StaticPool
    from cowork.common.settings.app_settings import get_app_settings

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.setenv("COWORK_MEMORY_DIR", str(tmp_path / "memory"))
    get_app_settings.cache_clear()
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    yield engine
    get_app_settings.cache_clear()


def test_org_same_slug_rename_moves_the_label_and_stays_gated(org_engine):
    """The org path's label-only rename: gated, and it actually writes.

    `resolved_name == current.name` has a fast path that treats the request as
    an unprotected active-selection update. A Cyrillic->Cyrillic rename lands
    exactly there -- same slug, different label -- so it must NOT take it:
      * the label has to change (otherwise ENG-1676 is unfixed on web), and
      * a non-creator must still be refused (otherwise any member could
        rewrite the name every other member reads, while the slug rename
        stays behind the creator/admin gate).
    """
    from cowork.api.v1.endpoints import projects as project_endpoints
    from cowork.schemas.projects import ProjectCreateRequest, ProjectUpdateRequest

    alice = _org_scoped(org_engine, ALICE)
    created = project_endpoints.create_project(
        ProjectCreateRequest(name="Мій тестовий проєкт"),
        alice,
        _org_principal(ALICE),
    )
    assert created["name"].startswith("untitled-project")
    assert created["display_name"] == "Мій тестовий проєкт"

    # Bob did not create it, and a label rename is still a rename.
    bob = _org_scoped(org_engine, BOB)
    with pytest.raises(HTTPException) as denied:
        project_endpoints.update_project(
            created["id"],
            ProjectUpdateRequest(name="Інший проєкт"),
            bob,
            _org_principal(BOB),
        )
    assert denied.value.status_code == 403

    renamed = project_endpoints.update_project(
        created["id"],
        ProjectUpdateRequest(name="Інший проєкт"),
        alice,
        _org_principal(ALICE),
    )
    assert renamed["display_name"] == "Інший проєкт"
    # Identity untouched: same slug, same directory. That is the design.
    assert renamed["name"] == created["name"]
    assert renamed["path"] == created["path"]
