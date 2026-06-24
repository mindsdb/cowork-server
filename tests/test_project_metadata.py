"""Tests for server-side project organization metadata.

Covers:
  * Metadata (pinned / sort_order / archived / last_selected_at) round-trips
    through the projects endpoints.
  * Reorder endpoint assigns sort_order.
  * include_archived filtering on the list endpoint.
  * The Alembic migration is backward compatible: a database created with the
    pre-metadata ``projects`` schema upgrades cleanly and back-fills defaults.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import UUID

import pytest
import sqlalchemy as sa
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from cowork.api.v1.router import api_router
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.project import Project
from cowork.models.project_collaboration import ProjectCollaborator
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.request_identity import RequestPrincipal


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(api_router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_extra_projects():
    """Remove any non-general projects before and after each test."""
    engine = get_engine(get_app_settings().database.uri)

    def _cleanup() -> None:
        with Session(engine) as session:
            for project in session.exec(select(Project)).all():
                if project.id == GENERAL_PROJECT_ID:
                    # Reset general's metadata so tests start from a clean slate.
                    project.pinned = False
                    project.sort_order = 0
                    project.archived = False
                    project.last_selected_at = None
                    session.add(project)
                else:
                    path = Path(project.path)
                    if path.is_dir():
                        import shutil

                        shutil.rmtree(path, ignore_errors=True)
                    session.delete(project)
            session.commit()

    _cleanup()
    yield
    _cleanup()


def _create_project(client: TestClient, name: str) -> dict:
    resp = client.post("/api/v1/projects/", json={"name": name})
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_new_project_has_default_metadata(client: TestClient):
    # Project endpoints return the raw ORM (snake_case).
    project = _create_project(client, "meta-defaults")
    assert project["pinned"] is False
    assert project["sort_order"] == 0
    assert project["archived"] is False
    assert project["last_selected_at"] is None


def test_metadata_round_trips_via_patch(client: TestClient):
    project = _create_project(client, "meta-roundtrip")
    project_id = project["id"]

    # Request bodies accept camelCase (CamelRequest) or snake_case.
    resp = client.patch(
        f"/api/v1/projects/{project_id}",
        json={"pinned": True, "sortOrder": 5, "archived": True},
    )
    assert resp.status_code == 200, resp.text
    updated = resp.json()
    assert updated["pinned"] is True
    assert updated["sort_order"] == 5
    assert updated["archived"] is True

    # Persisted across a fresh GET (include archived so it shows up).
    listed = client.get("/api/v1/projects/", params={"include_archived": True}).json()
    found = next(p for p in listed if p["id"] == project_id)
    assert found["pinned"] is True
    assert found["sort_order"] == 5
    assert found["archived"] is True


def test_partial_metadata_update_does_not_clobber(client: TestClient):
    project = _create_project(client, "meta-partial")
    project_id = project["id"]

    client.patch(f"/api/v1/projects/{project_id}", json={"pinned": True, "sortOrder": 3})
    # Only flip archived; pinned/sort_order must be preserved.
    resp = client.patch(f"/api/v1/projects/{project_id}", json={"archived": True})
    updated = resp.json()
    assert updated["pinned"] is True
    assert updated["sort_order"] == 3
    assert updated["archived"] is True


def test_selecting_project_touches_last_selected(client: TestClient):
    project = _create_project(client, "meta-active")
    project_id = project["id"]
    assert project["last_selected_at"] is None

    resp = client.patch(f"/api/v1/projects/{project_id}", json={"lastSelected": True})
    assert resp.status_code == 200, resp.text
    assert resp.json()["last_selected_at"] is not None


def test_list_include_archived_filter(client: TestClient):
    visible = _create_project(client, "meta-visible")
    hidden = _create_project(client, "meta-hidden")
    client.patch(f"/api/v1/projects/{hidden['id']}", json={"archived": True})

    active_only = client.get(
        "/api/v1/projects/", params={"include_archived": False}
    ).json()
    ids = {p["id"] for p in active_only}
    assert visible["id"] in ids
    assert hidden["id"] not in ids

    everything = client.get(
        "/api/v1/projects/", params={"include_archived": True}
    ).json()
    ids_all = {p["id"] for p in everything}
    assert hidden["id"] in ids_all


def test_reorder_endpoint_assigns_sort_order(client: TestClient):
    a = _create_project(client, "meta-order-a")
    b = _create_project(client, "meta-order-b")
    c = _create_project(client, "meta-order-c")

    # Desired display order: c, a, b
    resp = client.post(
        "/api/v1/projects/reorder",
        json={"projectIds": [c["id"], a["id"], b["id"]]},
    )
    assert resp.status_code == 200, resp.text

    listed = client.get("/api/v1/projects/").json()
    by_id = {p["id"]: p for p in listed}
    assert by_id[c["id"]]["sort_order"] == 0
    assert by_id[a["id"]]["sort_order"] == 1
    assert by_id[b["id"]]["sort_order"] == 2

    # The list endpoint orders by sort_order (among unpinned projects), so the
    # three reordered projects appear in c, a, b order relative to each other.
    positions = {p["id"]: i for i, p in enumerate(listed)}
    assert positions[c["id"]] < positions[a["id"]] < positions[b["id"]]


def test_owned_project_reorder_requires_owner_identity(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Reorder gates each owned project on the ``manage`` capability.

    Mirrors ``test_owned_project_update_and_delete_require_owner_identity`` in
    test_project_collaboration.py: once a project has an owner collaborator the
    reorder write path requires an authenticated owner (anonymous -> 401,
    non-owner -> 403, owner -> 200).
    """
    project_path = tmp_path / "owned-reorder"
    project_path.mkdir()
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        project = Project(name="owned-reorder", path=str(project_path))
        session.add(project)
        session.commit()
        session.refresh(project)
        project_id = project.id
        session.add(ProjectCollaborator(project_id=project_id, email="owner@example.com", role="owner"))
        session.add(ProjectCollaborator(project_id=project_id, email="reviewer@example.com", role="reviewer"))
        session.commit()

    def fake_principal(authorization: str | None):
        if not authorization:
            return None
        token = authorization.removeprefix("Bearer ").strip()
        if token == "owner":
            return RequestPrincipal("owner", "owner@example.com", "Owner", "test", {})
        if token == "reviewer":
            return RequestPrincipal("reviewer", "reviewer@example.com", "Reviewer", "test", {})
        return None

    monkeypatch.setattr(
        "cowork.api.v1.endpoints.projects.principal_from_authorization_header",
        fake_principal,
    )

    try:
        body = {"projectIds": [str(project_id)]}

        anonymous = client.post("/api/v1/projects/reorder", json=body)
        assert anonymous.status_code == 401, anonymous.text

        reviewer = client.post(
            "/api/v1/projects/reorder",
            headers={"Authorization": "Bearer reviewer"},
            json=body,
        )
        assert reviewer.status_code == 403, reviewer.text

        owner = client.post(
            "/api/v1/projects/reorder",
            headers={"Authorization": "Bearer owner"},
            json=body,
        )
        assert owner.status_code == 200, owner.text
        owned = next(p for p in owner.json() if p["id"] == str(project_id))
        assert owned["sort_order"] == 0
    finally:
        # Drop the seeded collaborator rows so they do not leak into the
        # autouse cleanup (which only resets General and removes projects).
        with Session(engine) as session:
            for row in session.exec(
                select(ProjectCollaborator).where(ProjectCollaborator.project_id == project_id)
            ).all():
                session.delete(row)
            session.commit()


def test_migration_backward_compatible_with_pre_metadata_schema(monkeypatch):
    """An existing DB at the prior revision upgrades cleanly to add metadata.

    Builds the real schema at the migration's down_revision (the revision just
    before this one) using the actual Alembic chain on an isolated temp DB, then
    runs the single upgrade step that adds the metadata columns. Asserts the new
    columns are added and back-fill with their defaults on a pre-existing row —
    i.e. an existing database without the columns upgrades without a backfill or
    a NOT NULL violation.
    """
    from alembic import command
    from alembic.config import Config

    from cowork.common.settings.app_settings import get_app_settings
    from cowork.db import session as session_module
    from cowork.db.migrations import _script_location

    # Revision under test and the one immediately before it.
    revision = "a3f7c9d1e2b4"
    down_revision = "fbe3964c2030"

    tmp = Path(tempfile.mkdtemp(prefix="cowork-migr-test-"))
    db_uri = f"sqlite:///{tmp / 'legacy.db'}"

    # env.py resolves the DB URL from app settings; point it at the temp DB for
    # the duration of this test (and reset the cached engine map) so the real
    # migration chain runs against an isolated database.
    settings = get_app_settings()
    monkeypatch.setattr(settings.database, "uri", db_uri)
    saved_engines = dict(session_module._engines)
    saved_factories = dict(session_module._session_factories)
    session_module._engines.clear()
    session_module._session_factories.clear()

    config = Config()
    config.set_main_option("script_location", str(_script_location()))
    config.set_main_option("sqlalchemy.url", db_uri)

    try:
        # Build the real schema as of the revision *before* the metadata change.
        command.upgrade(config, down_revision)

        engine = sa.create_engine(db_uri, connect_args={"check_same_thread": False})
        pre_columns = {c["name"] for c in sa.inspect(engine).get_columns("projects")}
        assert "pinned" not in pre_columns  # baseline truly predates the change

        # Seed a project row with the OLD schema (no metadata columns supplied).
        with engine.begin() as conn:
            conn.execute(
                sa.text(
                    "INSERT INTO projects (id, name, path, is_active) "
                    "VALUES (:id, :name, :path, 1)"
                ).bindparams(id="f" * 32, name="legacy", path=str(tmp / "legacy"))
            )

        # Apply exactly the metadata migration.
        command.upgrade(config, revision)

        # New columns exist.
        columns = {c["name"] for c in sa.inspect(engine).get_columns("projects")}
        assert {"pinned", "sort_order", "archived", "last_selected_at"}.issubset(columns)

        # Existing row back-filled with defaults (NOT NULL cols must not be null).
        with engine.begin() as conn:
            row = conn.execute(
                sa.text(
                    "SELECT pinned, sort_order, archived, last_selected_at "
                    "FROM projects WHERE name = 'legacy'"
                )
            ).one()
        pinned, sort_order, archived, last_selected_at = row
        assert bool(pinned) is False
        assert sort_order == 0
        assert bool(archived) is False
        assert last_selected_at is None

        # Downgrade removes the columns again (round-trip the migration).
        command.downgrade(config, down_revision)
        post_columns = {c["name"] for c in sa.inspect(engine).get_columns("projects")}
        assert "pinned" not in post_columns

        engine.dispose()
    finally:
        session_module._engines.clear()
        session_module._session_factories.clear()
        session_module._engines.update(saved_engines)
        session_module._session_factories.update(saved_factories)


def test_drop_is_active_migration_is_backward_compatible(monkeypatch):
    """An existing DB that still has ``projects.is_active`` upgrades cleanly to
    drop it (and the downgrade re-adds it). Builds the real schema at the
    revision just before the drop on an isolated temp DB, seeds a row with the
    column populated, then runs exactly the drop migration and asserts the
    column is gone while the row (and its metadata) survives.
    """
    from alembic import command
    from alembic.config import Config

    from cowork.common.settings.app_settings import get_app_settings
    from cowork.db import session as session_module
    from cowork.db.migrations import _script_location

    # The drop migration and the one immediately before it.
    revision = "c5d9e1f3a7b6"
    down_revision = "a3f7c9d1e2b4"

    tmp = Path(tempfile.mkdtemp(prefix="cowork-migr-drop-test-"))
    db_uri = f"sqlite:///{tmp / 'legacy.db'}"

    settings = get_app_settings()
    monkeypatch.setattr(settings.database, "uri", db_uri)
    saved_engines = dict(session_module._engines)
    saved_factories = dict(session_module._session_factories)
    session_module._engines.clear()
    session_module._session_factories.clear()

    config = Config()
    config.set_main_option("script_location", str(_script_location()))
    config.set_main_option("sqlalchemy.url", db_uri)

    try:
        # Build the real schema as of the revision *before* the drop — is_active
        # still exists here.
        command.upgrade(config, down_revision)

        engine = sa.create_engine(db_uri, connect_args={"check_same_thread": False})
        pre_columns = {c["name"] for c in sa.inspect(engine).get_columns("projects")}
        assert "is_active" in pre_columns  # baseline truly predates the drop

        # Seed a project row with is_active populated (the old contract).
        with engine.begin() as conn:
            conn.execute(
                sa.text(
                    "INSERT INTO projects (id, name, path, is_active) "
                    "VALUES (:id, :name, :path, 1)"
                ).bindparams(id="e" * 32, name="legacy-active", path=str(tmp / "legacy"))
            )

        # Apply exactly the drop migration.
        command.upgrade(config, revision)

        columns = {c["name"] for c in sa.inspect(engine).get_columns("projects")}
        assert "is_active" not in columns
        # The row survives the drop; the unified metadata column is intact.
        with engine.begin() as conn:
            name, last_selected_at = conn.execute(
                sa.text(
                    "SELECT name, last_selected_at FROM projects "
                    "WHERE name = 'legacy-active'"
                )
            ).one()
        assert name == "legacy-active"
        assert last_selected_at is None

        # Downgrade re-adds the column (schema round-trips) with its default.
        command.downgrade(config, down_revision)
        post_columns = {c["name"] for c in sa.inspect(engine).get_columns("projects")}
        assert "is_active" in post_columns

        engine.dispose()
    finally:
        session_module._engines.clear()
        session_module._session_factories.clear()
        session_module._engines.update(saved_engines)
        session_module._session_factories.update(saved_factories)
