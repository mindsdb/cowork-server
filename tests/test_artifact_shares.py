"""Tests for per-artifact shares + lightweight-signup accept.

Reuses the throwaway-SQLite + seeded-``general``-project bootstrap from
``tests/conftest.py``. The ``identity`` model module is auto-imported
there (``pkgutil.iter_modules`` over ``cowork.models``), so the
``users`` / ``artifact_shares`` tables exist without touching
``models/__init__.py``.

The router under test is mounted directly (the integration agent owns
the real ``router.py`` registration), under the same ``/api/v1/artifacts``
prefix the artifacts router uses.
"""
from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from cowork.api.v1.endpoints.artifact_shares import router as artifact_shares_router
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.artifact import Artifact
from cowork.models.identity import ArtifactShare, User
from cowork.models.project_collaboration import ProjectCollaborator
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.request_identity import RequestPrincipal


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(artifact_shares_router, prefix="/api/v1/artifacts")
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_share_rows():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        _delete_all(session)
    yield
    with Session(engine) as session:
        _delete_all(session)


def _delete_all(session: Session) -> None:
    for model in (ArtifactShare, User, ProjectCollaborator):
        for row in session.exec(select(model)).all():
            session.delete(row)
    session.commit()


def _make_artifact() -> str:
    """Create a minimal Artifact row and return its id."""
    engine = get_engine(get_app_settings().database.uri)
    slug = f"share-test-{uuid4().hex}"
    with Session(engine) as session:
        artifact = Artifact(
            project_id=GENERAL_PROJECT_ID,
            slug=slug,
            title="Share Test Artifact",
            path=f"/tmp/{slug}",
        )
        session.add(artifact)
        session.commit()
        session.refresh(artifact)
        return str(artifact.id)


def _own_general_project() -> None:
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        session.add(ProjectCollaborator(project_id=GENERAL_PROJECT_ID, email="owner@example.com", role="owner"))
        session.add(ProjectCollaborator(project_id=GENERAL_PROJECT_ID, email="editor@example.com", role="editor"))
        session.add(ProjectCollaborator(project_id=GENERAL_PROJECT_ID, email="viewer@example.com", role="viewer"))
        session.commit()


def _fake_share_principal(authorization: str | None):
    if not authorization:
        return None
    token = authorization.removeprefix("Bearer ").strip()
    if token == "editor":
        return RequestPrincipal("editor", "editor@example.com", "Editor", "test", {})
    if token == "viewer":
        return RequestPrincipal("viewer", "viewer@example.com", "Viewer", "test", {})
    if token == "owner":
        return RequestPrincipal("owner", "owner@example.com", "Owner", "test", {})
    return None


def test_create_share_returns_token_and_pending_row(client: TestClient):
    artifact_id = _make_artifact()

    response = client.post(
        f"/api/v1/artifacts/{artifact_id}/shares",
        json={"email": " Ada@Example.COM ", "role": "commenter"},
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    share = payload["share"]
    assert payload["created"] is True
    assert share["granteeEmail"] == "ada@example.com"
    assert share["role"] == "commenter"
    assert share["status"] == "pending"
    token = share["acceptToken"]
    assert token

    # Token is returned once and stored only as a hash at rest.
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        stored = session.exec(select(ArtifactShare).where(ArtifactShare.grantee_email == "ada@example.com")).one()
        assert stored.token_hash != token
        assert stored.status == "pending"
        # No User exists yet — viewing is open, signup happens on accept.
        assert session.exec(select(User).where(User.email == "ada@example.com")).first() is None

    # Invalid role is rejected.
    bad_role = client.post(
        f"/api/v1/artifacts/{artifact_id}/shares",
        json={"email": "ada@example.com", "role": "owner"},
    )
    assert bad_role.status_code == 400


def test_owned_project_share_routes_require_editor(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    _own_general_project()
    artifact_id = _make_artifact()
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifact_shares.principal_from_authorization_header",
        _fake_share_principal,
    )

    anonymous_create = client.post(
        f"/api/v1/artifacts/{artifact_id}/shares",
        json={"email": "ada@example.com", "role": "commenter"},
    )
    assert anonymous_create.status_code == 401

    viewer_create = client.post(
        f"/api/v1/artifacts/{artifact_id}/shares",
        headers={"Authorization": "Bearer viewer"},
        json={"email": "ada@example.com", "role": "commenter"},
    )
    assert viewer_create.status_code == 403

    editor_create = client.post(
        f"/api/v1/artifacts/{artifact_id}/shares",
        headers={"Authorization": "Bearer editor"},
        json={"email": "ada@example.com", "role": "commenter"},
    )
    assert editor_create.status_code == 201, editor_create.text
    share_id = editor_create.json()["share"]["id"]

    viewer_list = client.get(
        f"/api/v1/artifacts/{artifact_id}/shares",
        headers={"Authorization": "Bearer viewer"},
    )
    assert viewer_list.status_code == 403

    editor_list = client.get(
        f"/api/v1/artifacts/{artifact_id}/shares",
        headers={"Authorization": "Bearer editor"},
    )
    assert editor_list.status_code == 200, editor_list.text

    viewer_update = client.patch(
        f"/api/v1/artifacts/shares/{share_id}",
        headers={"Authorization": "Bearer viewer"},
        json={"role": "editor"},
    )
    assert viewer_update.status_code == 403

    editor_update = client.patch(
        f"/api/v1/artifacts/shares/{share_id}",
        headers={"Authorization": "Bearer editor"},
        json={"role": "editor"},
    )
    assert editor_update.status_code == 200, editor_update.text


def test_accept_share_creates_user_and_accepts_grant(client: TestClient):
    artifact_id = _make_artifact()
    created = client.post(
        f"/api/v1/artifacts/{artifact_id}/shares",
        json={"email": "ada@example.com", "role": "reviewer"},
    )
    assert created.status_code == 201, created.text
    token = created.json()["share"]["acceptToken"]

    # Wrong identity cannot accept.
    wrong = client.post(
        "/api/v1/artifacts/shares/accept",
        json={"token": token, "email": "other@example.com"},
    )
    assert wrong.status_code == 400, wrong.text
    assert "invited email" in wrong.text

    accepted = client.post(
        "/api/v1/artifacts/shares/accept",
        json={"token": token, "email": "ada@example.com", "displayName": "Ada Lovelace"},
    )
    assert accepted.status_code == 200, accepted.text
    body = accepted.json()
    assert body["userCreated"] is True
    assert body["user"]["email"] == "ada@example.com"
    assert body["user"]["displayName"] == "Ada Lovelace"
    assert body["share"]["status"] == "accepted"
    assert body["share"]["acceptedUserId"] == body["user"]["id"]
    assert "acceptToken" not in body["share"]

    # A used token cannot be replayed.
    replay = client.post(
        "/api/v1/artifacts/shares/accept",
        json={"token": token, "email": "ada@example.com"},
    )
    assert replay.status_code == 404, replay.text

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        user = session.exec(select(User).where(User.email == "ada@example.com")).one()
        assert user.display_name == "Ada Lovelace"


def test_list_shows_share(client: TestClient):
    artifact_id = _make_artifact()
    client.post(
        f"/api/v1/artifacts/{artifact_id}/shares",
        json={"email": "ada@example.com", "role": "viewer"},
    )

    listed = client.get(f"/api/v1/artifacts/{artifact_id}/shares")
    assert listed.status_code == 200, listed.text
    payload = listed.json()
    assert payload["artifactId"] == artifact_id
    assert [s["granteeEmail"] for s in payload["shares"]] == ["ada@example.com"]
    assert payload["shares"][0]["status"] == "pending"
    # Listing never leaks the accept token.
    assert "acceptToken" not in payload["shares"][0]


def test_set_share_role_via_patch(client: TestClient):
    artifact_id = _make_artifact()
    created = client.post(
        f"/api/v1/artifacts/{artifact_id}/shares",
        json={"email": "ada@example.com", "role": "viewer"},
    )
    share_id = created.json()["share"]["id"]

    updated = client.patch(f"/api/v1/artifacts/shares/{share_id}", json={"role": "editor"})
    assert updated.status_code == 200, updated.text
    assert updated.json()["share"]["role"] == "editor"

    revoked = client.patch(f"/api/v1/artifacts/shares/{share_id}", json={"status": "revoked"})
    assert revoked.status_code == 200, revoked.text
    assert revoked.json()["share"]["status"] == "revoked"
