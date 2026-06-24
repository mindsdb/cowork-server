from __future__ import annotations

import asyncio
import json
import shutil
import smtplib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from cowork.api.v1.router import api_router
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.artifact import Artifact
from cowork.models.project import Project
from cowork.models.project_collaboration import (
    NotificationDelivery,
    ProjectCollaborator,
    ProjectInvitation,
    ProjectNotificationHook,
)
from cowork.services.notifications import (
    EmailSmtpSender,
    SendResult,
    WebhookSender,
    dispatch_pending_notifications,
    send_notification_delivery,
)
from cowork.services.artifact_versions import ArtifactVersionService, record_deployment
from cowork.services.project_collaboration import decrypt_hook_secret
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.request_identity import RequestPrincipal


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(api_router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_collaboration_rows():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        _delete_all(session)
    project = _general_project_path()
    artifact_root = project / ".anton" / "artifacts"
    if artifact_root.is_dir():
        for folder in artifact_root.glob("collab-test-*"):
            shutil.rmtree(folder, ignore_errors=True)
    yield
    with Session(engine) as session:
        _delete_all(session)
    if artifact_root.is_dir():
        for folder in artifact_root.glob("collab-test-*"):
            shutil.rmtree(folder, ignore_errors=True)


def _delete_all(session: Session) -> None:
    for model in (NotificationDelivery, ProjectNotificationHook, ProjectInvitation, ProjectCollaborator):
        for row in session.exec(select(model)).all():
            session.delete(row)
    session.commit()


def _smtp_config(**overrides) -> dict:
    config = {
        "smtpHost": "smtp.example.com",
        "smtpPort": 587,
        "smtpUsername": "mailer@example.com",
        "smtpStartTls": True,
        "from": "cowork@example.com",
    }
    config.update(overrides)
    return config


def _general_project_path() -> Path:
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        project = session.get(Project, GENERAL_PROJECT_ID)
        assert project is not None
        return Path(project.path)


def _make_artifact(*, files: dict[str, str], artifact_type: str = "document") -> tuple[str, Path]:
    project = _general_project_path()
    slug = f"collab-test-{uuid4().hex}"
    artifact_id = f"artifact-{uuid4().hex}"
    folder = project / ".anton" / "artifacts" / slug
    folder.mkdir(parents=True)
    primary = next(iter(files), "")
    (folder / "metadata.json").write_text(
        json.dumps(
            {
                "id": artifact_id,
                "slug": slug,
                "name": "Collaboration Artifact",
                "description": "Artifact under review",
                "type": artifact_type,
                "primary": primary,
            }
        ),
        encoding="utf-8",
    )
    for rel_path, content in files.items():
        target = folder / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return artifact_id, folder


def _checkpoint(client: TestClient, artifact_id: str, **body) -> dict:
    response = client.post(f"/api/v1/artifacts/{artifact_id}/checkpoints", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def _fake_project_principal(authorization: str | None):
    if not authorization:
        return None
    token = authorization.removeprefix("Bearer ").strip()
    if token == "owner":
        return RequestPrincipal("owner", "owner@example.com", "Owner", "test", {})
    if token == "reviewer":
        return RequestPrincipal("reviewer", "reviewer@example.com", "Reviewer", "test", {})
    if token == "ada":
        return RequestPrincipal("ada", "ada@example.com", "Ada", "test", {})
    if token == "other":
        return RequestPrincipal("other", "other@example.com", "Other", "test", {})
    return None


def test_project_collaborators_are_normalized_and_upserted(client: TestClient):
    project_id = str(GENERAL_PROJECT_ID)

    created_response = client.post(
        f"/api/v1/projects/{project_id}/collaborators",
        json={"email": " Ada@Example.COM ", "displayName": "Ada", "role": "editor"},
    )
    assert created_response.status_code == 201, created_response.text
    created = created_response.json()
    collaborator = created["collaborator"]
    assert created["created"] is True
    assert collaborator["email"] == "ada@example.com"
    assert collaborator["displayName"] == "Ada"
    assert collaborator["role"] == "editor"

    duplicate_response = client.post(
        f"/api/v1/projects/{project_id}/collaborators",
        json={"email": "ADA@example.com", "role": "reviewer"},
    )
    assert duplicate_response.status_code == 201, duplicate_response.text
    duplicate = duplicate_response.json()
    assert duplicate["created"] is False
    assert duplicate["collaborator"]["id"] == collaborator["id"]
    assert duplicate["collaborator"]["role"] == "reviewer"

    listed_response = client.get(f"/api/v1/projects/{project_id}/collaborators")
    assert listed_response.status_code == 200, listed_response.text
    assert [item["email"] for item in listed_response.json()["collaborators"]] == ["ada@example.com"]

    invalid_response = client.post(
        f"/api/v1/projects/{project_id}/collaborators",
        json={"email": "reviewer@example.com", "role": "admin"},
    )
    assert invalid_response.status_code == 400

    deleted_response = client.delete(f"/api/v1/projects/{project_id}/collaborators/{collaborator['id']}")
    assert deleted_response.status_code == 204, deleted_response.text


def test_project_invitation_queues_email_and_accepts_invited_identity(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    project_id = str(GENERAL_PROJECT_ID)
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        session.add(ProjectCollaborator(project_id=GENERAL_PROJECT_ID, email="owner@example.com", role="owner"))
        session.commit()

    monkeypatch.setattr(
        "cowork.api.v1.endpoints.projects.principal_from_authorization_header",
        _fake_project_principal,
    )
    hook_response = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        headers={"Authorization": "Bearer owner"},
        json={
            "kind": "email",
            "target": "owner@example.com",
            "events": ["project.invited"],
            "secret": "smtp-password",
            "config": _smtp_config(subject="Cowork invitation"),
        },
    )
    assert hook_response.status_code == 201, hook_response.text

    invite_response = client.post(
        f"/api/v1/projects/{project_id}/invitations",
        headers={"Authorization": "Bearer owner"},
        json={"email": " Ada@Example.COM ", "displayName": "Ada", "role": "reviewer"},
    )
    assert invite_response.status_code == 201, invite_response.text
    payload = invite_response.json()
    invitation = payload["invitation"]
    token = invitation["acceptToken"]
    assert payload["created"] is True
    assert invitation["email"] == "ada@example.com"
    assert invitation["role"] == "reviewer"
    assert invitation["status"] == "pending"
    assert invitation["sendCount"] == 1
    assert invitation["invitedByEmail"] == "owner@example.com"
    assert len(payload["deliveries"]) == 1
    assert payload["deliveries"][0]["eventKey"] == "project.invited"
    assert payload["deliveries"][0]["details"]["recipientEmail"] == "ada@example.com"
    assert "inviteToken" not in payload["deliveries"][0]["details"]

    with Session(engine) as session:
        assert session.exec(select(ProjectCollaborator).where(ProjectCollaborator.email == "ada@example.com")).first() is None
        stored = session.exec(select(ProjectInvitation).where(ProjectInvitation.email == "ada@example.com")).one()
        assert stored.token_hash != token
        assert stored.notification_state["deliveries"][0]["details"]["recipientEmail"] == "ada@example.com"
        assert "inviteToken" not in stored.notification_state["deliveries"][0]["details"]

    wrong_identity = client.post(
        f"/api/v1/projects/{project_id}/invitations/accept",
        headers={"Authorization": "Bearer other"},
        json={"token": token},
    )
    assert wrong_identity.status_code == 400, wrong_identity.text
    assert "invited email" in wrong_identity.text

    accepted = client.post(
        f"/api/v1/projects/{project_id}/invitations/accept",
        headers={"Authorization": "Bearer ada"},
        json={"token": token},
    )
    assert accepted.status_code == 200, accepted.text
    accepted_payload = accepted.json()
    assert accepted_payload["created"] is True
    assert accepted_payload["collaborator"]["email"] == "ada@example.com"
    assert accepted_payload["collaborator"]["role"] == "reviewer"
    assert accepted_payload["invitation"]["status"] == "accepted"
    assert accepted_payload["invitation"]["acceptedByEmail"] == "ada@example.com"
    assert "acceptToken" not in accepted_payload["invitation"]


def test_project_invitation_resend_and_revoke_lifecycle(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    project_id = str(GENERAL_PROJECT_ID)
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        session.add(ProjectCollaborator(project_id=GENERAL_PROJECT_ID, email="owner@example.com", role="owner"))
        session.commit()

    monkeypatch.setattr(
        "cowork.api.v1.endpoints.projects.principal_from_authorization_header",
        _fake_project_principal,
    )
    hook_response = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        headers={"Authorization": "Bearer owner"},
        json={"kind": "email", "target": "owner@example.com", "events": ["project.invited"], "config": _smtp_config()},
    )
    assert hook_response.status_code == 201, hook_response.text
    invite = client.post(
        f"/api/v1/projects/{project_id}/invitations",
        headers={"Authorization": "Bearer owner"},
        json={"email": "reviewer@example.com", "role": "reviewer"},
    )
    assert invite.status_code == 201, invite.text
    invitation = invite.json()["invitation"]
    original_token = invitation["acceptToken"]

    resent = client.post(
        f"/api/v1/projects/{project_id}/invitations/{invitation['id']}/resend",
        headers={"Authorization": "Bearer owner"},
    )
    assert resent.status_code == 200, resent.text
    resent_invitation = resent.json()["invitation"]
    assert resent_invitation["acceptToken"] != original_token
    assert resent_invitation["sendCount"] == 2
    assert len(resent.json()["deliveries"]) == 1
    assert "inviteToken" not in resent.json()["deliveries"][0]["details"]

    listed = client.get(
        f"/api/v1/projects/{project_id}/invitations",
        headers={"Authorization": "Bearer owner"},
    )
    assert listed.status_code == 200, listed.text
    assert [item["id"] for item in listed.json()["invitations"]] == [invitation["id"]]

    revoked = client.post(
        f"/api/v1/projects/{project_id}/invitations/{invitation['id']}/revoke",
        headers={"Authorization": "Bearer owner"},
    )
    assert revoked.status_code == 200, revoked.text
    assert revoked.json()["invitation"]["status"] == "revoked"

    accept_revoked = client.post(
        f"/api/v1/projects/{project_id}/invitations/accept",
        headers={"Authorization": "Bearer reviewer"},
        json={"token": resent_invitation["acceptToken"]},
    )
    assert accept_revoked.status_code == 404, accept_revoked.text
    assert "not found" in accept_revoked.text.lower()


def test_project_notification_hook_masks_and_encrypts_secret(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    project_id = str(GENERAL_PROJECT_ID)
    secret = "https://example.com/reviews"

    response = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        json={
            "kind": "webhook",
            "target": "Review webhook",
            "events": ["artifact.suggested"],
            "secret": secret,
            "config": {"label": "Review room"},
        },
    )
    assert response.status_code == 201, response.text
    hook = response.json()["hook"]
    assert hook["kind"] == "webhook"
    assert hook["target"] == "Review webhook"
    assert hook["secretSet"] is True
    assert "secret" not in hook
    assert hook["events"] == ["artifact.suggested"]

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        row = session.get(ProjectNotificationHook, UUID(hook["id"]))
        assert row is not None
        assert row.secret_ciphertext != secret
        assert decrypt_hook_secret(row) == secret

    class FakeResponse:
        status_code = 200
        headers = {"x-request-id": "TEST123"}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, json):
            return FakeResponse()

    monkeypatch.setattr("cowork.services.notifications.httpx.AsyncClient", FakeAsyncClient)

    test_response = client.post(f"/api/v1/projects/{project_id}/notification-hooks/{hook['id']}/test")
    assert test_response.status_code == 200, test_response.text
    delivery = test_response.json()["delivery"]
    assert delivery["eventKey"] == "test"
    assert delivery["status"] == "sent"
    assert delivery["attempts"] == 1
    assert delivery["details"]["hookTarget"] == "Review webhook"

    second_test_response = client.post(f"/api/v1/projects/{project_id}/notification-hooks/{hook['id']}/test")
    assert second_test_response.status_code == 200, second_test_response.text
    assert second_test_response.json()["delivery"]["id"] != delivery["id"]

    deliveries_response = client.get(f"/api/v1/projects/{project_id}/notification-deliveries")
    assert deliveries_response.status_code == 200, deliveries_response.text
    assert len(deliveries_response.json()["deliveries"]) == 2

    unsupported_response = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        json={"kind": "chat", "target": "Review room", "secret": "https://hooks.example.com/chat"},
    )
    assert unsupported_response.status_code == 400


def test_email_hook_rejects_invalid_target_and_does_not_mask_password_as_target(client: TestClient):
    project_id = str(GENERAL_PROJECT_ID)
    response = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        json={
            "kind": "email",
            "target": "",
            "secret": "smtp-password",
            "config": _smtp_config(),
        },
    )
    assert response.status_code == 400
    assert "recipient email" in response.text
    assert "smtp-password" not in response.text

    invalid = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        json={
            "kind": "email",
            "target": "not an email",
            "secret": "smtp-password",
            "config": _smtp_config(),
        },
    )
    assert invalid.status_code == 400


def test_email_hook_config_rejects_secret_keys(client: TestClient):
    project_id = str(GENERAL_PROJECT_ID)
    response = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        json={
            "kind": "email",
            "target": "reviews@example.com",
            "config": {
                **_smtp_config(),
                "password": "do-not-return",
            },
        },
    )
    assert response.status_code == 400
    assert "secret value" in response.text
    assert "do-not-return" not in response.text


def test_email_hook_requires_sender_configuration(client: TestClient):
    project_id = str(GENERAL_PROJECT_ID)
    missing_host = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        json={"kind": "email", "target": "reviews@example.com", "config": {"from": "cowork@example.com"}},
    )
    assert missing_host.status_code == 400
    assert "smtp host" in missing_host.text.lower()

    invalid_from = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        json={"kind": "email", "target": "reviews@example.com", "config": {"smtpHost": "smtp.example.com", "from": "bad"}},
    )
    assert invalid_from.status_code == 400
    assert "from address" in invalid_from.text.lower()


def test_owned_project_notification_hooks_require_owner_identity(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    project_id = str(GENERAL_PROJECT_ID)
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        session.add(ProjectCollaborator(project_id=GENERAL_PROJECT_ID, email="owner@example.com", role="owner"))
        session.add(ProjectCollaborator(project_id=GENERAL_PROJECT_ID, email="reviewer@example.com", role="reviewer"))
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

    anonymous = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        json={"kind": "email", "target": "reviews@example.com"},
    )
    assert anonymous.status_code == 401

    reviewer = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        headers={"Authorization": "Bearer reviewer"},
        json={"kind": "email", "target": "reviews@example.com"},
    )
    assert reviewer.status_code == 403

    owner = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        headers={"Authorization": "Bearer owner"},
        json={"kind": "email", "target": "reviews@example.com", "config": _smtp_config()},
    )
    assert owner.status_code == 201, owner.text
    assert owner.json()["hook"]["target"] == "reviews@example.com"


def test_shared_project_cannot_lose_last_owner(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    project_id = str(GENERAL_PROJECT_ID)
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        owner = ProjectCollaborator(project_id=GENERAL_PROJECT_ID, email="owner@example.com", role="owner")
        session.add(owner)
        session.commit()
        session.refresh(owner)
        owner_id = owner.id

    def fake_principal(authorization: str | None):
        if authorization == "Bearer owner":
            return RequestPrincipal("owner", "owner@example.com", "Owner", "test", {})
        return None

    monkeypatch.setattr(
        "cowork.api.v1.endpoints.projects.principal_from_authorization_header",
        fake_principal,
    )

    demote = client.patch(
        f"/api/v1/projects/{project_id}/collaborators/{owner_id}",
        headers={"Authorization": "Bearer owner"},
        json={"role": "reviewer"},
    )
    assert demote.status_code == 400
    assert "owner" in demote.text.lower()

    upsert_demote = client.post(
        f"/api/v1/projects/{project_id}/collaborators",
        headers={"Authorization": "Bearer owner"},
        json={"email": "owner@example.com", "role": "reviewer"},
    )
    assert upsert_demote.status_code == 400

    delete = client.delete(
        f"/api/v1/projects/{project_id}/collaborators/{owner_id}",
        headers={"Authorization": "Bearer owner"},
    )
    assert delete.status_code == 400

    anonymous_hook = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        json={"kind": "email", "target": "reviews@example.com"},
    )
    assert anonymous_hook.status_code == 401


def test_authenticated_project_creation_bootstraps_owner(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_principal(authorization: str | None):
        if authorization == "Bearer owner":
            return RequestPrincipal("owner", "owner@example.com", "Owner", "test", {})
        return None

    monkeypatch.setattr(
        "cowork.api.v1.endpoints.projects.principal_from_authorization_header",
        fake_principal,
    )

    response = client.post(
        "/api/v1/projects/",
        headers={"Authorization": "Bearer owner"},
        json={"name": f"collab-bootstrap-{uuid4().hex[:8]}"},
    )
    assert response.status_code == 201, response.text
    project = response.json()

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        collaborator = session.exec(
            select(ProjectCollaborator)
            .where(ProjectCollaborator.project_id == UUID(project["id"]))
            .where(ProjectCollaborator.email == "owner@example.com")
        ).one()
        assert collaborator.role == "owner"
        row = session.get(Project, UUID(project["id"]))
        if row is not None:
            path = Path(row.path)
            session.delete(row)
            session.commit()
            shutil.rmtree(path, ignore_errors=True)


def test_owned_project_update_and_delete_require_owner_identity(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    project_path = tmp_path / "owned-project"
    project_path.mkdir()
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        project = Project(name=f"owned-{uuid4().hex[:8]}", path=str(project_path))
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

    anonymous = client.patch(f"/api/v1/projects/{project_id}", json={"lastSelected": True})
    assert anonymous.status_code == 401

    reviewer = client.delete(
        f"/api/v1/projects/{project_id}",
        headers={"Authorization": "Bearer reviewer"},
    )
    assert reviewer.status_code == 403

    owner_update = client.patch(
        f"/api/v1/projects/{project_id}",
        headers={"Authorization": "Bearer owner"},
        json={"lastSelected": True},
    )
    assert owner_update.status_code == 200, owner_update.text
    assert owner_update.json()["last_selected_at"] is not None

    owner_delete = client.delete(
        f"/api/v1/projects/{project_id}",
        headers={"Authorization": "Bearer owner"},
    )
    assert owner_delete.status_code == 204, owner_delete.text
    assert not project_path.exists()


def test_artifact_suggestion_records_project_notification_delivery(client: TestClient):
    project_id = str(GENERAL_PROJECT_ID)
    hook_response = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        json={"kind": "email", "target": "reviews@example.com", "events": ["artifact.suggested"], "config": _smtp_config()},
    )
    assert hook_response.status_code == 201, hook_response.text

    artifact_id, folder = _make_artifact(files={"report.md": "# Draft\n"})
    _checkpoint(client, artifact_id, label="Baseline")
    comment_response = client.post(
        "/api/v1/artifacts/comments",
        json={
            "path": str(folder),
            "kind": "suggestion",
            "body": "Tighten the summary.",
            "anchor": {"path": "report.md", "line": 1},
        },
    )
    assert comment_response.status_code == 201, comment_response.text
    comment = comment_response.json()["comment"]
    deliveries = comment["notificationState"]["deliveries"]
    assert len(deliveries) == 1
    assert deliveries[0]["eventKey"] == "artifact.suggested"
    assert deliveries[0]["status"] == "queued"

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        rows = session.exec(select(NotificationDelivery)).all()
        assert len(rows) == 1
        assert rows[0].event_key == "artifact.suggested"
        assert rows[0].details["commentId"] == comment["id"]


def test_all_updates_email_hook_receives_artifact_lifecycle_events(client: TestClient):
    project_id = str(GENERAL_PROJECT_ID)
    hook_response = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        json={"kind": "email", "target": "ops@example.com", "events": ["*"], "config": _smtp_config()},
    )
    assert hook_response.status_code == 201, hook_response.text

    artifact_id, folder = _make_artifact(files={"report.md": "# Draft\n"})
    first = _checkpoint(client, artifact_id, label="Baseline")
    first_version_id = UUID(first["version"]["id"])

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        artifact = session.exec(select(Artifact).where(Artifact.path == str(folder.resolve()))).one()
        first_version = ArtifactVersionService(session)._get_version(first_version_id)
        record_deployment(
            session,
            first_version,
            target="publish",
            status="published",
            url="https://example.com/report",
            details={"access": {"mode": "public"}},
        )
        (folder / "report.md").write_text("# Generated\n", encoding="utf-8")
        generated = ArtifactVersionService(session).snapshot_artifact(
            folder,
            artifact_id=artifact.id,
            operation_type="generated_update",
            label="Generated update",
        )
        record_deployment(
            session,
            generated,
            target="preview",
            status="failed",
            url=None,
            details={"error": "Preview crashed"},
        )
        ArtifactVersionService(session).restore_version(first_version.id, folder, label="Restore baseline")

    deleted_response = client.delete("/api/v1/artifacts/", params={"path": str(folder)})
    assert deleted_response.status_code == 204, deleted_response.text

    with Session(engine) as session:
        event_keys = [row.event_key for row in session.exec(select(NotificationDelivery)).all()]
        assert "artifact.published" in event_keys
        assert "artifact.generated_updated" in event_keys
        assert "artifact.preview_failed" in event_keys
        assert "artifact.restored" in event_keys
        assert "artifact.deleted" in event_keys
        for row in session.exec(select(NotificationDelivery)).all():
            assert row.details["hookKind"] == "email"


def test_owned_project_artifact_comments_respect_collaborator_roles(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        session.add(ProjectCollaborator(project_id=GENERAL_PROJECT_ID, email="owner@example.com", role="owner"))
        session.add(ProjectCollaborator(project_id=GENERAL_PROJECT_ID, email="commenter@example.com", role="commenter"))
        session.add(ProjectCollaborator(project_id=GENERAL_PROJECT_ID, email="reviewer@example.com", role="reviewer"))
        session.commit()

    def fake_principal(authorization: str | None):
        if not authorization:
            return None
        token = authorization.removeprefix("Bearer ").strip()
        if token == "commenter":
            return RequestPrincipal("commenter", "commenter@example.com", "Commenter", "test", {})
        if token == "reviewer":
            return RequestPrincipal("reviewer", "reviewer@example.com", "Reviewer", "test", {})
        return None

    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifacts.principal_from_authorization_header",
        fake_principal,
    )

    _artifact_id, folder = _make_artifact(files={"report.md": "# Draft\n"})

    anonymous = client.post(
        "/api/v1/artifacts/comments",
        json={"path": str(folder), "kind": "comment", "body": "Looks good."},
    )
    assert anonymous.status_code == 401

    commenter = client.post(
        "/api/v1/artifacts/comments",
        headers={"Authorization": "Bearer commenter"},
        json={"path": str(folder), "kind": "comment", "body": "Looks good.", "actorName": "Owner"},
    )
    assert commenter.status_code == 201, commenter.text
    assert commenter.json()["comment"]["actorName"] == "Commenter"

    commenter_suggestion = client.post(
        "/api/v1/artifacts/comments",
        headers={"Authorization": "Bearer commenter"},
        json={"path": str(folder), "kind": "suggestion", "body": "Tighten this."},
    )
    assert commenter_suggestion.status_code == 403

    reviewer = client.post(
        "/api/v1/artifacts/comments",
        headers={"Authorization": "Bearer reviewer"},
        json={"path": str(folder), "kind": "suggestion", "body": "Tighten this.", "actorName": "Owner"},
    )
    assert reviewer.status_code == 201, reviewer.text
    suggestion = reviewer.json()["comment"]
    assert suggestion["actorName"] == "Reviewer"

    rejected = client.post(
        f"/api/v1/artifacts/comments/{suggestion['id']}/reject",
        headers={"Authorization": "Bearer reviewer"},
    )
    assert rejected.status_code == 200, rejected.text

    listed = client.get(
        "/api/v1/artifacts/comments",
        headers={"Authorization": "Bearer reviewer"},
        params={"path": str(folder)},
    )
    assert listed.status_code == 200, listed.text
    reject_event = next(event for event in listed.json()["activity"] if event["eventType"] == "rejected")
    assert reject_event["actorName"] == "Reviewer"
    assert reject_event["details"]["actorEmail"] == "reviewer@example.com"
    assert reject_event["details"]["actorSubject"] == "reviewer"


def test_owned_project_artifact_list_uses_signed_serve_urls(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        session.add(ProjectCollaborator(project_id=GENERAL_PROJECT_ID, email="owner@example.com", role="owner"))
        session.add(ProjectCollaborator(project_id=GENERAL_PROJECT_ID, email="viewer@example.com", role="viewer"))
        session.commit()

    def fake_principal(authorization: str | None):
        if authorization == "Bearer viewer":
            return RequestPrincipal("viewer", "viewer@example.com", "Viewer", "test", {})
        return None

    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifacts.principal_from_authorization_header",
        fake_principal,
    )

    artifact_id, _folder = _make_artifact(files={"index.html": "<h1>Private draft</h1>\n"}, artifact_type="html-app")

    anonymous = client.get("/api/v1/artifacts/")
    assert anonymous.status_code == 200, anonymous.text
    assert artifact_id not in {item["id"] for item in anonymous.json()}

    listed = client.get("/api/v1/artifacts/", headers={"Authorization": "Bearer viewer"})
    assert listed.status_code == 200, listed.text
    card = next(item for item in listed.json() if item["id"] == artifact_id)
    assert "?token=" in card["serveUrl"]

    raw_serve_url = card["serveUrl"].split("?", 1)[0]
    unsigned = client.get(raw_serve_url)
    assert unsigned.status_code == 401, unsigned.text

    signed = client.get(card["serveUrl"])
    assert signed.status_code == 200, signed.text
    assert "<h1>Private draft</h1>" in signed.text


def test_notification_dispatcher_marks_queued_delivery_sent(client: TestClient):
    class FakeSender:
        kind = "email"

        def __init__(self):
            self.sent = []

        async def send(self, hook, delivery):
            self.sent.append((hook.id, delivery.id))
            return SendResult(status="sent", external_id="fake-delivery-id")

    project_id = str(GENERAL_PROJECT_ID)
    hook_response = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        json={"kind": "email", "target": "reviews@example.com", "events": ["artifact.suggested"], "config": _smtp_config()},
    )
    assert hook_response.status_code == 201, hook_response.text
    artifact_id, folder = _make_artifact(files={"report.md": "# Draft\n"})
    _checkpoint(client, artifact_id, label="Baseline")
    comment_response = client.post(
        "/api/v1/artifacts/comments",
        json={
            "path": str(folder),
            "kind": "suggestion",
            "body": "Tighten the summary.",
        },
    )
    assert comment_response.status_code == 201, comment_response.text

    engine = get_engine(get_app_settings().database.uri)
    fake_sender = FakeSender()
    with Session(engine) as session:
        results = asyncio.run(dispatch_pending_notifications(session, senders={"email": fake_sender}))
        assert len(results) == 1
        row = session.exec(select(NotificationDelivery)).one()
        assert row.status == "sent"
        assert row.attempts == 1
        assert row.error is None
        assert row.details["externalId"] == "fake-delivery-id"
        assert fake_sender.sent == [(row.hook_id, row.id)]


def test_artifact_review_email_hook_dispatches_through_smtp(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple] = []

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            calls.append(("connect", host, port, timeout))

        def __enter__(self):
            calls.append(("enter",))
            return self

        def __exit__(self, *args):
            calls.append(("exit",))
            return None

        def starttls(self):
            calls.append(("starttls",))

        def login(self, username, password):
            calls.append(("login", username, password))

        def send_message(self, message):
            calls.append(("send_message", message["From"], message["To"], message["Subject"], message.get_content()))

    monkeypatch.setattr("cowork.services.notifications.smtplib.SMTP", FakeSMTP)

    project_id = str(GENERAL_PROJECT_ID)
    hook_response = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        json={
            "kind": "email",
            "target": "reviews@example.com",
            "secret": "smtp-password",
            "events": ["artifact.review_requested"],
            "config": _smtp_config(subject="Review requested"),
        },
    )
    assert hook_response.status_code == 201, hook_response.text

    artifact_id, folder = _make_artifact(files={"report.md": "# Draft\n"})
    _checkpoint(client, artifact_id, label="Baseline")
    comment_response = client.post(
        "/api/v1/artifacts/comments",
        json={"path": str(folder), "kind": "review", "body": "Please review this draft."},
    )
    assert comment_response.status_code == 201, comment_response.text

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        delivery = session.exec(select(NotificationDelivery)).one()
        assert delivery.event_key == "artifact.review_requested"
        assert delivery.status == "queued"

        results = asyncio.run(dispatch_pending_notifications(session))
        assert len(results) == 1
        session.refresh(delivery)
        assert delivery.status == "sent"
        assert delivery.attempts == 1
        assert delivery.error is None
        assert delivery.details["externalId"] == "email:reviews@example.com"

    assert calls[0] == ("connect", "smtp.example.com", 587, 15)
    assert ("starttls",) in calls
    assert ("login", "mailer@example.com", "smtp-password") in calls
    sent = next(call for call in calls if call[0] == "send_message")
    assert sent[1:4] == ("cowork@example.com", "reviews@example.com", "Review requested")
    assert "review requested" in sent[4]


def test_send_delivery_skips_unclaimable_delivery(client: TestClient):
    class FakeSender:
        kind = "email"

        def __init__(self):
            self.sent = []

        async def send(self, hook, delivery):
            self.sent.append(delivery.id)
            return SendResult(status="sent")

    project_id = str(GENERAL_PROJECT_ID)
    hook_response = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        json={"kind": "email", "target": "reviews@example.com", "events": ["artifact.suggested"], "config": _smtp_config()},
    )
    assert hook_response.status_code == 201, hook_response.text
    hook = hook_response.json()["hook"]
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        delivery = NotificationDelivery(
            project_id=GENERAL_PROJECT_ID,
            hook_id=UUID(hook["id"]),
            event_key="artifact.suggested",
            dedupe_key=f"claim-test:{uuid4()}",
            status="sending",
            attempts=1,
            details={},
        )
        session.add(delivery)
        session.commit()
        session.refresh(delivery)
        delivery_id = delivery.id

        fake_sender = FakeSender()
        result = asyncio.run(send_notification_delivery(session, delivery_id, senders={"email": fake_sender}))

        row = session.get(NotificationDelivery, delivery_id)
        assert result["skipped"] is True
        assert row is not None
        assert row.status == "sending"
        assert fake_sender.sent == []


def test_dispatcher_claims_delivery_once_under_concurrent_workers(client: TestClient):
    class SlowSender:
        kind = "email"

        def __init__(self):
            self.sent = []

        async def send(self, hook, delivery):
            self.sent.append(delivery.id)
            await asyncio.sleep(0.01)
            return SendResult(status="sent", external_id="sent-once")

    project_id = str(GENERAL_PROJECT_ID)
    hook_response = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        json={"kind": "email", "target": "reviews@example.com", "events": ["artifact.suggested"], "config": _smtp_config()},
    )
    assert hook_response.status_code == 201, hook_response.text
    hook = hook_response.json()["hook"]
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        delivery = NotificationDelivery(
            project_id=GENERAL_PROJECT_ID,
            hook_id=UUID(hook["id"]),
            event_key="artifact.suggested",
            dedupe_key=f"concurrent-claim:{uuid4()}",
            status="queued",
            details={},
        )
        session.add(delivery)
        session.commit()
        session.refresh(delivery)
        delivery_id = delivery.id

    fake_sender = SlowSender()

    async def run_two_workers():
        with Session(engine) as first, Session(engine) as second:
            return await asyncio.gather(
                send_notification_delivery(first, delivery_id, senders={"email": fake_sender}),
                send_notification_delivery(second, delivery_id, senders={"email": fake_sender}),
            )

    results = asyncio.run(run_two_workers())
    assert len(fake_sender.sent) == 1
    assert any(result.get("skipped") for result in results)
    with Session(engine) as session:
        row = session.get(NotificationDelivery, delivery_id)
        assert row is not None
        assert row.status == "sent"
        assert row.attempts == 1


def test_retryable_failure_waits_until_next_attempt(client: TestClient):
    class RetrySender:
        kind = "email"

        async def send(self, hook, delivery):
            return SendResult(status="failed", error="temporary", retryable=True)

    project_id = str(GENERAL_PROJECT_ID)
    hook_response = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        json={"kind": "email", "target": "reviews@example.com", "events": ["artifact.suggested"], "config": _smtp_config()},
    )
    assert hook_response.status_code == 201, hook_response.text
    artifact_id, folder = _make_artifact(files={"report.md": "# Draft\n"})
    _checkpoint(client, artifact_id, label="Baseline")
    comment_response = client.post(
        "/api/v1/artifacts/comments",
        json={"path": str(folder), "kind": "suggestion", "body": "Tighten the summary."},
    )
    assert comment_response.status_code == 201, comment_response.text

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        first = asyncio.run(dispatch_pending_notifications(session, senders={"email": RetrySender()}))
        assert len(first) == 1
        row = session.exec(select(NotificationDelivery)).one()
        assert row.status == "failed"
        assert row.attempts == 1
        assert row.details["retryable"] is True
        assert "nextAttemptAt" in row.details

        second = asyncio.run(dispatch_pending_notifications(session, senders={"email": RetrySender()}))
        assert second == []
        session.refresh(row)
        assert row.attempts == 1

        row.details = {**row.details, "nextAttemptAt": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()}
        session.add(row)
        session.commit()
        third = asyncio.run(dispatch_pending_notifications(session, senders={"email": RetrySender()}))
        assert len(third) == 1
        session.refresh(row)
        assert row.attempts == 2


def test_max_attempts_marks_delivery_exhausted(client: TestClient):
    class RetrySender:
        kind = "email"

        async def send(self, hook, delivery):
            return SendResult(status="failed", error="temporary", retryable=True)

    project_id = str(GENERAL_PROJECT_ID)
    hook_response = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        json={"kind": "email", "target": "reviews@example.com", "events": ["artifact.suggested"], "config": _smtp_config()},
    )
    assert hook_response.status_code == 201, hook_response.text
    hook = hook_response.json()["hook"]
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        delivery = NotificationDelivery(
            project_id=GENERAL_PROJECT_ID,
            hook_id=UUID(hook["id"]),
            event_key="artifact.suggested",
            dedupe_key=f"exhausted:{uuid4()}",
            status="queued",
            attempts=4,
            details={},
        )
        session.add(delivery)
        session.commit()
        session.refresh(delivery)
        result = asyncio.run(send_notification_delivery(session, delivery.id, senders={"email": RetrySender()}))
        exhausted = result["delivery"]
        assert exhausted["status"] == "exhausted"
        assert exhausted["attempts"] == 5
        assert exhausted["details"]["retryable"] is False
        assert "exhaustedAt" in exhausted["details"]


def test_notification_delivery_retry_endpoint_requeues_failed_delivery(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    project_id = str(GENERAL_PROJECT_ID)
    hook_response = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        json={"kind": "webhook", "target": "Review webhook", "secret": "https://example.com/hook"},
    )
    assert hook_response.status_code == 201, hook_response.text
    hook = hook_response.json()["hook"]

    class FakeResponse:
        status_code = 200
        headers = {}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, json):
            return FakeResponse()

    monkeypatch.setattr("cowork.services.notifications.httpx.AsyncClient", FakeAsyncClient)

    delivery_response = client.post(f"/api/v1/projects/{project_id}/notification-hooks/{hook['id']}/test")
    assert delivery_response.status_code == 200, delivery_response.text
    delivery = delivery_response.json()["delivery"]

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        row = session.get(NotificationDelivery, UUID(delivery["id"]))
        assert row is not None
        row.status = "failed"
        row.attempts = 1
        row.error = "timeout"
        row.details = {**(row.details or {}), "retryable": True}
        session.add(row)
        session.commit()

    listed_response = client.get(f"/api/v1/projects/{project_id}/notification-deliveries")
    assert listed_response.status_code == 200, listed_response.text
    assert listed_response.json()["deliveries"][0]["status"] == "failed"

    retry_response = client.post(f"/api/v1/projects/{project_id}/notification-deliveries/{delivery['id']}/retry")
    assert retry_response.status_code == 200, retry_response.text
    retried = retry_response.json()["delivery"]
    assert retried["status"] == "queued"
    assert retried["attempts"] == 0
    assert retried["error"] is None
    assert retried["details"]["retryable"] is True
    assert "retryRequestedAt" in retried["details"]
    assert retried["details"]["previousAttempts"] == 1


def test_webhook_sender_posts_sanitized_payload(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    project_id = str(GENERAL_PROJECT_ID)
    secret = "https://example.com/reviews"
    hook_response = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        json={"kind": "webhook", "target": "Review webhook", "secret": secret},
    )
    assert hook_response.status_code == 201, hook_response.text
    hook = hook_response.json()["hook"]
    calls = []

    class FakeResponse:
        status_code = 200
        headers = {"x-request-id": "REQ123"}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, json):
            calls.append((url, json))
            return FakeResponse()

    monkeypatch.setattr("cowork.services.notifications.httpx.AsyncClient", FakeAsyncClient)

    delivery_response = client.post(f"/api/v1/projects/{project_id}/notification-hooks/{hook['id']}/test")
    assert delivery_response.status_code == 200, delivery_response.text
    delivery = delivery_response.json()["delivery"]

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        hook_row = session.get(ProjectNotificationHook, UUID(hook["id"]))
        delivery_row = session.get(NotificationDelivery, UUID(delivery["id"]))
        assert hook_row is not None
        assert delivery_row is not None
        result = asyncio.run(WebhookSender().send(hook_row, delivery_row))

    assert result.status == "sent"
    assert result.external_id == "REQ123"
    assert calls[0][0] == secret
    assert secret not in json.dumps(calls[0][1])


def test_email_smtp_sender_uses_tls_login_and_send_message(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    project_id = str(GENERAL_PROJECT_ID)
    password = "smtp-secret-password"
    hook_response = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        json={
            "kind": "email",
            "target": "reviews@example.com",
            "secret": password,
            "config": {
                "smtpHost": "smtp.example.com",
                "smtpPort": 2525,
                "smtpUsername": "mailer@example.com",
                "smtpStartTls": True,
                "from": "cowork@example.com",
                "subject": "Review update",
            },
        },
    )
    assert hook_response.status_code == 201, hook_response.text
    hook = hook_response.json()["hook"]
    calls: list[tuple] = []

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            calls.append(("connect", host, port, timeout))

        def __enter__(self):
            calls.append(("enter",))
            return self

        def __exit__(self, *args):
            calls.append(("exit",))
            return None

        def starttls(self):
            calls.append(("starttls",))

        def login(self, username, password_value):
            calls.append(("login", username, password_value))

        def send_message(self, message):
            calls.append(("send_message", message["From"], message["To"], message["Subject"], message.get_content()))

    monkeypatch.setattr("cowork.services.notifications.smtplib.SMTP", FakeSMTP)

    delivery_response = client.post(f"/api/v1/projects/{project_id}/notification-hooks/{hook['id']}/test")
    assert delivery_response.status_code == 200, delivery_response.text
    delivery = delivery_response.json()["delivery"]
    assert delivery["status"] == "sent"
    assert delivery["attempts"] == 1

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        hook_row = session.get(ProjectNotificationHook, UUID(hook["id"]))
        delivery_row = session.get(NotificationDelivery, UUID(delivery["id"]))
        assert hook_row is not None
        assert delivery_row is not None
        result = asyncio.run(EmailSmtpSender().send(hook_row, delivery_row))

    assert result.status == "sent"
    assert result.external_id == "email:reviews@example.com"
    assert calls[0] == ("connect", "smtp.example.com", 2525, 15)
    assert ("starttls",) in calls
    assert ("login", "mailer@example.com", password) in calls
    send_call = next(call for call in calls if call[0] == "send_message")
    assert send_call[1:4] == ("cowork@example.com", "reviews@example.com", "Review update")
    assert "smtp-secret-password" not in send_call[4]


def test_email_smtp_sender_auth_failure_is_non_retryable(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    project_id = str(GENERAL_PROJECT_ID)
    hook_response = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        json={
            "kind": "email",
            "target": "reviews@example.com",
            "secret": "bad-password",
            "config": {
                "smtpHost": "smtp.example.com",
                "smtpUsername": "mailer@example.com",
                "from": "cowork@example.com",
            },
        },
    )
    assert hook_response.status_code == 201, hook_response.text
    hook = hook_response.json()["hook"]

    class AuthFailSMTP:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def starttls(self):
            pass

        def login(self, username, password):
            raise smtplib.SMTPAuthenticationError(535, b"authentication failed for bad-password")

        def send_message(self, message):
            raise AssertionError("send_message should not be called after auth failure")

    monkeypatch.setattr("cowork.services.notifications.smtplib.SMTP", AuthFailSMTP)

    delivery_response = client.post(f"/api/v1/projects/{project_id}/notification-hooks/{hook['id']}/test")
    assert delivery_response.status_code == 200, delivery_response.text
    delivery = delivery_response.json()["delivery"]
    assert delivery["status"] == "failed"
    assert delivery["error"] == "smtp_auth_failed"

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        hook_row = session.get(ProjectNotificationHook, UUID(hook["id"]))
        delivery_row = session.get(NotificationDelivery, UUID(delivery["id"]))
        assert hook_row is not None
        assert delivery_row is not None
        result = asyncio.run(EmailSmtpSender().send(hook_row, delivery_row))

    assert result.status == "failed"
    assert result.error == "smtp_auth_failed"
    assert result.retryable is False
    assert "bad-password" not in json.dumps(result.details or {})


def test_email_smtp_sender_451_is_retryable(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    project_id = str(GENERAL_PROJECT_ID)
    hook_response = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        json={
            "kind": "email",
            "target": "reviews@example.com",
            "secret": "smtp-password",
            "config": _smtp_config(),
        },
    )
    assert hook_response.status_code == 201, hook_response.text
    hook = hook_response.json()["hook"]

    class TemporarySMTP:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def starttls(self):
            pass

        def login(self, username, password):
            pass

        def send_message(self, message):
            raise smtplib.SMTPResponseException(451, b"try again later with smtp-password")

    monkeypatch.setattr("cowork.services.notifications.smtplib.SMTP", TemporarySMTP)

    delivery_response = client.post(f"/api/v1/projects/{project_id}/notification-hooks/{hook['id']}/test")
    assert delivery_response.status_code == 200, delivery_response.text
    delivery = delivery_response.json()["delivery"]
    assert delivery["status"] == "failed"
    assert delivery["error"] == "smtp_451"
    assert delivery["details"]["retryable"] is True
    assert delivery["details"]["smtpCode"] == 451
    assert "smtp-password" not in json.dumps(delivery)


def test_email_smtp_sender_550_is_non_retryable(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    project_id = str(GENERAL_PROJECT_ID)
    hook_response = client.post(
        f"/api/v1/projects/{project_id}/notification-hooks",
        json={
            "kind": "email",
            "target": "reviews@example.com",
            "secret": "smtp-password",
            "config": _smtp_config(),
        },
    )
    assert hook_response.status_code == 201, hook_response.text
    hook = hook_response.json()["hook"]

    class PermanentSMTP:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def starttls(self):
            pass

        def login(self, username, password):
            pass

        def send_message(self, message):
            raise smtplib.SMTPResponseException(550, b"mailbox unavailable with smtp-password")

    monkeypatch.setattr("cowork.services.notifications.smtplib.SMTP", PermanentSMTP)

    delivery_response = client.post(f"/api/v1/projects/{project_id}/notification-hooks/{hook['id']}/test")
    assert delivery_response.status_code == 200, delivery_response.text
    delivery = delivery_response.json()["delivery"]
    assert delivery["status"] == "failed"
    assert delivery["error"] == "smtp_550"
    assert delivery["details"]["retryable"] is False
    assert delivery["details"]["smtpCode"] == 550
    assert "smtp-password" not in json.dumps(delivery)


def test_server_lifespan_starts_and_stops_notification_dispatcher(monkeypatch: pytest.MonkeyPatch):
    import cowork.channels.registry as channel_registry
    import cowork.channels.webhooks as channel_webhooks
    import cowork.server as server
    import cowork.services.notifications as notifications

    calls: list[str] = []

    class FakeAdapters:
        async def refresh_all(self):
            calls.append("channels_refreshed")

        async def shutdown(self):
            calls.append("channels_shutdown")

    class FakeIngress:
        async def stop_all(self):
            calls.append("ingress_stopped")

    class FakeRegistry:
        def all(self):
            return []

    def fake_install_channels(app):
        app.state.channel_adapters = FakeAdapters()
        app.state.channel_ingress = FakeIngress()

    def fake_start_dispatcher():
        calls.append("notifications_started")
        return object()

    async def fake_drain_background_tasks():
        calls.append("drained")

    async def fake_stop_dispatcher():
        calls.append("notifications_stopped")

    monkeypatch.setattr(server, "_install_channels", fake_install_channels)
    monkeypatch.setattr(server, "run_dev_setup", lambda: calls.append("dev_setup"))
    monkeypatch.setattr(server, "start_scheduler", lambda: calls.append("scheduler_started"))
    monkeypatch.setattr(channel_registry, "get_registry", lambda: FakeRegistry())
    monkeypatch.setattr(channel_webhooks, "drain_background_tasks", fake_drain_background_tasks)
    monkeypatch.setattr(notifications, "start_notification_dispatcher", fake_start_dispatcher)
    monkeypatch.setattr(notifications, "stop_notification_dispatcher", fake_stop_dispatcher)

    app = server.create_app()
    with TestClient(app) as active_client:
        response = active_client.get("/api/v1/health/")
        assert response.status_code in {200, 404}

    assert "notifications_started" in calls
    assert "notifications_stopped" in calls
    assert calls.index("notifications_started") < calls.index("notifications_stopped")
