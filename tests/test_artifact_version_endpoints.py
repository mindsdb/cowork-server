from __future__ import annotations

import asyncio
import json
import shutil
import sys
import types
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from sqlmodel import Session, select

from cowork.api.v1.router import api_router
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.artifact import Artifact, ArtifactActivityEvent, ArtifactDeployment, ArtifactVersion
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.models.project_collaboration import ProjectCollaborator
from cowork.services.artifact_versions import ArtifactVersionService
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.request_identity import RequestPrincipal
from cowork.services.screenshot_diff import ScreenshotDiffUnavailable


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(api_router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_version_test_artifacts():
    project = _general_project_path()
    roots = [
        project / ".anton" / "artifacts",
        project / ".anton" / "artifact_versions",
    ]
    for root in roots:
        if root.is_dir():
            for folder in root.glob("version-test-*"):
                shutil.rmtree(folder, ignore_errors=True)
    _clear_project_collaborators()
    yield
    for root in roots:
        if root.is_dir():
            for folder in root.glob("version-test-*"):
                shutil.rmtree(folder, ignore_errors=True)
    _clear_project_collaborators()


def _clear_project_collaborators() -> None:
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        for row in session.exec(select(ProjectCollaborator)).all():
            session.delete(row)
        session.commit()


def _general_project_path() -> Path:
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        project = session.get(Project, GENERAL_PROJECT_ID)
        assert project is not None
        return Path(project.path)


def _make_artifact(*, files: dict[str, str], artifact_type: str = "document") -> tuple[str, Path]:
    project = _general_project_path()
    slug = f"version-test-{uuid4().hex}"
    artifact_id = f"artifact-{uuid4().hex}"
    folder = project / ".anton" / "artifacts" / slug
    sidecar = project / ".anton" / "artifact_versions" / slug
    if folder.exists():
        shutil.rmtree(folder)
    if sidecar.exists():
        shutil.rmtree(sidecar)
    folder.mkdir(parents=True)
    primary = next(iter(files), "")
    metadata = {
        "id": artifact_id,
        "slug": slug,
        "name": "Quarterly Research",
        "description": "Analyst-ready artifact",
        "type": artifact_type,
        "primary": primary,
    }
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for rel_path, content in files.items():
        target = folder / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return artifact_id, folder


def _checkpoint(client: TestClient, artifact_id: str, **body) -> dict:
    response = client.post(f"/api/v1/artifacts/{artifact_id}/checkpoints", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def _own_general_project_for_versions() -> None:
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        session.add(ProjectCollaborator(project_id=GENERAL_PROJECT_ID, email="owner@example.com", role="owner"))
        session.add(ProjectCollaborator(project_id=GENERAL_PROJECT_ID, email="viewer@example.com", role="viewer"))
        session.add(ProjectCollaborator(project_id=GENERAL_PROJECT_ID, email="editor@example.com", role="editor"))
        session.commit()


def _make_owned_project(tmp_path: Path, *, roles: dict[str, str]) -> tuple[UUID, Path]:
    engine = get_engine(get_app_settings().database.uri)
    name = f"version-test-project-{uuid4().hex}"
    path = tmp_path / name
    path.mkdir(parents=True)
    with Session(engine) as session:
        project = Project(name=name, path=str(path))
        session.add(project)
        session.commit()
        session.refresh(project)
        session.add(ProjectCollaborator(project_id=project.id, email="owner@example.com", role="owner"))
        for email, role in roles.items():
            session.add(ProjectCollaborator(project_id=project.id, email=email, role=role))
        session.commit()
        return project.id, path


def _make_artifact_in_project(project_path: Path, *, files: dict[str, str], artifact_type: str = "document") -> tuple[str, Path]:
    slug = f"version-test-{uuid4().hex}"
    artifact_id = f"artifact-{uuid4().hex}"
    folder = project_path / ".anton" / "artifacts" / slug
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    primary = next(iter(files), "")
    (folder / "metadata.json").write_text(
        json.dumps(
            {
                "id": artifact_id,
                "slug": slug,
                "name": "Project Artifact",
                "description": "Project-scoped artifact",
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


def _fake_version_principal(authorization: str | None):
    if not authorization:
        return None
    token = authorization.removeprefix("Bearer ").strip()
    if token == "viewer":
        return RequestPrincipal("viewer", "viewer@example.com", "Viewer", "test", {})
    if token == "editor":
        return RequestPrincipal("editor", "editor@example.com", "Editor", "test", {})
    if token == "owner":
        return RequestPrincipal("owner", "owner@example.com", "Owner", "test", {})
    return None


def test_checkpoint_creation_and_listing(client: TestClient):
    artifact_id, folder = _make_artifact(files={"report.md": "# Draft\n"})

    first = _checkpoint(client, artifact_id, label="Analyst draft", prompt="Summarize the draft for analysts")
    first_version = first["version"]
    assert first_version["humanLabel"] == "Analyst draft"
    assert first_version["prompt"] == "Summarize the draft for analysts"
    assert first_version["fileCount"] == 1
    assert first_version["files"][0]["path"] == "report.md"
    assert first["artifact"]["title"] == "Quarterly Research"
    assert first["preview"]["serveUrl"].startswith("/api/v1/artifacts/serve/")
    assert first["publish"]["accessMode"] == "public"

    (folder / "report.md").write_text("# Final\n", encoding="utf-8")
    second = _checkpoint(client, artifact_id, kind="auto")

    response = client.get(f"/api/v1/artifacts/{artifact_id}/versions")
    assert response.status_code == 200, response.text
    listed = response.json()
    assert listed["artifact"]["id"] == artifact_id
    assert [version["id"] for version in listed["versions"]] == [
        second["version"]["id"],
        first_version["id"],
    ]
    assert listed["versions"][1]["prompt"] == "Summarize the draft for analysts"
    assert "sourceConversationId" in listed["versions"][1]
    assert "sourceMessageId" in listed["versions"][1]
    assert listed["latest"]["id"] == second["version"]["id"]


def test_owned_project_id_version_routes_respect_collaborator_roles(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    _own_general_project_for_versions()
    artifact_id, folder = _make_artifact(files={"report.md": "# Draft\n"})
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifact_versions.principal_from_authorization_header",
        _fake_version_principal,
    )
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifacts.principal_from_authorization_header",
        _fake_version_principal,
    )
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifacts.principal_from_authorization_header",
        _fake_version_principal,
    )
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifacts.principal_from_authorization_header",
        _fake_version_principal,
    )

    anonymous_list = client.get(f"/api/v1/artifacts/{artifact_id}/versions")
    assert anonymous_list.status_code == 401

    viewer_checkpoint = client.post(
        f"/api/v1/artifacts/{artifact_id}/checkpoints",
        headers={"Authorization": "Bearer viewer"},
        json={"label": "Viewer edit attempt"},
    )
    assert viewer_checkpoint.status_code == 403

    editor_checkpoint = client.post(
        f"/api/v1/artifacts/{artifact_id}/checkpoints",
        headers={"Authorization": "Bearer editor"},
        json={"label": "Editor checkpoint"},
    )
    assert editor_checkpoint.status_code == 201, editor_checkpoint.text
    version_id = editor_checkpoint.json()["version"]["id"]

    viewer_list = client.get(
        f"/api/v1/artifacts/{artifact_id}/versions",
        headers={"Authorization": "Bearer viewer"},
    )
    assert viewer_list.status_code == 200, viewer_list.text
    assert viewer_list.json()["versions"][0]["id"] == version_id

    viewer_diff = client.get(
        f"/api/v1/artifacts/{artifact_id}/diff",
        headers={"Authorization": "Bearer viewer"},
        params={"base": version_id, "compare": "current"},
    )
    assert viewer_diff.status_code == 200, viewer_diff.text

    (folder / "report.md").write_text("# Updated\n", encoding="utf-8")
    viewer_restore = client.post(
        f"/api/v1/artifacts/{artifact_id}/restore",
        headers={"Authorization": "Bearer viewer"},
        json={"versionId": version_id},
    )
    assert viewer_restore.status_code == 403

    editor_restore = client.post(
        f"/api/v1/artifacts/{artifact_id}/restore",
        headers={"Authorization": "Bearer editor"},
        json={"versionId": version_id},
    )
    assert editor_restore.status_code == 200, editor_restore.text
    assert (folder / "report.md").read_text(encoding="utf-8") == "# Draft\n"


def test_path_version_history_is_visible_to_viewers(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    _own_general_project_for_versions()
    _artifact_id, folder = _make_artifact(files={"report.md": "# Draft\n"})
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifacts.principal_from_authorization_header",
        _fake_version_principal,
    )

    checkpoint = client.post(
        "/api/v1/artifacts/versions",
        headers={"Authorization": "Bearer owner"},
        json={"path": str(folder), "label": "Owner checkpoint"},
    )
    assert checkpoint.status_code == 200, checkpoint.text

    anonymous_history = client.get("/api/v1/artifacts/versions", params={"path": str(folder)})
    assert anonymous_history.status_code == 401

    viewer_history = client.get(
        "/api/v1/artifacts/versions",
        headers={"Authorization": "Bearer viewer"},
        params={"path": str(folder)},
    )
    assert viewer_history.status_code == 200, viewer_history.text
    assert viewer_history.json()["versions"][0]["label"] == "Owner checkpoint"


def test_project_activity_feed_is_visible_to_project_viewers(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    project_id, project_path = _make_owned_project(tmp_path, roles={"viewer@example.com": "viewer"})
    artifact_id, folder = _make_artifact_in_project(project_path, files={"report.md": "# Draft\n"})
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.projects.principal_from_authorization_header",
        _fake_version_principal,
    )
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        first = ArtifactVersionService(session).snapshot_artifact(
            folder,
            operation_type="checkpoint",
            label="First activity",
        )
        first_version_id = str(first.id)
        (folder / "report.md").write_text("# Updated\n", encoding="utf-8")
        second = ArtifactVersionService(session).snapshot_artifact(
            folder,
            operation_type="checkpoint",
            label="Second activity",
        )
        second_version_id = str(second.id)

    anonymous = client.get(f"/api/v1/projects/{project_id}/activity")
    assert anonymous.status_code == 401

    response = client.get(
        f"/api/v1/projects/{project_id}/activity",
        headers={"Authorization": "Bearer viewer"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    version_ids = [item["versionId"] for item in payload["activity"]]
    assert second_version_id in version_ids
    assert first_version_id in version_ids
    assert payload["activity"][0]["createdAt"] >= payload["activity"][-1]["createdAt"]
    assert any(item["externalArtifactId"] == artifact_id for item in payload["activity"])


def test_global_activity_feed_filters_projects_without_view_access(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    visible_project_id, visible_project_path = _make_owned_project(
        tmp_path / "visible",
        roles={"viewer@example.com": "viewer"},
    )
    visible_artifact_id, visible_folder = _make_artifact_in_project(
        visible_project_path,
        files={"report.md": "# Visible\n"},
    )
    private_project_id, private_project_path = _make_owned_project(tmp_path / "private", roles={})
    private_artifact_id, private_folder = _make_artifact_in_project(
        private_project_path,
        files={"report.md": "# Private\n"},
    )
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.activity.principal_from_authorization_header",
        _fake_version_principal,
    )

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        visible_version = ArtifactVersionService(session).snapshot_artifact(
            visible_folder,
            operation_type="checkpoint",
            label="Visible activity",
        )
        visible_version_id = str(visible_version.id)
        private_version = ArtifactVersionService(session).snapshot_artifact(
            private_folder,
            operation_type="checkpoint",
            label="Private activity",
        )
        private_version_id = str(private_version.id)

    response = client.get(
        "/api/v1/activity/",
        headers={"Authorization": "Bearer viewer"},
        params={"limit": 100},
    )
    assert response.status_code == 200, response.text
    activity = response.json()["activity"]
    version_ids = {item["versionId"] for item in activity}
    external_ids = {item["externalArtifactId"] for item in activity}

    assert visible_version_id in version_ids
    assert visible_artifact_id in external_ids
    assert private_version_id not in version_ids
    assert private_artifact_id not in external_ids
    assert str(private_project_id) not in {item["projectId"] for item in activity}
    assert str(visible_project_id) in {item["projectId"] for item in activity}


def test_owned_project_fork_requires_editor_when_copying_into_same_project(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    _own_general_project_for_versions()
    artifact_id, folder = _make_artifact(files={"report.md": "old\n"})
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifact_versions.principal_from_authorization_header",
        _fake_version_principal,
    )
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifacts.principal_from_authorization_header",
        _fake_version_principal,
    )
    checkpoint = client.post(
        f"/api/v1/artifacts/{artifact_id}/checkpoints",
        headers={"Authorization": "Bearer editor"},
        json={"label": "Fork source"},
    )
    assert checkpoint.status_code == 201, checkpoint.text
    version_id = checkpoint.json()["version"]["id"]

    viewer_fork = client.post(
        f"/api/v1/artifacts/{artifact_id}/fork",
        headers={"Authorization": "Bearer viewer"},
        json={"path": str(folder), "versionId": version_id, "title": "Viewer copy"},
    )
    assert viewer_fork.status_code == 403

    viewer_path_fork = client.post(
        "/api/v1/artifacts/versions/fork",
        headers={"Authorization": "Bearer viewer"},
        json={"path": str(folder), "versionId": version_id, "name": "Viewer path copy"},
    )
    assert viewer_path_fork.status_code == 403

    editor_fork = client.post(
        f"/api/v1/artifacts/{artifact_id}/fork",
        headers={"Authorization": "Bearer editor"},
        json={
            "path": str(folder),
            "versionId": version_id,
            "title": "Editor copy",
            "slug": "version-test-editor-copy",
        },
    )
    assert editor_fork.status_code == 201, editor_fork.text
    fork_folder = Path(editor_fork.json()["artifactPath"])
    assert fork_folder.name.startswith("version-test-editor-copy")
    assert (fork_folder / "report.md").read_text(encoding="utf-8") == "old\n"


def test_viewer_can_fork_version_into_project_they_can_edit(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _own_general_project_for_versions()
    target_project_id, target_path = _make_owned_project(
        tmp_path,
        roles={"viewer@example.com": "editor"},
    )
    artifact_id, folder = _make_artifact(files={"report.md": "source checkpoint\n"})
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifact_versions.principal_from_authorization_header",
        _fake_version_principal,
    )
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifacts.principal_from_authorization_header",
        _fake_version_principal,
    )
    checkpoint = client.post(
        f"/api/v1/artifacts/{artifact_id}/checkpoints",
        headers={"Authorization": "Bearer editor"},
        json={"label": "Copy source"},
    )
    assert checkpoint.status_code == 201, checkpoint.text
    version_id = checkpoint.json()["version"]["id"]
    (folder / "report.md").write_text("live draft\n", encoding="utf-8")

    response = client.post(
        f"/api/v1/artifacts/{artifact_id}/fork",
        headers={"Authorization": "Bearer viewer"},
        json={
            "path": str(folder),
            "versionId": version_id,
            "targetProjectId": str(target_project_id),
            "title": "Viewer cross-project copy",
            "slug": "version-test-viewer-copy",
        },
    )
    assert response.status_code == 201, response.text
    fork_folder = Path(response.json()["artifactPath"])
    assert fork_folder.parent == target_path / ".anton" / "artifacts"
    assert (fork_folder / "report.md").read_text(encoding="utf-8") == "source checkpoint\n"
    assert response.json()["version"]["forkedFromVersionId"] == version_id

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        artifact = session.exec(select(Artifact).where(Artifact.path == str(fork_folder.resolve()))).one()
        assert artifact.project_id == target_project_id


def test_viewer_cannot_fork_version_into_project_without_edit(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _own_general_project_for_versions()
    target_project_id, target_path = _make_owned_project(
        tmp_path,
        roles={"viewer@example.com": "viewer"},
    )
    artifact_id, folder = _make_artifact(files={"report.md": "source checkpoint\n"})
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifact_versions.principal_from_authorization_header",
        _fake_version_principal,
    )
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifacts.principal_from_authorization_header",
        _fake_version_principal,
    )
    checkpoint = client.post(
        f"/api/v1/artifacts/{artifact_id}/checkpoints",
        headers={"Authorization": "Bearer editor"},
        json={"label": "Copy source"},
    )
    assert checkpoint.status_code == 201, checkpoint.text
    version_id = checkpoint.json()["version"]["id"]

    response = client.post(
        "/api/v1/artifacts/versions/fork",
        headers={"Authorization": "Bearer viewer"},
        json={
            "path": str(folder),
            "versionId": version_id,
            "projectId": str(target_project_id),
            "name": "Denied copy",
            "slug": "version-test-denied-copy",
        },
    )
    assert response.status_code == 403
    assert not (target_path / ".anton" / "artifacts" / "version-test-denied-copy").exists()


def test_viewers_do_not_receive_owner_side_publish_access_state(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    _own_general_project_for_versions()
    artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Protected</h1>\n"},
        artifact_type="html-app",
    )
    (folder / ".published.json").write_text(
        json.dumps(
            {
                "index.html": {
                    "url": "https://4nton.ai/p/protected",
                    "mode": "password",
                    "requires_password": True,
                    "access_password": "secret-password",
                    "pwd_version": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifacts.principal_from_authorization_header",
        _fake_version_principal,
    )
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifact_versions.principal_from_authorization_header",
        _fake_version_principal,
    )

    viewer_list = client.get("/api/v1/artifacts/", headers={"Authorization": "Bearer viewer"})
    assert viewer_list.status_code == 200, viewer_list.text
    viewer_card = next(item for item in viewer_list.json() if item["id"] == artifact_id)
    assert viewer_card["accessMode"] == "password"
    assert viewer_card["accessProtected"] is True
    assert viewer_card["accessPassword"] == ""

    editor_list = client.get("/api/v1/artifacts/", headers={"Authorization": "Bearer editor"})
    assert editor_list.status_code == 200, editor_list.text
    editor_card = next(item for item in editor_list.json() if item["id"] == artifact_id)
    assert editor_card["accessPassword"] == "secret-password"

    viewer_mount = client.post(
        "/api/v1/artifacts/preview-mount",
        headers={"Authorization": "Bearer viewer"},
        json={"path": str(folder / "index.html")},
    )
    assert viewer_mount.status_code == 200, viewer_mount.text
    assert viewer_mount.json()["accessPassword"] == ""

    checkpoint = client.post(
        f"/api/v1/artifacts/{artifact_id}/checkpoints",
        headers={"Authorization": "Bearer editor"},
        json={"label": "Protected checkpoint"},
    )
    assert checkpoint.status_code == 201, checkpoint.text

    viewer_versions = client.get(
        f"/api/v1/artifacts/{artifact_id}/versions",
        headers={"Authorization": "Bearer viewer"},
    )
    assert viewer_versions.status_code == 200, viewer_versions.text
    assert viewer_versions.json()["publish"]["accessPassword"] == ""

    editor_versions = client.get(
        f"/api/v1/artifacts/{artifact_id}/versions",
        headers={"Authorization": "Bearer editor"},
    )
    assert editor_versions.status_code == 200, editor_versions.text
    assert editor_versions.json()["publish"]["accessPassword"] == "secret-password"


def test_viewer_diff_without_checkpoint_does_not_create_version_state(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    _own_general_project_for_versions()
    _artifact_id, folder = _make_artifact(files={"report.md": "# Draft\n"})
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifacts.principal_from_authorization_header",
        _fake_version_principal,
    )

    response = client.get(
        "/api/v1/artifacts/diff",
        headers={"Authorization": "Bearer viewer"},
        params={"path": str(folder), "to": "current"},
    )
    assert response.status_code == 400, response.text

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        artifact = session.exec(select(Artifact).where(Artifact.path == str(folder.resolve()))).first()
        versions = [] if artifact is None else session.exec(
            select(ArtifactVersion).where(ArtifactVersion.artifact_id == artifact.id)
        ).all()
    assert versions == []


def test_restore_checkpoint_restores_artifact_files(client: TestClient):
    artifact_id, folder = _make_artifact(files={"report.md": "old\n"})
    first = _checkpoint(client, artifact_id, label="Before revision")

    (folder / "report.md").write_text("new\n", encoding="utf-8")
    _checkpoint(client, artifact_id, label="After revision")
    (folder / "report.md").write_text("scratch\n", encoding="utf-8")

    response = client.post(
        f"/api/v1/artifacts/{artifact_id}/restore",
        json={"versionId": first["version"]["id"]},
    )
    assert response.status_code == 200, response.text
    restored = response.json()
    assert restored["status"] == "ok"
    assert restored["restoredVersion"]["id"] == first["version"]["id"]
    assert restored["version"]["id"] != first["version"]["id"]
    assert restored["version"]["restoredFromVersionId"] == first["version"]["id"]
    assert (folder / "report.md").read_text(encoding="utf-8") == "old\n"


def test_restore_can_roll_forward_to_later_version(client: TestClient):
    artifact_id, folder = _make_artifact(files={"report.md": "old\n"})
    first = _checkpoint(client, artifact_id, label="Version one")

    (folder / "report.md").write_text("new\n", encoding="utf-8")
    second = _checkpoint(client, artifact_id, label="Version two")
    (folder / "report.md").write_text("scratch\n", encoding="utf-8")

    rollback = client.post(
        f"/api/v1/artifacts/{artifact_id}/restore",
        json={"versionId": first["version"]["id"]},
    )
    assert rollback.status_code == 200, rollback.text
    assert (folder / "report.md").read_text(encoding="utf-8") == "old\n"

    roll_forward = client.post(
        f"/api/v1/artifacts/{artifact_id}/restore",
        json={"versionId": second["version"]["id"]},
    )
    assert roll_forward.status_code == 200, roll_forward.text
    payload = roll_forward.json()
    assert payload["restoredVersion"]["id"] == second["version"]["id"]
    assert payload["version"]["id"] not in {first["version"]["id"], second["version"]["id"]}
    assert payload["version"]["restoredFromVersionId"] == second["version"]["id"]
    assert (folder / "report.md").read_text(encoding="utf-8") == "new\n"

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        artifact = session.exec(select(Artifact).where(Artifact.path == str(folder.resolve()))).one()
        restore_versions = session.exec(
            select(ArtifactVersion)
            .where(ArtifactVersion.artifact_id == artifact.id)
            .where(ArtifactVersion.operation_type == "restore")
        ).all()
    assert {str(version.restored_from_version_id) for version in restore_versions} >= {
        first["version"]["id"],
        second["version"]["id"],
    }


def test_restore_with_create_checkpoint_preserves_pre_restore_draft(client: TestClient):
    artifact_id, folder = _make_artifact(files={"report.md": "old\n"})
    first = _checkpoint(client, artifact_id, label="Before revision")

    (folder / "report.md").write_text("new\n", encoding="utf-8")
    _checkpoint(client, artifact_id, label="After revision")
    (folder / "report.md").write_text("scratch\n", encoding="utf-8")

    response = client.post(
        f"/api/v1/artifacts/{artifact_id}/restore",
        json={"versionId": first["version"]["id"], "createCheckpoint": True},
    )
    assert response.status_code == 200, response.text
    restored = response.json()
    assert restored["createdCheckpoint"]["operationType"] == "restore_safety"
    assert restored["createdCheckpoint"]["id"] != restored["version"]["id"]
    assert restored["version"]["restoredFromVersionId"] == first["version"]["id"]
    assert (folder / "report.md").read_text(encoding="utf-8") == "old\n"

    diff_response = client.get(
        f"/api/v1/artifacts/{artifact_id}/diff",
        params={
            "base": restored["createdCheckpoint"]["id"],
            "compare": restored["version"]["id"],
            "kind": "text",
        },
    )
    assert diff_response.status_code == 200, diff_response.text
    assert "-scratch" in diff_response.json()["textDiff"]
    assert "+old" in diff_response.json()["textDiff"]


def test_direct_artifact_delete_service_snapshots_before_removal(client: TestClient):
    from cowork.services.artifacts import delete_artifact

    _artifact_id, folder = _make_artifact(files={"report.md": "# Delete me\n"})

    delete_artifact(str(folder))

    assert not folder.exists()
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        artifact = session.exec(select(Artifact).where(Artifact.path == str(folder.resolve()))).one()
        versions = session.exec(
            select(ArtifactVersion)
            .where(ArtifactVersion.artifact_id == artifact.id)
            .order_by(ArtifactVersion.version_number)
        ).all()
    assert [version.operation_type for version in versions] == ["pre_delete"]


def test_path_restore_with_create_checkpoint_preserves_pre_restore_draft(client: TestClient):
    artifact_id, folder = _make_artifact(files={"report.md": "old\n"})
    first = _checkpoint(client, artifact_id, label="Before revision")
    (folder / "report.md").write_text("new\n", encoding="utf-8")
    _checkpoint(client, artifact_id, label="After revision")
    (folder / "report.md").write_text("scratch\n", encoding="utf-8")

    response = client.post(
        "/api/v1/artifacts/versions/restore",
        json={"path": str(folder), "version_id": first["version"]["id"], "createCheckpoint": True},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["createdCheckpoint"]["operationType"] == "restore_safety"
    assert payload["version"]["restoredFromVersionId"] == first["version"]["id"]
    assert (folder / "report.md").read_text(encoding="utf-8") == "old\n"

    listed = client.get("/api/v1/artifacts/versions", params={"path": str(folder)})
    assert listed.status_code == 200, listed.text
    operations = [version["operationType"] for version in listed.json()["versions"]]
    assert operations[:2] == ["restore", "restore_safety"]


def test_deleted_artifact_can_be_restored_from_pre_delete_version(client: TestClient):
    artifact_id, folder = _make_artifact(
        files={
            "report.md": "recover me\n",
            "data/summary.txt": "nested content\n",
        }
    )

    delete_response = client.delete("/api/v1/artifacts/", params={"path": str(folder)})
    assert delete_response.status_code == 204, delete_response.text
    assert not folder.exists()

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        artifact = session.exec(select(Artifact).where(Artifact.path == str(folder.resolve()))).one()
        versions = session.exec(
            select(ArtifactVersion)
            .where(ArtifactVersion.artifact_id == artifact.id)
            .order_by(ArtifactVersion.version_number)
        ).all()
        deleted_event = session.exec(
            select(ArtifactActivityEvent)
            .where(ArtifactActivityEvent.artifact_id == artifact.id)
            .where(ArtifactActivityEvent.event_type == "deleted")
        ).one()

    assert [version.operation_type for version in versions] == ["pre_delete"]
    pre_delete_id = str(versions[0].id)
    assert deleted_event.details["externalArtifactId"] == artifact_id
    assert deleted_event.details["preDeleteVersionId"] == pre_delete_id

    deleted_versions = client.get(f"/api/v1/artifacts/{artifact_id}/versions")
    assert deleted_versions.status_code == 200, deleted_versions.text
    assert deleted_versions.json()["versions"][0]["operationType"] == "pre_delete"
    deleted_list = client.get("/api/v1/artifacts/deleted")
    assert deleted_list.status_code == 200, deleted_list.text
    tombstone = deleted_list.json()["artifacts"][0]
    assert tombstone["artifactId"] == artifact_id
    assert tombstone["preDeleteVersionId"] == pre_delete_id
    assert tombstone["restoreEligible"] is True
    assert tombstone["fileCount"] >= 2

    restore_response = client.post(
        f"/api/v1/artifacts/{artifact_id}/restore",
        json={"versionId": pre_delete_id},
    )
    assert restore_response.status_code == 200, restore_response.text
    restored = restore_response.json()
    assert restored["version"]["operationType"] == "restore_deleted"
    assert restored["version"]["restoredFromVersionId"] == pre_delete_id
    assert (folder / "report.md").read_text(encoding="utf-8") == "recover me\n"
    assert (folder / "data" / "summary.txt").read_text(encoding="utf-8") == "nested content\n"
    assert json.loads((folder / "metadata.json").read_text(encoding="utf-8"))["id"] == artifact_id

    listed = client.get("/api/v1/artifacts/", params={"project_path": str(_general_project_path())})
    assert listed.status_code == 200, listed.text
    assert artifact_id in {item["id"] for item in listed.json()}
    deleted_after_restore = client.get("/api/v1/artifacts/deleted")
    assert deleted_after_restore.status_code == 200, deleted_after_restore.text
    assert artifact_id not in {item["artifactId"] for item in deleted_after_restore.json()["artifacts"]}


def test_delete_restores_folder_when_tombstone_recording_fails(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    from cowork.api.v1.endpoints import artifacts as artifacts_endpoint

    _artifact_id, folder = _make_artifact(
        files={
            "report.md": "keep me\n",
            "data/summary.txt": "nested content\n",
        }
    )

    def fail_notification(*args, **kwargs):
        raise RuntimeError("notification state failed")

    monkeypatch.setattr(artifacts_endpoint, "dispatch_project_notification", fail_notification)

    delete_response = client.delete("/api/v1/artifacts/", params={"path": str(folder)})
    assert delete_response.status_code == 500
    assert folder.exists()
    assert (folder / "report.md").read_text(encoding="utf-8") == "keep me\n"
    assert (folder / "data" / "summary.txt").read_text(encoding="utf-8") == "nested content\n"

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        artifact = session.exec(select(Artifact).where(Artifact.path == str(folder.resolve()))).one()
        deleted_events = session.exec(
            select(ArtifactActivityEvent)
            .where(ArtifactActivityEvent.artifact_id == artifact.id)
            .where(ArtifactActivityEvent.event_type == "deleted")
        ).all()
        assert deleted_events == []


def test_delete_rejects_existing_folder_outside_project_artifacts(
    client: TestClient,
    tmp_path: Path,
):
    folder = tmp_path / "not-a-project-artifact"
    folder.mkdir()
    (folder / "metadata.json").write_text(
        json.dumps({"id": "outside-delete", "slug": "outside-delete", "name": "Outside"}),
        encoding="utf-8",
    )
    (folder / "report.md").write_text("do not delete\n", encoding="utf-8")

    delete_response = client.delete("/api/v1/artifacts/", params={"path": str(folder)})

    assert delete_response.status_code == 400
    assert "project artifacts folder" in delete_response.json()["detail"]
    assert folder.exists()
    assert (folder / "report.md").read_text(encoding="utf-8") == "do not delete\n"


def test_deleted_artifact_restore_respects_project_permissions(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    _own_general_project_for_versions()
    artifact_id, folder = _make_artifact(files={"report.md": "owned\n"})
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifacts.principal_from_authorization_header",
        _fake_version_principal,
    )
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifact_versions.principal_from_authorization_header",
        _fake_version_principal,
    )

    delete_response = client.delete(
        "/api/v1/artifacts/",
        headers={"Authorization": "Bearer owner"},
        params={"path": str(folder)},
    )
    assert delete_response.status_code == 204, delete_response.text
    assert not folder.exists()

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        artifact = session.exec(select(Artifact).where(Artifact.path == str(folder.resolve()))).one()
        pre_delete = session.exec(
            select(ArtifactVersion)
            .where(ArtifactVersion.artifact_id == artifact.id)
            .where(ArtifactVersion.operation_type == "pre_delete")
        ).one()

    viewer_restore = client.post(
        f"/api/v1/artifacts/{artifact_id}/restore",
        headers={"Authorization": "Bearer viewer"},
        json={"versionId": str(pre_delete.id)},
    )
    assert viewer_restore.status_code == 403
    assert not folder.exists()

    viewer_deleted = client.get("/api/v1/artifacts/deleted", headers={"Authorization": "Bearer viewer"})
    assert viewer_deleted.status_code == 200, viewer_deleted.text
    assert artifact_id in {item["artifactId"] for item in viewer_deleted.json()["artifacts"]}

    editor_restore = client.post(
        f"/api/v1/artifacts/{artifact_id}/restore",
        headers={"Authorization": "Bearer editor"},
        json={"versionId": str(pre_delete.id)},
    )
    assert editor_restore.status_code == 200, editor_restore.text
    assert (folder / "report.md").read_text(encoding="utf-8") == "owned\n"


def test_diff_versions_reports_changed_files_with_human_labels(client: TestClient):
    artifact_id, folder = _make_artifact(
        files={
            "report.md": "# Old\n",
            "data.csv": "region,total\nwest,1\n",
        }
    )
    base = _checkpoint(client, artifact_id, label="Baseline")

    (folder / "report.md").write_text("# New\n", encoding="utf-8")
    (folder / "data.csv").unlink()
    (folder / "notes.txt").write_text("follow up\n", encoding="utf-8")
    compare = _checkpoint(client, artifact_id, label="Updated")

    response = client.get(
        f"/api/v1/artifacts/{artifact_id}/diff",
        params={"base": base["version"]["id"], "compare": compare["version"]["id"], "kind": "text"},
    )
    assert response.status_code == 200, response.text
    diff = response.json()
    assert diff["summary"] == {"added": 1, "modified": 1, "removed": 1, "unchanged": 0, "totalChanged": 3}

    changes = {change["path"]: change for change in diff["changedFiles"]}
    assert changes["report.md"]["status"] == "modified"
    assert changes["report.md"]["humanLabel"] == "Updated report.md"
    assert "-# Old" in changes["report.md"]["textDiff"]
    assert "+# New" in changes["report.md"]["textDiff"]
    assert changes["data.csv"]["status"] == "removed"
    assert changes["data.csv"]["humanLabel"] == "Removed data.csv"
    assert changes["notes.txt"]["status"] == "added"
    assert changes["notes.txt"]["humanLabel"] == "Added notes.txt"


def test_diff_against_current_draft_uses_live_files_without_checkpointing(client: TestClient):
    artifact_id, folder = _make_artifact(files={"report.md": "# Saved\n"})
    base = _checkpoint(client, artifact_id, label="Saved")
    (folder / "report.md").write_text("# Unsaved draft\n", encoding="utf-8")

    response = client.get(
        f"/api/v1/artifacts/{artifact_id}/diff",
        params={"base": base["version"]["id"], "compare": "current", "kind": "text"},
    )
    assert response.status_code == 200, response.text
    diff = response.json()
    assert diff["compare"]["id"] == "current"
    assert diff["compare"]["humanLabel"] == "Current draft"
    assert diff["summary"]["modified"] == 1
    assert "-# Saved" in diff["textDiff"]
    assert "+# Unsaved draft" in diff["textDiff"]

    listed = client.get(f"/api/v1/artifacts/{artifact_id}/versions")
    assert listed.status_code == 200, listed.text
    assert [version["id"] for version in listed.json()["versions"]] == [base["version"]["id"]]


def test_path_diff_current_draft_uses_live_files(client: TestClient):
    artifact_id, folder = _make_artifact(files={"report.md": "# Saved\n"})
    base = _checkpoint(client, artifact_id, label="Saved")
    (folder / "report.md").write_text("# Unsaved draft\n", encoding="utf-8")

    response = client.get(
        "/api/v1/artifacts/diff",
        params={"path": str(folder), "from": base["version"]["id"], "to": "current"},
    )
    assert response.status_code == 200, response.text
    diff = response.json()
    assert diff["compare"]["id"] == "current"
    assert diff["summary"]["modified"] == 1
    assert "+# Unsaved draft" in diff["textDiff"]


def test_id_version_routes_map_bad_version_refs_to_client_errors(client: TestClient):
    artifact_id, _folder = _make_artifact(files={"report.md": "# Saved\n"})
    base = _checkpoint(client, artifact_id, label="Saved")
    missing = str(uuid4())

    bad_restore = client.post(
        f"/api/v1/artifacts/{artifact_id}/restore",
        json={"versionId": "not-a-version-id"},
    )
    assert bad_restore.status_code == 400, bad_restore.text

    missing_restore = client.post(
        f"/api/v1/artifacts/{artifact_id}/restore",
        json={"versionId": missing},
    )
    assert missing_restore.status_code == 404, missing_restore.text

    bad_fork = client.post(
        f"/api/v1/artifacts/{artifact_id}/fork",
        json={"versionId": "not-a-version-id"},
    )
    assert bad_fork.status_code == 400, bad_fork.text

    bad_diff = client.get(
        f"/api/v1/artifacts/{artifact_id}/diff",
        params={"base": "not-a-version-id", "compare": base["version"]["id"]},
    )
    assert bad_diff.status_code == 404, bad_diff.text


def test_restore_and_fork_reject_checkpoint_from_another_artifact(client: TestClient):
    first_id, _first_folder = _make_artifact(files={"report.md": "first\n"})
    second_id, second_folder = _make_artifact(files={"report.md": "second\n"})
    first_checkpoint = _checkpoint(client, first_id, label="First artifact")

    restore = client.post(
        f"/api/v1/artifacts/{second_id}/restore",
        json={"versionId": first_checkpoint["version"]["id"]},
    )
    assert restore.status_code == 404, restore.text
    assert (second_folder / "report.md").read_text(encoding="utf-8") == "second\n"

    fork = client.post(
        f"/api/v1/artifacts/{second_id}/fork",
        json={
            "path": str(second_folder),
            "versionId": first_checkpoint["version"]["id"],
            "title": "Wrong source",
        },
    )
    assert fork.status_code == 404, fork.text


def test_html_diff_exposes_side_by_side_visual_previews(client: TestClient):
    artifact_id, folder = _make_artifact(
        files={
            "index.html": "<h1>Before</h1><script src=\"assets/app.js\"></script>\n",
            "assets/app.js": "window.version = 'before';\n",
        },
        artifact_type="html-app",
    )
    base = _checkpoint(client, artifact_id, label="Before review")

    (folder / "index.html").write_text(
        "<h1>After</h1><script src=\"assets/app.js\"></script>\n",
        encoding="utf-8",
    )
    (folder / "assets" / "app.js").write_text("window.version = 'after';\n", encoding="utf-8")
    compare = _checkpoint(client, artifact_id, label="After review")

    response = client.get(
        f"/api/v1/artifacts/{artifact_id}/diff",
        params={"base": base["version"]["id"], "compare": compare["version"]["id"]},
    )
    assert response.status_code == 200, response.text
    visual = response.json()["visualDiff"]
    assert visual["available"] is True
    assert visual["kind"] in {"html-preview", "screenshot-pixel-diff"}
    if visual["kind"] == "screenshot-pixel-diff":
        assert visual["mode"] == "visual-diff"
        assert visual["changedPixels"] >= 0
        diff_image = client.get(f"/api/v1{visual['diff']['imageRelUrl']}")
        assert diff_image.status_code == 200
        assert diff_image.headers["content-type"].startswith("image/png")
    else:
        assert visual["mode"] == "side-by-side"
    assert visual["base"]["path"] == "index.html"
    assert visual["compare"]["path"] == "index.html"

    base_html = client.get(f"/api/v1{visual['base']['relUrl']}")
    compare_html = client.get(f"/api/v1{visual['compare']['relUrl']}")
    assert base_html.status_code == 200, base_html.text
    assert compare_html.status_code == 200, compare_html.text
    assert "<h1>Before</h1>" in base_html.text
    assert "<h1>After</h1>" in compare_html.text

    base_token = visual["base"]["relUrl"].split("/")[3]
    compare_token = visual["compare"]["relUrl"].split("/")[3]
    base_asset = client.get(f"/api/v1/artifacts/preview-asset/{base_token}/assets/app.js")
    compare_asset = client.get(f"/api/v1/artifacts/preview-asset/{compare_token}/assets/app.js")
    assert base_asset.status_code == 200, base_asset.text
    assert compare_asset.status_code == 200, compare_asset.text
    assert "before" in base_asset.text
    assert "after" in compare_asset.text


def test_preview_mount_can_render_saved_html_version_not_live_folder(client: TestClient):
    artifact_id, folder = _make_artifact(
        files={
            "index.html": "<h1>Saved version</h1><link rel=\"stylesheet\" href=\"assets/app.css\">\n",
            "assets/app.css": "body { color: rgb(1, 2, 3); }\n",
        },
        artifact_type="html-app",
    )
    saved = _checkpoint(client, artifact_id, label="Version to preview")

    (folder / "index.html").write_text(
        "<h1>Live draft</h1><link rel=\"stylesheet\" href=\"assets/app.css\">\n",
        encoding="utf-8",
    )
    (folder / "assets" / "app.css").write_text("body { color: rgb(9, 9, 9); }\n", encoding="utf-8")

    mounted = client.post(
        "/api/v1/artifacts/preview-mount",
        json={
            "path": str(folder / "index.html"),
            "versionId": saved["version"]["id"],
        },
    )
    assert mounted.status_code == 200, mounted.text
    payload = mounted.json()
    assert payload["kind"] == "static"

    html = client.get(f"/api/v1{payload['relUrl']}")
    assert html.status_code == 200, html.text
    assert "<h1>Saved version</h1>" in html.text
    assert "<h1>Live draft</h1>" not in html.text

    token = payload["relUrl"].split("/")[3]
    css = client.get(f"/api/v1/artifacts/preview-asset/{token}/assets/app.css")
    assert css.status_code == 200, css.text
    assert "rgb(1, 2, 3)" in css.text
    assert "rgb(9, 9, 9)" not in css.text


def test_html_diff_can_return_screenshot_pixel_diff(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    from cowork.services import artifact_versions as version_service

    def fake_screenshot_diff(base_entry, compare_entry, output_dir):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (2, 2), (255, 255, 255, 255)).save(output / "base.png")
        Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(output / "compare.png")
        Image.new("RGBA", (2, 2), (255, 38, 0, 190)).save(output / "diff.png")
        return {
            "changedPixels": 4,
            "totalPixels": 4,
            "ratio": 1,
            "threshold": 16,
            "viewport": {"width": 1440, "height": 900},
            "width": 2,
            "height": 2,
        }

    monkeypatch.setattr(version_service, "render_static_html_screenshot_diff", fake_screenshot_diff)
    artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Before</h1>\n"},
        artifact_type="html-app",
    )
    base = _checkpoint(client, artifact_id, label="Before review")
    (folder / "index.html").write_text("<h1>After</h1>\n", encoding="utf-8")
    compare = _checkpoint(client, artifact_id, label="After review")

    response = client.get(
        f"/api/v1/artifacts/{artifact_id}/diff",
        params={"base": base["version"]["id"], "compare": compare["version"]["id"]},
    )
    assert response.status_code == 200, response.text
    visual = response.json()["visualDiff"]
    assert visual["kind"] == "screenshot-pixel-diff"
    assert visual["changedPixels"] == 4
    assert visual["ratio"] == 1
    assert visual["base"]["relUrl"]
    assert visual["base"]["screenshotRelUrl"]
    assert visual["compare"]["screenshotRelUrl"]
    assert visual["diff"]["imageRelUrl"]

    for rel_url in (
        visual["base"]["screenshotRelUrl"],
        visual["compare"]["screenshotRelUrl"],
        visual["diff"]["imageRelUrl"],
    ):
        image_response = client.get(f"/api/v1{rel_url}")
        assert image_response.status_code == 200, image_response.text
        assert image_response.headers["content-type"].startswith("image/png")


def test_html_diff_falls_back_when_screenshot_renderer_is_unavailable(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    from cowork.services import artifact_versions as version_service

    def fail_screenshot_diff(*args, **kwargs):
        raise ScreenshotDiffUnavailable("playwright-unavailable")

    monkeypatch.setattr(version_service, "render_static_html_screenshot_diff", fail_screenshot_diff)
    artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Before</h1>\n"},
        artifact_type="html-app",
    )
    base = _checkpoint(client, artifact_id, label="Before review")
    (folder / "index.html").write_text("<h1>After</h1>\n", encoding="utf-8")
    compare = _checkpoint(client, artifact_id, label="After review")

    response = client.get(
        f"/api/v1/artifacts/{artifact_id}/diff",
        params={"base": base["version"]["id"], "compare": compare["version"]["id"]},
    )
    assert response.status_code == 200, response.text
    visual = response.json()["visualDiff"]
    assert visual["kind"] == "html-preview"
    assert visual["mode"] == "side-by-side"
    assert visual["screenshotUnavailable"] == "playwright-unavailable"
    assert visual["base"]["relUrl"]
    assert visual["compare"]["relUrl"]


def test_html_diff_against_current_draft_exposes_live_preview(client: TestClient):
    artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Saved</h1>\n"},
        artifact_type="html-app",
    )
    base = _checkpoint(client, artifact_id, label="Saved")
    (folder / "index.html").write_text("<h1>Current draft</h1>\n", encoding="utf-8")

    response = client.get(
        "/api/v1/artifacts/diff",
        params={"path": str(folder), "from": base["version"]["id"], "to": "current"},
    )
    assert response.status_code == 200, response.text
    visual = response.json()["visualDiff"]
    assert visual["available"] is True
    assert visual["compare"]["id"] == "current"

    current_html = client.get(f"/api/v1{visual['compare']['relUrl']}")
    assert current_html.status_code == 200, current_html.text
    assert "<h1>Current draft</h1>" in current_html.text


def test_non_html_diff_reports_visual_preview_unavailable(client: TestClient):
    artifact_id, folder = _make_artifact(files={"report.md": "# Saved\n"})
    base = _checkpoint(client, artifact_id, label="Saved")
    (folder / "report.md").write_text("# Updated\n", encoding="utf-8")
    compare = _checkpoint(client, artifact_id, label="Updated")

    response = client.get(
        f"/api/v1/artifacts/{artifact_id}/diff",
        params={"base": base["version"]["id"], "compare": compare["version"]["id"]},
    )
    assert response.status_code == 200, response.text
    visual = response.json()["visualDiff"]
    assert visual["available"] is False
    assert visual["reason"] == "no-html-entry"


def test_fullstack_diff_uses_proxy_url_screenshot_renderer(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    from cowork.services import artifact_versions as version_service

    def fail_if_static_renderer_is_called(*args, **kwargs):
        raise AssertionError("fullstack visual diff should use the proxy URL screenshot renderer")

    calls: dict[str, str] = {}

    def fake_url_screenshot_diff(base_url, compare_url, output_dir, **kwargs):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (2, 2), (255, 255, 255, 255)).save(output / "base.png")
        Image.new("RGBA", (2, 2), (0, 120, 255, 255)).save(output / "compare.png")
        Image.new("RGBA", (2, 2), (255, 38, 0, 190)).save(output / "diff.png")
        calls["base"] = base_url
        calls["compare"] = compare_url
        return {
            "changedPixels": 4,
            "totalPixels": 4,
            "ratio": 1,
            "threshold": 16,
            "viewport": {"width": 1440, "height": 900},
            "width": 2,
            "height": 2,
        }

    monkeypatch.setattr(version_service, "render_static_html_screenshot_diff", fail_if_static_renderer_is_called)
    monkeypatch.setattr(version_service, "render_url_screenshot_diff", fake_url_screenshot_diff)
    artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Runtime before</h1>\n", "server.py": "print('before')\n"},
        artifact_type="fullstack-stateful-app",
    )
    metadata_path = folder / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["port"] = 43210
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    base = _checkpoint(client, artifact_id, label="Runtime before")
    (folder / "index.html").write_text("<h1>Runtime after</h1>\n", encoding="utf-8")
    (folder / "server.py").write_text("print('after')\n", encoding="utf-8")
    compare = _checkpoint(client, artifact_id, label="Runtime after")

    response = client.get(
        f"/api/v1/artifacts/{artifact_id}/diff",
        params={"base": base["version"]["id"], "compare": compare["version"]["id"]},
    )
    assert response.status_code == 200, response.text
    visual = response.json()["visualDiff"]
    assert visual["available"] is True
    assert visual["kind"] == "screenshot-pixel-diff"
    assert visual["limitations"] == ["runtime-proxy-screenshot"]
    assert visual["base"]["relUrl"].startswith("/artifacts/proxy/")
    assert visual["compare"]["relUrl"].startswith("/artifacts/proxy/")
    assert "/api/v1/artifacts/proxy/" in calls["base"]
    assert "/api/v1/artifacts/proxy/" in calls["compare"]


def test_artifact_list_exposes_materialized_last_known_good_path(client: TestClient):
    artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Good</h1>\n"},
        artifact_type="html-app",
    )
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        version = ArtifactVersionService(session).snapshot_artifact(
            folder,
            operation_type="preview",
            label="Last good preview",
            preview_status="ready",
        )

    (folder / "index.html").write_text("<h1>Broken draft</h1>\n", encoding="utf-8")
    response = client.get("/api/v1/artifacts/", params={"project_path": str(_general_project_path())})
    assert response.status_code == 200, response.text
    card = next(item for item in response.json() if item["id"] == artifact_id)

    assert card["lastKnownGoodVersionId"] == str(version.id)
    assert card["lastGood"]["versionId"] == str(version.id)
    assert card["lastGoodPath"] != str(folder / "index.html")
    assert Path(card["lastGoodPath"]).is_file()
    assert "<h1>Good</h1>" in Path(card["lastGoodPath"]).read_text(encoding="utf-8")


def test_live_preview_mount_promotes_ready_preview_version(client: TestClient):
    _artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Preview ready</h1>\n"},
        artifact_type="html-app",
    )

    response = client.post("/api/v1/artifacts/preview-mount", json={"path": str(folder / "index.html")})
    assert response.status_code == 200, response.text

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        artifact = session.exec(select(Artifact).where(Artifact.path == str(folder.resolve()))).one()
        version = session.get(ArtifactVersion, artifact.current_version_id)
        deployment = session.exec(
            select(ArtifactDeployment)
            .where(ArtifactDeployment.artifact_id == artifact.id)
            .where(ArtifactDeployment.target == "preview")
            .where(ArtifactDeployment.status == "ready")
        ).one()

    assert version is not None
    assert version.preview_status == "ready"
    assert version.publish_status == "unpublished"
    assert artifact.last_known_good_version_id == version.id
    assert deployment.version_id == version.id


def test_failed_live_preview_rolls_back_to_last_known_good(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    _artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Preview good</h1>\n"},
        artifact_type="html-app",
    )
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        good = ArtifactVersionService(session).snapshot_artifact(
            folder,
            operation_type="preview",
            label="Last good preview",
            preview_status="ready",
        )

    (folder / "index.html").write_text("<h1>Preview broken</h1>\n", encoding="utf-8")

    async def fail_mount(path):
        raise RuntimeError("preview render crashed")

    monkeypatch.setattr("cowork.api.v1.endpoints.artifacts.mount_preview", fail_mount)

    response = client.post("/api/v1/artifacts/preview-mount", json={"path": str(folder / "index.html")})
    assert response.status_code == 500, response.text
    assert (folder / "index.html").read_text(encoding="utf-8") == "<h1>Preview good</h1>\n"

    with Session(engine) as session:
        artifact = session.exec(select(Artifact).where(Artifact.path == str(folder.resolve()))).one()
        failed_deployment = session.exec(
            select(ArtifactDeployment)
            .where(ArtifactDeployment.artifact_id == artifact.id)
            .where(ArtifactDeployment.target == "preview")
            .where(ArtifactDeployment.status == "failed")
        ).one()
        failed_version = session.get(ArtifactVersion, failed_deployment.version_id)

    assert artifact.current_version_id == good.id
    assert artifact.last_known_good_version_id == good.id
    assert failed_version is not None
    assert failed_version.preview_status == "failed"
    assert failed_version.publish_status == "unpublished"
    assert failed_deployment.details["error"] == "preview render crashed"


def test_proxy_preview_launch_failure_does_not_promote_last_known_good(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    _artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Proxy good</h1>\n"},
        artifact_type="fullstack-stateful-app",
    )
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        good = ArtifactVersionService(session).snapshot_artifact(
            folder,
            operation_type="preview",
            label="Last good proxy preview",
            preview_status="ready",
        )

    (folder / "index.html").write_text("<h1>Proxy broken</h1>\n", encoding="utf-8")

    async def failed_proxy_mount(path):
        return {
            "kind": "proxy",
            "token": "proxy-token",
            "artifactDir": str(folder),
            "port": 43210,
            "backendRunning": False,
            "launchError": "backend failed to launch",
        }

    monkeypatch.setattr("cowork.api.v1.endpoints.artifacts.mount_preview", failed_proxy_mount)

    response = client.post("/api/v1/artifacts/preview-mount", json={"path": str(folder / "index.html")})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["kind"] == "proxy"
    assert payload["backendRunning"] is False
    assert payload["launchError"] == "backend failed to launch"
    assert (folder / "index.html").read_text(encoding="utf-8") == "<h1>Proxy good</h1>\n"

    with Session(engine) as session:
        artifact = session.exec(select(Artifact).where(Artifact.path == str(folder.resolve()))).one()
        deployments = session.exec(
            select(ArtifactDeployment)
            .where(ArtifactDeployment.artifact_id == artifact.id)
            .where(ArtifactDeployment.target == "preview")
        ).all()
        failed = [row for row in deployments if row.status == "failed"]
        ready = [row for row in deployments if row.status == "ready" and row.version_id != good.id]

    assert artifact.current_version_id == good.id
    assert artifact.last_known_good_version_id == good.id
    assert len(failed) == 1
    assert failed[0].details["error"] == "backend failed to launch"
    assert ready == []


def test_proxy_preview_mount_exposes_published_version_fields(
    monkeypatch: pytest.MonkeyPatch,
):
    from cowork.services import artifacts as artifact_service

    _artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Published proxy</h1>\n"},
        artifact_type="fullstack-stateful-app",
    )
    metadata_path = folder / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["port"] = 43210
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    (folder / ".published.json").write_text(
        json.dumps({
            "index.html": {
                "url": "https://4nton.ai/p/proxy",
                "version_id": "version-proxy",
                "files_hash": "files-proxy",
                "manifest_hash": "manifest-proxy",
                "version_number": 7,
            }
        }),
        encoding="utf-8",
    )

    async def backend_running(root, port):
        return True, "", port

    monkeypatch.setattr(artifact_service, "_ensure_backend_running", backend_running)

    payload = asyncio.run(artifact_service.mount_preview(folder / "index.html"))
    assert payload["kind"] == "proxy"
    assert payload["publishedUrl"] == "https://4nton.ai/p/proxy"
    assert payload["publishedVersionId"] == "version-proxy"
    assert payload["publishedFilesHash"] == "files-proxy"
    assert payload["publishedManifestHash"] == "manifest-proxy"
    assert payload["publishedVersionNumber"] == 7


def test_dataset_diff_reports_schema_and_row_changes(client: TestClient):
    artifact_id, folder = _make_artifact(
        files={"data.csv": "id,region,total\n1,west,10\n2,east,20\n"},
        artifact_type="dataset",
    )
    base = _checkpoint(client, artifact_id, label="Baseline")

    (folder / "data.csv").write_text(
        "id,region,total,owner\n1,west,12,Alex\n3,north,30,Sam\n",
        encoding="utf-8",
    )
    compare = _checkpoint(client, artifact_id, label="Updated")

    response = client.get(
        f"/api/v1/artifacts/{artifact_id}/diff",
        params={"base": base["version"]["id"], "compare": compare["version"]["id"], "kind": "text"},
    )
    assert response.status_code == 200, response.text
    dataset = response.json()["datasetDiff"]
    assert dataset["path"] == "data.csv"
    assert dataset["schema"]["added"] == ["owner"]
    assert dataset["rows"] == {"before": 2, "after": 2, "added": 1, "removed": 1, "modified": 1}
    changed = {row["key"]: row["status"] for row in dataset["changedRows"]}
    assert changed == {"1": "modified", "2": "removed", "3": "added"}


def test_dataset_diff_reports_schema_changes_for_empty_csv(client: TestClient):
    artifact_id, folder = _make_artifact(
        files={"data.csv": "id,total\n"},
        artifact_type="dataset",
    )
    base = _checkpoint(client, artifact_id, label="Baseline")

    (folder / "data.csv").write_text("id,total,owner\n", encoding="utf-8")
    compare = _checkpoint(client, artifact_id, label="Updated schema")

    response = client.get(
        f"/api/v1/artifacts/{artifact_id}/diff",
        params={"base": base["version"]["id"], "compare": compare["version"]["id"], "kind": "text"},
    )
    assert response.status_code == 200, response.text
    dataset = response.json()["datasetDiff"]
    assert dataset["schema"]["before"] == ["id", "total"]
    assert dataset["schema"]["after"] == ["id", "total", "owner"]
    assert dataset["schema"]["added"] == ["owner"]
    assert dataset["rows"] == {"before": 0, "after": 0, "added": 0, "removed": 0, "modified": 0}


def test_dataset_diff_reports_json_array_schema_and_row_changes(client: TestClient):
    artifact_id, folder = _make_artifact(
        files={
            "data.json": json.dumps(
                [
                    {"id": "1", "region": "west", "total": 10},
                    {"id": "2", "region": "east", "total": 20},
                ]
            )
        },
        artifact_type="dataset",
    )
    base = _checkpoint(client, artifact_id, label="Baseline")

    (folder / "data.json").write_text(
        json.dumps(
            [
                {"id": "1", "region": "west", "total": 12, "owner": "Alex"},
                {"id": "3", "region": "north", "total": 30, "owner": "Sam"},
            ]
        ),
        encoding="utf-8",
    )
    compare = _checkpoint(client, artifact_id, label="Updated")

    response = client.get(
        f"/api/v1/artifacts/{artifact_id}/diff",
        params={"base": base["version"]["id"], "compare": compare["version"]["id"], "kind": "text"},
    )
    assert response.status_code == 200, response.text
    dataset = response.json()["datasetDiff"]
    assert dataset["path"] == "data.json"
    assert dataset["schema"]["added"] == ["owner"]
    assert dataset["rows"] == {"before": 2, "after": 2, "added": 1, "removed": 1, "modified": 1}
    changed = {row["key"]: row["status"] for row in dataset["changedRows"]}
    assert changed == {"1": "modified", "2": "removed", "3": "added"}


def test_fork_version_materializes_copy_from_checkpoint(client: TestClient):
    artifact_id, folder = _make_artifact(files={"report.md": "old\n"})
    first = _checkpoint(client, artifact_id, label="Original")
    live_metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    live_metadata.update(
        {
            "name": "Live Mutated Artifact",
            "description": "Changed after the checkpoint",
            "type": "html-app",
        }
    )
    (folder / "metadata.json").write_text(json.dumps(live_metadata), encoding="utf-8")
    (folder / "report.md").write_text("new\n", encoding="utf-8")
    _checkpoint(client, artifact_id, label="Later")
    live_metadata["id"] = "live-mutated-artifact"
    (folder / "metadata.json").write_text(json.dumps(live_metadata), encoding="utf-8")

    response = client.post(
        f"/api/v1/artifacts/{artifact_id}/fork",
        json={
            "path": str(folder),
            "versionId": first["version"]["id"],
            "title": "Forked Research",
            "slug": "version-test-forked-research",
        },
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    fork_folder = Path(payload["artifactPath"])
    assert fork_folder != folder
    assert fork_folder.name.startswith("version-test-forked-research")
    assert (fork_folder / "report.md").read_text(encoding="utf-8") == "old\n"
    fork_metadata = json.loads((fork_folder / "metadata.json").read_text(encoding="utf-8"))
    assert fork_metadata["id"].startswith(f"{artifact_id}-")
    assert fork_metadata["description"] == "Analyst-ready artifact"
    assert fork_metadata["type"] == "document"
    assert payload["version"]["forkedFromVersionId"] == first["version"]["id"]
    assert payload["version"]["branchName"] == "Forked Research"


def test_checkpoint_body_path_can_create_artifact_metadata(client: TestClient, tmp_path: Path):
    project_path = tmp_path / "project"
    project_path.mkdir(parents=True)
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        project = Project(name=f"path-created-project-{uuid4().hex}", path=str(project_path))
        session.add(project)
        session.commit()
    folder = project_path / ".anton" / "artifacts" / "path-created"

    response = client.post(
        "/api/v1/artifacts/path-only/checkpoints",
        json={
            "path": str(folder),
            "title": "Path Created Artifact",
            "type": "document",
            "primary": "report.md",
            "label": "Empty starting point",
        },
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "path-only"
    assert metadata["name"] == "Path Created Artifact"
    assert metadata["primary"] == "report.md"
    assert payload["version"]["fileCount"] == 0
    assert payload["artifact"]["title"] == "Path Created Artifact"


def test_artifact_comments_can_be_suggested_resolved_and_reopened(client: TestClient):
    artifact_id, folder = _make_artifact(files={"report.md": "# Draft\n"})
    _checkpoint(client, artifact_id, label="Baseline")

    comment_response = client.post(
        "/api/v1/artifacts/comments",
        json={
            "path": str(folder),
            "body": "Please tighten the executive summary.",
            "kind": "suggestion",
            "anchor": {"path": "report.md", "line": 1},
        },
    )
    assert comment_response.status_code == 201, comment_response.text
    created = comment_response.json()["comment"]
    assert created["kind"] == "suggestion"
    assert created["status"] == "open"
    assert created["anchor"] == {"path": "report.md", "line": 1}

    listed_response = client.get("/api/v1/artifacts/comments", params={"path": str(folder)})
    assert listed_response.status_code == 200, listed_response.text
    listed = listed_response.json()
    assert listed["comments"][0]["id"] == created["id"]
    assert "suggested" in {event["eventType"] for event in listed["activity"]}

    resolved_response = client.post(f"/api/v1/artifacts/comments/{created['id']}/resolve")
    assert resolved_response.status_code == 200, resolved_response.text
    assert resolved_response.json()["comment"]["status"] == "resolved"

    reopened_response = client.post(f"/api/v1/artifacts/comments/{created['id']}/reopen")
    assert reopened_response.status_code == 200, reopened_response.text
    assert reopened_response.json()["comment"]["status"] == "open"

    accepted_response = client.post(f"/api/v1/artifacts/comments/{created['id']}/accept")
    assert accepted_response.status_code == 200, accepted_response.text
    assert accepted_response.json()["comment"]["status"] == "accepted"

    rejected_response = client.post(f"/api/v1/artifacts/comments/{created['id']}/reject")
    assert rejected_response.status_code == 200, rejected_response.text
    assert rejected_response.json()["comment"]["status"] == "rejected"


def test_artifact_comments_track_collaborator_read_state(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifacts.principal_from_authorization_header",
        _fake_version_principal,
    )
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifact_versions.principal_from_authorization_header",
        _fake_version_principal,
    )
    _own_general_project_for_versions()
    artifact_id, folder = _make_artifact(files={"report.md": "# Draft\n"})
    baseline_response = client.post(
        f"/api/v1/artifacts/{artifact_id}/checkpoints",
        headers={"Authorization": "Bearer owner"},
        json={"label": "Baseline"},
    )
    assert baseline_response.status_code == 201, baseline_response.text

    created_response = client.post(
        "/api/v1/artifacts/comments",
        headers={"Authorization": "Bearer owner"},
        json={
            "path": str(folder),
            "body": "Please review the finance summary.",
            "kind": "review",
        },
    )
    assert created_response.status_code == 201, created_response.text
    review = created_response.json()["comment"]

    listed_response = client.get(
        "/api/v1/artifacts/comments",
        headers={"Authorization": "Bearer editor"},
        params={"path": str(folder)},
    )
    assert listed_response.status_code == 200, listed_response.text
    listed = listed_response.json()
    assert listed["viewerState"]["available"] is True
    assert listed["viewerState"]["unreadComments"] == 1
    assert listed["viewerState"]["needsAction"] == 1
    assert listed["viewerState"]["reviewRequests"] == {"open": 1, "needsAction": 1, "unread": 1}
    assert listed["comments"][0]["viewerState"]["unread"] is True
    assert listed["comments"][0]["viewerState"]["needsAction"] is True

    mark_seen_response = client.post(
        "/api/v1/artifacts/comments/read",
        headers={"Authorization": "Bearer editor"},
        json={"path": str(folder)},
    )
    assert mark_seen_response.status_code == 200, mark_seen_response.text
    marked = mark_seen_response.json()
    assert marked["viewerState"]["unreadComments"] == 0
    assert marked["viewerState"]["needsAction"] == 1
    assert marked["comments"][0]["viewerState"]["seen"] is True

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        collaborator = session.exec(
            select(ProjectCollaborator)
            .where(ProjectCollaborator.project_id == GENERAL_PROJECT_ID)
            .where(ProjectCollaborator.email == "editor@example.com")
        ).one()
        artifact = session.exec(select(Artifact).where(Artifact.path == str(folder.resolve()))).one()
        assert collaborator.notification_state["artifacts"][str(artifact.id)]["lastReadAt"]

    resolved_response = client.post(
        f"/api/v1/artifacts/comments/{review['id']}/resolve",
        headers={"Authorization": "Bearer editor"},
    )
    assert resolved_response.status_code == 200, resolved_response.text
    after_resolve = client.get(
        "/api/v1/artifacts/comments",
        headers={"Authorization": "Bearer editor"},
        params={"path": str(folder)},
    ).json()
    assert after_resolve["viewerState"]["needsAction"] == 0
    assert after_resolve["viewerState"]["reviewRequests"]["open"] == 0


def test_artifact_list_exposes_viewer_review_state(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifacts.principal_from_authorization_header",
        _fake_version_principal,
    )
    _own_general_project_for_versions()
    artifact_id, folder = _make_artifact(files={"report.md": "# Draft\n"})

    created_response = client.post(
        "/api/v1/artifacts/comments",
        headers={"Authorization": "Bearer owner"},
        json={
            "path": str(folder),
            "body": "Please review before this goes out.",
            "kind": "review",
        },
    )
    assert created_response.status_code == 201, created_response.text
    review = created_response.json()["comment"]

    listed_response = client.get(
        "/api/v1/artifacts/",
        headers={"Authorization": "Bearer editor"},
        params={"project_path": str(_general_project_path())},
    )
    assert listed_response.status_code == 200, listed_response.text
    card = next(item for item in listed_response.json() if item["id"] == artifact_id)
    viewer_state = card["reviewSummary"]["viewerState"]
    assert viewer_state["available"] is True
    assert viewer_state["unreadComments"] == 1
    assert viewer_state["needsAction"] == 1
    assert viewer_state["reviewRequests"] == {"open": 1, "needsAction": 1, "unread": 1}

    mark_seen_response = client.post(
        "/api/v1/artifacts/comments/read",
        headers={"Authorization": "Bearer editor"},
        json={"path": str(folder)},
    )
    assert mark_seen_response.status_code == 200, mark_seen_response.text

    after_seen_response = client.get(
        "/api/v1/artifacts/",
        headers={"Authorization": "Bearer editor"},
        params={"project_path": str(_general_project_path())},
    )
    assert after_seen_response.status_code == 200, after_seen_response.text
    after_seen = next(item for item in after_seen_response.json() if item["id"] == artifact_id)
    after_seen_state = after_seen["reviewSummary"]["viewerState"]
    assert after_seen_state["unreadComments"] == 0
    assert after_seen_state["unreadActivity"] == 0
    assert after_seen_state["needsAction"] == 1
    assert after_seen_state["reviewRequests"] == {"open": 1, "needsAction": 1, "unread": 0}

    resolved_response = client.post(
        f"/api/v1/artifacts/comments/{review['id']}/resolve",
        headers={"Authorization": "Bearer editor"},
    )
    assert resolved_response.status_code == 200, resolved_response.text

    after_resolve_response = client.get(
        "/api/v1/artifacts/",
        headers={"Authorization": "Bearer editor"},
        params={"project_path": str(_general_project_path())},
    )
    assert after_resolve_response.status_code == 200, after_resolve_response.text
    after_resolve = next(item for item in after_resolve_response.json() if item["id"] == artifact_id)
    assert after_resolve["reviewSummary"]["viewerState"]["needsAction"] == 0
    assert after_resolve["reviewSummary"]["viewerState"]["reviewRequests"]["open"] == 0


def test_artifact_read_state_is_isolated_per_collaborator(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifacts.principal_from_authorization_header",
        _fake_version_principal,
    )
    _own_general_project_for_versions()
    artifact_id, folder = _make_artifact(files={"report.md": "# Draft\n"})

    created_response = client.post(
        "/api/v1/artifacts/comments",
        headers={"Authorization": "Bearer owner"},
        json={"path": str(folder), "body": "Reviewer eyes needed.", "kind": "review"},
    )
    assert created_response.status_code == 201, created_response.text

    editor_before = client.get(
        "/api/v1/artifacts/comments",
        headers={"Authorization": "Bearer editor"},
        params={"path": str(folder)},
    )
    assert editor_before.status_code == 200, editor_before.text
    assert editor_before.json()["viewerState"]["unreadComments"] == 1
    assert editor_before.json()["viewerState"]["needsAction"] == 1

    viewer_before = client.get(
        "/api/v1/artifacts/comments",
        headers={"Authorization": "Bearer viewer"},
        params={"path": str(folder)},
    )
    assert viewer_before.status_code == 200, viewer_before.text
    assert viewer_before.json()["viewerState"]["unreadComments"] == 1
    assert viewer_before.json()["viewerState"]["needsAction"] == 0

    owner_before = client.get(
        "/api/v1/artifacts/comments",
        headers={"Authorization": "Bearer owner"},
        params={"path": str(folder)},
    )
    assert owner_before.status_code == 200, owner_before.text
    assert owner_before.json()["viewerState"]["unreadComments"] == 0
    assert owner_before.json()["viewerState"]["needsAction"] == 0

    mark_seen_response = client.post(
        "/api/v1/artifacts/comments/read",
        headers={"Authorization": "Bearer editor"},
        json={"path": str(folder)},
    )
    assert mark_seen_response.status_code == 200, mark_seen_response.text

    editor_after = client.get(
        "/api/v1/artifacts/comments",
        headers={"Authorization": "Bearer editor"},
        params={"path": str(folder)},
    )
    assert editor_after.status_code == 200, editor_after.text
    assert editor_after.json()["viewerState"]["unreadComments"] == 0
    assert editor_after.json()["viewerState"]["needsAction"] == 1

    viewer_after = client.get(
        "/api/v1/artifacts/comments",
        headers={"Authorization": "Bearer viewer"},
        params={"path": str(folder)},
    )
    assert viewer_after.status_code == 200, viewer_after.text
    assert viewer_after.json()["viewerState"]["unreadComments"] == 1
    assert viewer_after.json()["viewerState"]["reviewRequests"]["unread"] == 1

    listed_as_editor = client.get(
        "/api/v1/artifacts/",
        headers={"Authorization": "Bearer editor"},
        params={"project_path": str(_general_project_path())},
    )
    assert listed_as_editor.status_code == 200, listed_as_editor.text
    editor_card = next(item for item in listed_as_editor.json() if item["id"] == artifact_id)
    assert editor_card["reviewSummary"]["viewerState"]["unreadComments"] == 0

    listed_as_viewer = client.get(
        "/api/v1/artifacts/",
        headers={"Authorization": "Bearer viewer"},
        params={"project_path": str(_general_project_path())},
    )
    assert listed_as_viewer.status_code == 200, listed_as_viewer.text
    viewer_card = next(item for item in listed_as_viewer.json() if item["id"] == artifact_id)
    assert viewer_card["reviewSummary"]["viewerState"]["unreadComments"] == 1
    assert viewer_card["reviewSummary"]["viewerState"]["needsAction"] == 0


def test_artifact_comment_replies_must_belong_to_same_artifact(client: TestClient):
    _first_id, first_folder = _make_artifact(files={"report.md": "# First\n"})
    _second_id, second_folder = _make_artifact(files={"report.md": "# Second\n"})

    parent_response = client.post(
        "/api/v1/artifacts/comments",
        json={"path": str(first_folder), "body": "Parent note.", "kind": "comment"},
    )
    assert parent_response.status_code == 201, parent_response.text
    parent = parent_response.json()["comment"]

    reply_response = client.post(
        "/api/v1/artifacts/comments",
        json={
            "path": str(first_folder),
            "body": "Same artifact reply.",
            "kind": "comment",
            "parentCommentId": parent["id"],
        },
    )
    assert reply_response.status_code == 201, reply_response.text
    assert reply_response.json()["comment"]["parentCommentId"] == parent["id"]

    cross_artifact = client.post(
        "/api/v1/artifacts/comments",
        json={
            "path": str(second_folder),
            "body": "Wrong artifact reply.",
            "kind": "comment",
            "parentCommentId": parent["id"],
        },
    )
    assert cross_artifact.status_code == 400, cross_artifact.text
    assert "same artifact" in cross_artifact.text


def test_artifact_list_exposes_review_summary_counts(client: TestClient):
    artifact_id, folder = _make_artifact(files={"report.md": "# Draft\n"})
    _checkpoint(client, artifact_id, label="Baseline")

    note_response = client.post(
        "/api/v1/artifacts/comments",
        json={"path": str(folder), "body": "Looks good after this note.", "kind": "comment"},
    )
    assert note_response.status_code == 201, note_response.text
    note_id = note_response.json()["comment"]["id"]
    resolved_response = client.post(f"/api/v1/artifacts/comments/{note_id}/resolve")
    assert resolved_response.status_code == 200, resolved_response.text

    suggestion_response = client.post(
        "/api/v1/artifacts/comments",
        json={"path": str(folder), "body": "Tighten this paragraph.", "kind": "suggestion"},
    )
    assert suggestion_response.status_code == 201, suggestion_response.text

    review_response = client.post(
        "/api/v1/artifacts/comments",
        json={"path": str(folder), "body": "Please review before publishing.", "kind": "review"},
    )
    assert review_response.status_code == 201, review_response.text

    listed = client.get("/api/v1/artifacts/", params={"project_path": str(_general_project_path())})
    assert listed.status_code == 200, listed.text
    card = next(item for item in listed.json() if item["id"] == artifact_id)
    summary = card["reviewSummary"]

    assert summary["open"] == 2
    assert summary["unresolved"] == 2
    assert summary["comments"] == 0
    assert summary["suggestions"] == 1
    assert summary["reviewRequests"] == 1
    assert summary["resolved"] == 1
    assert summary["needsReview"] is True
    assert summary["latestAt"]


def test_suggestion_patch_can_be_previewed_and_accepted_with_checkpoints(client: TestClient):
    artifact_id, folder = _make_artifact(files={"report.md": "# Draft\nSummary: rough\n"})
    _checkpoint(client, artifact_id, label="Baseline")

    comment_response = client.post(
        "/api/v1/artifacts/comments",
        json={
            "path": str(folder),
            "body": "Make the summary clearer.",
            "kind": "suggestion",
            "anchor": {"path": "report.md", "line": 2},
            "proposedPatch": {
                "operations": [
                    {
                        "type": "replace_text",
                        "path": "report.md",
                        "find": "Summary: rough",
                        "replace": "Summary: clear",
                    }
                ]
            },
        },
    )
    assert comment_response.status_code == 201, comment_response.text
    created = comment_response.json()["comment"]
    assert created["proposedPatch"]["operations"][0]["path"] == "report.md"

    preview_response = client.post(f"/api/v1/artifacts/comments/{created['id']}/preview")
    assert preview_response.status_code == 200, preview_response.text
    preview = preview_response.json()
    assert preview["available"] is True
    assert preview["changedPaths"] == ["report.md"]
    assert "-Summary: rough" in preview["diff"]["textDiff"]
    assert "+Summary: clear" in preview["diff"]["textDiff"]
    assert (folder / "report.md").read_text(encoding="utf-8") == "# Draft\nSummary: rough\n"

    accepted_response = client.post(f"/api/v1/artifacts/comments/{created['id']}/accept")
    assert accepted_response.status_code == 200, accepted_response.text
    accepted = accepted_response.json()
    assert accepted["comment"]["status"] == "accepted"
    assert accepted["createdCheckpoint"]["operationType"] == "review_safety"
    assert accepted["version"]["operationType"] == "review_accept"
    assert accepted["changedPaths"] == ["report.md"]
    assert accepted["comment"]["notificationState"]["appliedVersionId"] == accepted["version"]["id"]
    assert (folder / "report.md").read_text(encoding="utf-8") == "# Draft\nSummary: clear\n"

    listed_response = client.get(f"/api/v1/artifacts/{artifact_id}/versions")
    assert listed_response.status_code == 200, listed_response.text
    operations = [version["operationType"] for version in listed_response.json()["versions"]]
    assert operations[:2] == ["review_accept", "review_safety"]


def test_stale_suggestion_patch_must_be_reviewed_again_before_apply(client: TestClient):
    artifact_id, folder = _make_artifact(files={"report.md": "# Draft\nSummary: rough\n"})
    _checkpoint(client, artifact_id, label="Baseline")

    comment_response = client.post(
        "/api/v1/artifacts/comments",
        json={
            "path": str(folder),
            "body": "Make the summary clearer.",
            "kind": "suggestion",
            "proposedPatch": {
                "operations": [
                    {
                        "type": "replace_text",
                        "path": "report.md",
                        "find": "rough",
                        "replace": "clear",
                    }
                ]
            },
        },
    )
    assert comment_response.status_code == 201, comment_response.text
    comment = comment_response.json()["comment"]

    (folder / "report.md").write_text("# Draft\nSummary: current edit\n", encoding="utf-8")
    _checkpoint(client, artifact_id, label="Independent edit")

    preview_response = client.post(f"/api/v1/artifacts/comments/{comment['id']}/preview")
    assert preview_response.status_code == 400, preview_response.text
    assert "older artifact version" in preview_response.json()["detail"]

    accept_response = client.post(f"/api/v1/artifacts/comments/{comment['id']}/accept")
    assert accept_response.status_code == 400, accept_response.text
    assert "older artifact version" in accept_response.json()["detail"]
    assert (folder / "report.md").read_text(encoding="utf-8") == "# Draft\nSummary: current edit\n"


def test_rejecting_suggestion_patch_does_not_change_files(client: TestClient):
    artifact_id, folder = _make_artifact(files={"report.md": "# Draft\nSummary: rough\n"})
    _checkpoint(client, artifact_id, label="Baseline")

    comment_response = client.post(
        "/api/v1/artifacts/comments",
        json={
            "path": str(folder),
            "body": "Make the summary clearer.",
            "kind": "suggestion",
            "proposedPatch": {
                "operations": [
                    {
                        "type": "replace_text",
                        "path": "report.md",
                        "find": "rough",
                        "replace": "clear",
                    }
                ]
            },
        },
    )
    assert comment_response.status_code == 201, comment_response.text
    comment = comment_response.json()["comment"]

    rejected_response = client.post(f"/api/v1/artifacts/comments/{comment['id']}/reject")
    assert rejected_response.status_code == 200, rejected_response.text
    assert rejected_response.json()["comment"]["status"] == "rejected"
    assert (folder / "report.md").read_text(encoding="utf-8") == "# Draft\nSummary: rough\n"


def test_failed_suggestion_patch_apply_leaves_files_unchanged(client: TestClient):
    artifact_id, folder = _make_artifact(files={"report.md": "# Draft\nSummary: rough\n"})
    _checkpoint(client, artifact_id, label="Baseline")

    comment_response = client.post(
        "/api/v1/artifacts/comments",
        json={
            "path": str(folder),
            "body": "Apply two edits.",
            "kind": "suggestion",
            "proposedPatch": {
                "operations": [
                    {
                        "type": "replace_text",
                        "path": "report.md",
                        "find": "rough",
                        "replace": "clear",
                    },
                    {
                        "type": "remove_file",
                        "path": "missing.md",
                    },
                ]
            },
        },
    )
    assert comment_response.status_code == 201, comment_response.text
    comment = comment_response.json()["comment"]

    apply_response = client.post(f"/api/v1/artifacts/comments/{comment['id']}/apply")
    assert apply_response.status_code == 404, apply_response.text
    assert (folder / "report.md").read_text(encoding="utf-8") == "# Draft\nSummary: rough\n"


def test_publish_records_deployment_against_exact_version(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    from cowork.api.v1.endpoints import publish as publish_endpoint

    artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Published</h1>\n"},
        artifact_type="html-app",
    )
    published_url = f"https://4nton.ai/p/{artifact_id}"
    monkeypatch.setattr(
        publish_endpoint,
        "_publish",
        lambda path, password=None, access=None: {
            "url": published_url,
            "publishedUrl": published_url,
            "access": {"mode": "public"},
        },
    )

    response = client.post("/api/v1/publish/", json={"path": str(folder)})
    assert response.status_code == 200, response.text
    assert response.json()["publishedVersionId"]

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        deployment = session.exec(
            select(ArtifactDeployment)
            .join(ArtifactVersion, ArtifactDeployment.version_id == ArtifactVersion.id)
            .where(ArtifactVersion.operation_type == "publish")
            .where(ArtifactDeployment.url == published_url)
        ).one()
        artifact = session.get(Artifact, deployment.artifact_id)
        version = session.get(ArtifactVersion, deployment.version_id)

        assert artifact is not None
        assert version is not None
        assert Path(artifact.path).resolve() == folder.resolve()
        assert version.operation_type == "publish"
        assert version.publish_status == "published"
        assert deployment.version_id == version.id
        assert deployment.status == "published"
        assert deployment.url == published_url
        assert response.json()["publishedVersionId"] == str(version.id)


def test_publish_activity_uses_authenticated_actor(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    from cowork.api.v1.endpoints import publish as publish_endpoint

    _own_general_project_for_versions()
    artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Actor publish</h1>\n"},
        artifact_type="html-app",
    )
    published_url = f"https://4nton.ai/p/{artifact_id}"
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.publish.principal_from_authorization_header",
        _fake_version_principal,
    )
    monkeypatch.setattr(
        publish_endpoint,
        "_publish",
        lambda path, password=None, access=None: {
            "url": published_url,
            "publishedUrl": published_url,
            "access": {"mode": "public"},
        },
    )

    response = client.post(
        "/api/v1/publish/",
        headers={"Authorization": "Bearer editor"},
        json={"path": str(folder)},
    )
    assert response.status_code == 200, response.text
    version_id = UUID(response.json()["publishedVersionId"])

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        checkpoint_event = session.exec(
            select(ArtifactActivityEvent)
            .where(ArtifactActivityEvent.version_id == version_id)
            .where(ArtifactActivityEvent.event_type == "publish")
        ).one()
        published_event = session.exec(
            select(ArtifactActivityEvent)
            .where(ArtifactActivityEvent.version_id == version_id)
            .where(ArtifactActivityEvent.event_type == "published")
        ).one()
        deployment = session.exec(
            select(ArtifactDeployment)
            .where(ArtifactDeployment.version_id == version_id)
            .where(ArtifactDeployment.status == "published")
        ).one()

    assert checkpoint_event.actor_name == "Editor"
    assert checkpoint_event.details["actorEmail"] == "editor@example.com"
    assert checkpoint_event.details["actorSubject"] == "editor"
    assert published_event.actor_name == "Editor"
    assert deployment.details["actorEmail"] == "editor@example.com"


def test_unpublish_records_deployment_against_published_version(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    from cowork.api.v1.endpoints import publish as publish_endpoint

    artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Unpublish me</h1>\n"},
        artifact_type="html-app",
    )
    checkpoint = _checkpoint(client, artifact_id, path=str(folder), label="Published baseline")
    version_id = checkpoint["version"]["id"]
    published_url = f"https://4nton.ai/p/{artifact_id}"
    monkeypatch.setattr(
        publish_endpoint,
        "_unpublish",
        lambda path: {
            "status": "ok",
            "publishedUrl": published_url,
            "publishedVersionId": version_id,
            "publishedFilesHash": checkpoint["version"]["filesHash"],
            "publishedManifestHash": checkpoint["version"]["manifestHash"],
            "publishedVersionNumber": checkpoint["version"]["versionNumber"],
        },
    )

    response = client.delete("/api/v1/publish/", params={"path": str(folder)})
    assert response.status_code == 200, response.text

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        version = session.get(ArtifactVersion, UUID(version_id))
        deployment = session.exec(
            select(ArtifactDeployment)
            .where(ArtifactDeployment.version_id == UUID(version_id))
            .where(ArtifactDeployment.status == "unpublished")
        ).one()
        event = session.exec(
            select(ArtifactActivityEvent)
            .where(ArtifactActivityEvent.version_id == UUID(version_id))
            .where(ArtifactActivityEvent.event_type == "unpublished")
        ).one()

    assert version is not None
    assert version.publish_status == "unpublished"
    assert deployment.url == published_url
    assert deployment.details["previousPublishedVersionId"] == version_id
    assert event.details["target"] == "publish"


def test_publish_sidecar_and_preview_are_pinned_to_version(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    from cowork.services import publish as publish_service

    artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Versioned publish</h1>\n"},
        artifact_type="html-app",
    )
    published_url = f"https://4nton.ai/p/{artifact_id}"

    class FakeVault:
        def __init__(self, path):
            self.path = path

    def fake_publish(*args, **kwargs):
        return {"view_url": published_url, "report_id": "report-versioned", "md5": "abc123"}

    monkeypatch.setattr(
        publish_service,
        "get_user_settings",
        lambda: types.SimpleNamespace(minds_api_key="test-key", publish_url="https://4nton.ai"),
    )

    monkeypatch.setitem(sys.modules, "anton", types.ModuleType("anton"))
    monkeypatch.setitem(sys.modules, "anton.core", types.ModuleType("anton.core"))
    monkeypatch.setitem(sys.modules, "anton.core.datasources", types.ModuleType("anton.core.datasources"))
    data_vault_mod = types.ModuleType("anton.core.datasources.data_vault")
    data_vault_mod.LocalDataVault = FakeVault
    monkeypatch.setitem(sys.modules, "anton.core.datasources.data_vault", data_vault_mod)
    publisher_mod = types.ModuleType("anton.publisher")
    publisher_mod.publish = fake_publish
    monkeypatch.setitem(sys.modules, "anton.publisher", publisher_mod)

    response = client.post("/api/v1/publish/", json={"path": str(folder)})
    assert response.status_code == 200, response.text
    payload = response.json()
    published_version_id = payload["publishedVersionId"]
    assert published_version_id

    sidecar = json.loads((folder / ".published.json").read_text(encoding="utf-8"))
    entry = sidecar["index.html"]
    assert entry["url"] == published_url
    assert entry["version_id"] == published_version_id
    assert entry["files_hash"] == payload["publishedFilesHash"]
    assert entry["manifest_hash"] == payload["publishedManifestHash"]

    listed = client.get("/api/v1/artifacts/", params={"project_path": str(_general_project_path())})
    assert listed.status_code == 200, listed.text
    card = next(item for item in listed.json() if item["id"] == artifact_id)
    assert card["publishedUrl"] == published_url
    assert card["publishedVersionId"] == published_version_id
    assert card["publishedFilesHash"] == payload["publishedFilesHash"]

    mounted = client.post("/api/v1/artifacts/preview-mount", json={"path": str(folder / "index.html")})
    assert mounted.status_code == 200, mounted.text
    assert mounted.json()["publishedUrl"] == published_url
    assert mounted.json()["publishedVersionId"] == published_version_id


def test_publish_sidecar_persistence_failure_does_not_record_published_deployment(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    from cowork.services import publish as publish_service

    artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Sidecar failure</h1>\n"},
        artifact_type="html-app",
    )
    published_url = f"https://4nton.ai/p/{artifact_id}-sidecar-fail"

    class FakeVault:
        def __init__(self, path):
            self.path = path

    def fake_publish(*args, **kwargs):
        return {"view_url": published_url, "report_id": "report-sidecar-fail", "md5": "sidecar-md5"}

    def fail_publish_record(path, payload):
        raise RuntimeError("disk full")

    monkeypatch.setattr(
        publish_service,
        "get_user_settings",
        lambda: types.SimpleNamespace(minds_api_key="test-key", publish_url="https://4nton.ai"),
    )
    monkeypatch.setattr(publish_service, "_write_publish_record", fail_publish_record)

    monkeypatch.setitem(sys.modules, "anton", types.ModuleType("anton"))
    monkeypatch.setitem(sys.modules, "anton.core", types.ModuleType("anton.core"))
    monkeypatch.setitem(sys.modules, "anton.core.datasources", types.ModuleType("anton.core.datasources"))
    data_vault_mod = types.ModuleType("anton.core.datasources.data_vault")
    data_vault_mod.LocalDataVault = FakeVault
    monkeypatch.setitem(sys.modules, "anton.core.datasources.data_vault", data_vault_mod)
    publisher_mod = types.ModuleType("anton.publisher")
    publisher_mod.publish = fake_publish
    monkeypatch.setitem(sys.modules, "anton.publisher", publisher_mod)

    response = client.post("/api/v1/publish/", json={"path": str(folder)})
    assert response.status_code == 502, response.text
    assert not (folder / ".published.json").exists()

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        artifact = session.exec(select(Artifact).where(Artifact.path == str(folder.resolve()))).one()
        deployments = session.exec(
            select(ArtifactDeployment).where(ArtifactDeployment.artifact_id == artifact.id)
        ).all()
        assert not any(deployment.status == "published" for deployment in deployments)
        assert [deployment.status for deployment in deployments] == ["failed"]


def test_publish_uses_materialized_version_snapshot_not_live_folder(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    from cowork.services import publish as publish_service

    artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Snapshot source</h1>\n"},
        artifact_type="html-app",
    )
    published_url = f"https://4nton.ai/p/{artifact_id}-snapshot"
    observed: dict[str, str] = {}

    class FakeVault:
        def __init__(self, path):
            self.path = path

    def fake_publish(source, *args, **kwargs):
        (folder / "index.html").write_text("<h1>Late live mutation</h1>\n", encoding="utf-8")
        source_path = Path(source)
        published_file = source_path / "index.html" if source_path.is_dir() else source_path
        observed["content"] = published_file.read_text(encoding="utf-8")
        observed["source"] = str(source_path)
        return {"view_url": published_url, "report_id": "report-snapshot", "md5": "snapshot-md5"}

    monkeypatch.setattr(
        publish_service,
        "get_user_settings",
        lambda: types.SimpleNamespace(minds_api_key="test-key", publish_url="https://4nton.ai"),
    )

    monkeypatch.setitem(sys.modules, "anton", types.ModuleType("anton"))
    monkeypatch.setitem(sys.modules, "anton.core", types.ModuleType("anton.core"))
    monkeypatch.setitem(sys.modules, "anton.core.datasources", types.ModuleType("anton.core.datasources"))
    data_vault_mod = types.ModuleType("anton.core.datasources.data_vault")
    data_vault_mod.LocalDataVault = FakeVault
    monkeypatch.setitem(sys.modules, "anton.core.datasources.data_vault", data_vault_mod)
    publisher_mod = types.ModuleType("anton.publisher")
    publisher_mod.publish = fake_publish
    monkeypatch.setitem(sys.modules, "anton.publisher", publisher_mod)

    response = client.post("/api/v1/publish/", json={"path": str(folder)})
    assert response.status_code == 200, response.text
    assert observed["content"] == "<h1>Snapshot source</h1>\n"
    assert observed["source"] != str(folder)

    sidecar = json.loads((folder / ".published.json").read_text(encoding="utf-8"))
    assert sidecar["index.html"]["version_id"] == response.json()["publishedVersionId"]


def test_publish_can_upload_selected_historical_version(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    from cowork.services import publish as publish_service

    artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Saved version</h1>\n"},
        artifact_type="html-app",
    )
    first = _checkpoint(client, artifact_id, label="Saved")
    (folder / "index.html").write_text("<h1>Current draft</h1>\n", encoding="utf-8")
    _checkpoint(client, artifact_id, label="Current")
    published_url = f"https://4nton.ai/p/{artifact_id}-selected"
    observed: dict[str, str] = {}

    class FakeVault:
        def __init__(self, path):
            self.path = path

    def fake_publish(source, *args, **kwargs):
        source_path = Path(source)
        published_file = source_path / "index.html" if source_path.is_dir() else source_path
        observed["content"] = published_file.read_text(encoding="utf-8")
        observed["source"] = str(source_path)
        return {"view_url": published_url, "report_id": "report-selected", "md5": "selected-md5"}

    monkeypatch.setattr(
        publish_service,
        "get_user_settings",
        lambda: types.SimpleNamespace(minds_api_key="test-key", publish_url="https://4nton.ai"),
    )

    monkeypatch.setitem(sys.modules, "anton", types.ModuleType("anton"))
    monkeypatch.setitem(sys.modules, "anton.core", types.ModuleType("anton.core"))
    monkeypatch.setitem(sys.modules, "anton.core.datasources", types.ModuleType("anton.core.datasources"))
    data_vault_mod = types.ModuleType("anton.core.datasources.data_vault")
    data_vault_mod.LocalDataVault = FakeVault
    monkeypatch.setitem(sys.modules, "anton.core.datasources.data_vault", data_vault_mod)
    publisher_mod = types.ModuleType("anton.publisher")
    publisher_mod.publish = fake_publish
    monkeypatch.setitem(sys.modules, "anton.publisher", publisher_mod)

    response = client.post(
        "/api/v1/publish/",
        json={"path": str(folder), "versionId": first["version"]["id"]},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert observed["content"] == "<h1>Saved version</h1>\n"
    assert observed["source"] != str(folder)
    assert payload["publishedVersionId"] == first["version"]["id"]

    sidecar = json.loads((folder / ".published.json").read_text(encoding="utf-8"))
    assert sidecar["index.html"]["version_id"] == first["version"]["id"]
    assert sidecar["index.html"]["files_hash"] == first["version"]["filesHash"]

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        deployment = session.exec(
            select(ArtifactDeployment).where(ArtifactDeployment.url == published_url)
        ).one()
        assert str(deployment.version_id) == first["version"]["id"]


def test_failed_selected_version_publish_does_not_mutate_current_draft(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    from cowork.api.v1.endpoints import publish as publish_endpoint

    artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Known good</h1>\n"},
        artifact_type="html-app",
    )
    good_url = f"https://4nton.ai/p/{artifact_id}-good"
    monkeypatch.setattr(
        publish_endpoint,
        "_publish",
        lambda path, password=None, access=None, **kwargs: {
            "url": good_url,
            "publishedUrl": good_url,
            "access": {"mode": "public"},
        },
    )
    good_response = client.post("/api/v1/publish/", json={"path": str(folder)})
    assert good_response.status_code == 200, good_response.text
    good_version_id = good_response.json()["publishedVersionId"]

    (folder / "index.html").write_text("<h1>Current draft</h1>\n", encoding="utf-8")
    current = _checkpoint(client, artifact_id, label="Current draft")

    def fail_publish(path, password=None, access=None, **kwargs):
        raise RuntimeError("selected upload failed")

    monkeypatch.setattr(publish_endpoint, "_publish", fail_publish)
    failed_response = client.post(
        "/api/v1/publish/",
        json={"path": str(folder), "versionId": good_version_id},
    )
    assert failed_response.status_code == 502, failed_response.text
    assert (folder / "index.html").read_text(encoding="utf-8") == "<h1>Current draft</h1>\n"

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        artifact = session.exec(select(Artifact).where(Artifact.path == str(folder.resolve()))).one()
        failed_deployment = session.exec(
            select(ArtifactDeployment)
            .where(ArtifactDeployment.artifact_id == artifact.id)
            .where(ArtifactDeployment.status == "failed")
        ).one()

    assert artifact.current_version_id == UUID(current["version"]["id"])
    assert artifact.last_known_good_version_id == UUID(good_version_id)
    assert failed_deployment.version_id == UUID(good_version_id)
    assert failed_deployment.details["error"] == "selected upload failed"


def test_failed_publish_records_failure_and_rolls_back_current_version(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    from cowork.api.v1.endpoints import publish as publish_endpoint

    artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Good</h1>\n"},
        artifact_type="html-app",
    )
    good_url = f"https://4nton.ai/p/{artifact_id}-good"
    monkeypatch.setattr(
        publish_endpoint,
        "_publish",
        lambda path, password=None, access=None: {
            "url": good_url,
            "publishedUrl": good_url,
            "access": {"mode": "public"},
        },
    )
    good_response = client.post("/api/v1/publish/", json={"path": str(folder)})
    assert good_response.status_code == 200, good_response.text
    good_payload = good_response.json()

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        good_deployment = session.exec(
            select(ArtifactDeployment).where(ArtifactDeployment.url == good_url)
        ).one()
        good_version_id = good_deployment.version_id
        artifact_id_internal = good_deployment.artifact_id

    good_sidecar = {
        "index.html": {
            "url": good_url,
            "version_id": str(good_version_id),
            "files_hash": good_payload["publishedFilesHash"],
            "manifest_hash": good_payload["publishedManifestHash"],
            "version_number": good_payload["publishedVersionNumber"],
        }
    }
    (folder / ".published.json").write_text(json.dumps(good_sidecar), encoding="utf-8")

    (folder / "index.html").write_text("<h1>Broken publish</h1>\n", encoding="utf-8")

    def fail_publish(path, password=None, access=None):
        raise RuntimeError("upload failed")

    monkeypatch.setattr(publish_endpoint, "_publish", fail_publish)
    failed_response = client.post("/api/v1/publish/", json={"path": str(folder)})
    assert failed_response.status_code == 502, failed_response.text

    with Session(engine) as session:
        artifact = session.get(Artifact, artifact_id_internal)
        failed_deployment = session.exec(
            select(ArtifactDeployment)
            .where(ArtifactDeployment.artifact_id == artifact_id_internal)
            .where(ArtifactDeployment.status == "failed")
        ).one()
        failed_version = session.get(ArtifactVersion, failed_deployment.version_id)

        assert artifact is not None
        assert artifact.current_version_id == good_version_id
        assert artifact.last_known_good_version_id == good_version_id
        assert failed_version is not None
        assert failed_version.publish_status == "failed"
        assert failed_deployment.details["error"] == "upload failed"
    assert (folder / "index.html").read_text(encoding="utf-8") == "<h1>Good</h1>\n"
    assert json.loads((folder / ".published.json").read_text(encoding="utf-8")) == good_sidecar

    listed = client.get("/api/v1/artifacts/", params={"project_path": str(_general_project_path())})
    assert listed.status_code == 200, listed.text
    card = next(item for item in listed.json() if item["id"] == artifact_id)
    assert card["publishedUrl"] == good_url
    assert card["publishedVersionId"] == str(good_version_id)
    assert card["publishedFilesHash"] == good_payload["publishedFilesHash"]

    mounted = client.post("/api/v1/artifacts/preview-mount", json={"path": str(folder / "index.html")})
    assert mounted.status_code == 200, mounted.text
    assert mounted.json()["publishedUrl"] == good_url
    assert mounted.json()["publishedVersionId"] == str(good_version_id)
    assert mounted.json()["publishedFilesHash"] == good_payload["publishedFilesHash"]


def test_failed_publish_keeps_failed_current_when_rollback_materialization_fails(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    from cowork.api.v1.endpoints import publish as publish_endpoint

    artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Good</h1>\n"},
        artifact_type="html-app",
    )
    good_url = f"https://4nton.ai/p/{artifact_id}-good"
    monkeypatch.setattr(
        publish_endpoint,
        "_publish",
        lambda path, password=None, access=None: {
            "url": good_url,
            "publishedUrl": good_url,
            "access": {"mode": "public"},
        },
    )
    good_response = client.post("/api/v1/publish/", json={"path": str(folder)})
    assert good_response.status_code == 200, good_response.text

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        good_deployment = session.exec(
            select(ArtifactDeployment).where(ArtifactDeployment.url == good_url)
        ).one()
        good_version_id = good_deployment.version_id
        artifact_id_internal = good_deployment.artifact_id

    (folder / "index.html").write_text("<h1>Broken publish</h1>\n", encoding="utf-8")

    def fail_publish(path, password=None, access=None):
        raise RuntimeError("upload failed")

    original_replace = ArtifactVersionService.replace_with_version

    def fail_live_rollback(self, version_id, target_dir, **kwargs):
        if Path(target_dir).resolve(strict=False) == folder.resolve():
            raise RuntimeError("rollback copy failed")
        return original_replace(self, version_id, target_dir, **kwargs)

    monkeypatch.setattr(publish_endpoint, "_publish", fail_publish)
    monkeypatch.setattr(ArtifactVersionService, "replace_with_version", fail_live_rollback)

    failed_response = client.post("/api/v1/publish/", json={"path": str(folder)})
    assert failed_response.status_code == 502, failed_response.text

    with Session(engine) as session:
        artifact = session.get(Artifact, artifact_id_internal)
        failed_deployment = session.exec(
            select(ArtifactDeployment)
            .where(ArtifactDeployment.artifact_id == artifact_id_internal)
            .where(ArtifactDeployment.status == "failed")
        ).one()
        failed_version = session.get(ArtifactVersion, failed_deployment.version_id)

        assert artifact is not None
        assert failed_version is not None
        assert artifact.current_version_id == failed_version.id
        assert artifact.last_known_good_version_id == good_version_id
        assert failed_version.publish_status == "failed"
        assert failed_deployment.details["error"] == "upload failed"
        assert failed_deployment.details["rollbackError"] == "rollback copy failed"
    assert (folder / "index.html").read_text(encoding="utf-8") == "<h1>Broken publish</h1>\n"


def test_owned_project_publish_and_unpublish_require_editor_identity(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    from cowork.api.v1.endpoints import publish as publish_endpoint

    _own_general_project_for_versions()
    artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Private publish</h1>\n"},
        artifact_type="html-app",
    )
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.publish.principal_from_authorization_header",
        _fake_version_principal,
    )
    monkeypatch.setattr(
        publish_endpoint,
        "_publish",
        lambda path, password=None, access=None: {
            "url": f"https://4nton.ai/p/{artifact_id}",
            "publishedUrl": f"https://4nton.ai/p/{artifact_id}",
            "access": {"mode": "public"},
        },
    )
    monkeypatch.setattr(publish_endpoint, "_unpublish", lambda path: {"status": "ok"})

    anonymous = client.post("/api/v1/publish/", json={"path": str(folder)})
    assert anonymous.status_code == 401

    viewer = client.post(
        "/api/v1/publish/",
        headers={"Authorization": "Bearer viewer"},
        json={"path": str(folder)},
    )
    assert viewer.status_code == 403

    editor = client.post(
        "/api/v1/publish/",
        headers={"Authorization": "Bearer editor"},
        json={"path": str(folder)},
    )
    assert editor.status_code == 200, editor.text

    viewer_unpublish = client.delete(
        "/api/v1/publish/",
        headers={"Authorization": "Bearer viewer"},
        params={"path": str(folder)},
    )
    assert viewer_unpublish.status_code == 403

    editor_unpublish = client.delete(
        "/api/v1/publish/",
        headers={"Authorization": "Bearer editor"},
        params={"path": str(folder)},
    )
    assert editor_unpublish.status_code == 200, editor_unpublish.text


def test_owned_project_open_and_reveal_require_viewer_identity(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    from cowork.api.v1.endpoints import artifacts as artifacts_endpoint

    _own_general_project_for_versions()
    _artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Private local action</h1>\n"},
        artifact_type="html-app",
    )
    opened: list[str] = []
    revealed: list[str] = []

    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifacts.principal_from_authorization_header",
        _fake_version_principal,
    )
    monkeypatch.setattr(
        artifacts_endpoint.subprocess,
        "run",
        lambda args, check=False: opened.append(str(args[-1])),
    )
    monkeypatch.setattr(
        artifacts_endpoint,
        "reveal_in_file_manager",
        lambda path: revealed.append(str(path)),
    )

    anonymous_open = client.post("/api/v1/artifacts/open", json={"path": str(folder)})
    assert anonymous_open.status_code == 401
    assert opened == []

    viewer_open = client.post(
        "/api/v1/artifacts/open",
        headers={"Authorization": "Bearer viewer"},
        json={"path": str(folder)},
    )
    assert viewer_open.status_code == 200, viewer_open.text
    assert opened == [str(folder.resolve())]

    anonymous_reveal = client.post("/api/v1/artifacts/reveal", json={"path": str(folder / "index.html")})
    assert anonymous_reveal.status_code == 401
    assert revealed == []

    viewer_reveal = client.post(
        "/api/v1/artifacts/reveal",
        headers={"Authorization": "Bearer viewer"},
        json={"path": str(folder / "index.html")},
    )
    assert viewer_reveal.status_code == 200, viewer_reveal.text
    assert revealed == [str((folder / "index.html").resolve())]


def test_publish_listing_is_scoped_to_viewable_projects(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    _own_general_project_for_versions()
    _artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Publishable</h1>\n"},
        artifact_type="html-app",
    )
    path = str((folder / "index.html").resolve())
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.publish.principal_from_authorization_header",
        _fake_version_principal,
    )

    anonymous = client.get("/api/v1/publish/")
    assert anonymous.status_code == 200, anonymous.text
    assert path not in {str(Path(item.get("path")).resolve()) for item in anonymous.json()["artifacts"]}

    viewer = client.get("/api/v1/publish/", headers={"Authorization": "Bearer viewer"})
    assert viewer.status_code == 200, viewer.text
    assert path in {str(Path(item.get("path")).resolve()) for item in viewer.json()["artifacts"]}


def test_chat_publish_tool_records_deployment_against_version(monkeypatch: pytest.MonkeyPatch):
    from cowork.harnesses.anton_harness import tools as anton_tools

    artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Chat publish</h1>\n"},
        artifact_type="html-app",
    )
    published_url = f"https://4nton.ai/p/{artifact_id}-chat"

    def fake_publish(path, *, api_key, report_id=None, publish_url=None, ssl_verify=True):
        return {"view_url": published_url, "report_id": f"report-{artifact_id}", "md5": "abc123"}

    monkeypatch.setenv("ANTON_MINDS_API_KEY", "test-key")
    monkeypatch.setattr("anton.publisher.publish", fake_publish)

    result = asyncio.run(
        anton_tools._cowork_publish_or_preview(
            object(),
            {"file_path": str(folder / "index.html"), "title": "Chat publish", "action": "publish"},
        )
    )

    assert published_url in result
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        deployment = session.exec(
            select(ArtifactDeployment)
            .join(ArtifactVersion, ArtifactDeployment.version_id == ArtifactVersion.id)
            .where(ArtifactVersion.operation_type == "publish")
            .where(ArtifactDeployment.url == published_url)
        ).one()
        version = session.get(ArtifactVersion, deployment.version_id)

    assert version is not None
    assert version.publish_status == "published"
    assert deployment.status == "published"


def test_chat_publish_tool_uploads_materialized_version_not_live_file(monkeypatch: pytest.MonkeyPatch):
    from cowork.harnesses.anton_harness import tools as anton_tools

    artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Chat snapshot source</h1>\n"},
        artifact_type="html-app",
    )
    published_url = f"https://4nton.ai/p/{artifact_id}-chat-snapshot"
    observed: dict[str, str] = {}

    def fake_publish(path, *, api_key, report_id=None, publish_url=None, ssl_verify=True):
        (folder / "index.html").write_text("<h1>Late chat mutation</h1>\n", encoding="utf-8")
        source = Path(path)
        observed["source"] = str(source)
        observed["content"] = source.read_text(encoding="utf-8")
        return {"view_url": published_url, "report_id": f"report-{artifact_id}", "md5": "abc123"}

    monkeypatch.setenv("ANTON_MINDS_API_KEY", "test-key")
    monkeypatch.setattr("anton.publisher.publish", fake_publish)

    result = asyncio.run(
        anton_tools._cowork_publish_or_preview(
            object(),
            {"file_path": str(folder / "index.html"), "title": "Chat publish", "action": "publish"},
        )
    )

    assert published_url in result
    assert observed["source"] != str(folder / "index.html")
    assert observed["content"] == "<h1>Chat snapshot source</h1>\n"
    sidecar = json.loads((folder / ".published.json").read_text(encoding="utf-8"))
    assert sidecar["index.html"]["version_id"]
    assert sidecar["index.html"]["files_hash"]
    assert sidecar["index.html"]["manifest_hash"]
    assert sidecar["index.html"]["version_number"] == 1


def test_chat_publish_tool_preserves_prior_access(monkeypatch: pytest.MonkeyPatch):
    """Re-publishing a password-protected artifact from chat must keep it
    protected, not silently downgrade it to public."""
    from cowork.harnesses.anton_harness import tools as anton_tools

    artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Protected</h1>\n"},
        artifact_type="html-app",
    )
    # Seed a prior password-protected publish record (as the GUI path writes it).
    (folder / ".published.json").write_text(
        json.dumps({
            "index.html": {
                "report_id": f"report-{artifact_id}",
                "url": f"https://4nton.ai/p/{artifact_id}",
                "mode": "password",
                "requires_password": True,
                "access_password": "s3cret",
                "pwd_version": 2,
            }
        }),
        encoding="utf-8",
    )

    captured: dict = {}

    def fake_publish(path, *, api_key, report_id=None, publish_url=None, ssl_verify=True,
                     access=None, access_version=None, pwd_version=None):
        captured["access"] = access
        captured["report_id"] = report_id
        return {"view_url": f"https://4nton.ai/p/{artifact_id}", "report_id": f"report-{artifact_id}", "md5": "z"}

    monkeypatch.setenv("ANTON_MINDS_API_KEY", "test-key")
    monkeypatch.setattr("anton.publisher.publish", fake_publish)

    result = asyncio.run(
        anton_tools._cowork_publish_or_preview(
            object(),
            {"file_path": str(folder / "index.html"), "title": "Protected", "action": "publish"},
        )
    )

    assert "4nton.ai" in result
    # Protection carried through to the publisher (not reset to public)...
    assert captured["access"] == {"mode": "password", "password": "s3cret"}
    assert captured["report_id"] == f"report-{artifact_id}"  # report_id reused
    # ...and persisted in the sidecar so the artifact stays locked.
    sidecar = json.loads((folder / ".published.json").read_text(encoding="utf-8"))["index.html"]
    assert sidecar["mode"] == "password"
    assert sidecar["access_password"] == "s3cret"
    assert sidecar["published"] is True


def test_chat_publish_aborts_if_version_source_cannot_materialize(monkeypatch: pytest.MonkeyPatch):
    from cowork.harnesses.anton_harness import tools as anton_tools

    artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Chat materialize failure</h1>\n"},
        artifact_type="html-app",
    )
    called = {"publish": False}

    def fake_publish(path, *, api_key, report_id=None, publish_url=None, ssl_verify=True):
        called["publish"] = True
        return {"view_url": f"https://4nton.ai/p/{artifact_id}", "report_id": f"report-{artifact_id}"}

    def fail_materialize(self, version_id, target_dir, *, clean=True):
        raise FileNotFoundError("missing blob")

    monkeypatch.setenv("ANTON_MINDS_API_KEY", "test-key")
    monkeypatch.setattr("anton.publisher.publish", fake_publish)
    monkeypatch.setattr(
        "cowork.services.artifact_versions.ArtifactVersionService.materialize_version",
        fail_materialize,
    )

    result = asyncio.run(
        anton_tools._cowork_publish_or_preview(
            object(),
            {"file_path": str(folder / "index.html"), "title": "Chat publish", "action": "publish"},
        )
    )

    assert "PUBLISH FAILED: Could not prepare the versioned artifact for publishing" in result
    assert called["publish"] is False
    assert not (folder / ".published.json").exists()
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        artifact = session.exec(select(Artifact).where(Artifact.path == str(folder.resolve()))).one()
        failed_deployment = session.exec(
            select(ArtifactDeployment)
            .where(ArtifactDeployment.artifact_id == artifact.id)
            .where(ArtifactDeployment.status == "failed")
        ).one()
        failed_version = session.get(ArtifactVersion, failed_deployment.version_id)

    assert failed_version is not None
    assert failed_version.publish_status == "failed"
    assert failed_deployment.details["error"] == "Could not prepare the versioned artifact for publishing"


def test_chat_failed_publish_rolls_back_live_files_to_last_good(monkeypatch: pytest.MonkeyPatch):
    from cowork.harnesses.anton_harness import tools as anton_tools

    artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Chat good</h1>\n"},
        artifact_type="html-app",
    )
    good_url = f"https://4nton.ai/p/{artifact_id}-chat-good"

    def publish_good(path, *, api_key, report_id=None, publish_url=None, ssl_verify=True):
        return {"view_url": good_url, "report_id": f"report-{artifact_id}", "md5": "good"}

    monkeypatch.setenv("ANTON_MINDS_API_KEY", "test-key")
    monkeypatch.setattr("anton.publisher.publish", publish_good)

    good_result = asyncio.run(
        anton_tools._cowork_publish_or_preview(
            object(),
            {"file_path": str(folder / "index.html"), "title": "Chat publish", "action": "publish"},
        )
    )
    assert good_url in good_result

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        good_deployment = session.exec(
            select(ArtifactDeployment).where(ArtifactDeployment.url == good_url)
        ).one()
        good_version_id = good_deployment.version_id
        artifact_internal_id = good_deployment.artifact_id

    (folder / "index.html").write_text("<h1>Chat broken</h1>\n", encoding="utf-8")

    def publish_failure(path, *, api_key, report_id=None, publish_url=None, ssl_verify=True):
        raise RuntimeError("chat upload failed")

    monkeypatch.setattr("anton.publisher.publish", publish_failure)

    failed_result = asyncio.run(
        anton_tools._cowork_publish_or_preview(
            object(),
            {"file_path": str(folder / "index.html"), "title": "Chat publish", "action": "publish"},
        )
    )

    assert "PUBLISH FAILED" in failed_result
    assert (folder / "index.html").read_text(encoding="utf-8") == "<h1>Chat good</h1>\n"
    with Session(engine) as session:
        artifact = session.get(Artifact, artifact_internal_id)
        failed_deployment = session.exec(
            select(ArtifactDeployment)
            .where(ArtifactDeployment.artifact_id == artifact_internal_id)
            .where(ArtifactDeployment.status == "failed")
        ).one()
        failed_version = session.get(ArtifactVersion, failed_deployment.version_id)

    assert artifact is not None
    assert artifact.current_version_id == good_version_id
    assert artifact.last_known_good_version_id == good_version_id
    assert failed_version is not None
    assert failed_version.publish_status == "failed"
    assert failed_deployment.details["error"] == "chat upload failed"


def test_chat_failed_publish_keeps_failed_current_when_rollback_materialization_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    from cowork.harnesses.anton_harness import tools as anton_tools

    artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Chat good</h1>\n"},
        artifact_type="html-app",
    )
    good_url = f"https://4nton.ai/p/{artifact_id}-chat-good"

    def publish_good(path, *, api_key, report_id=None, publish_url=None, ssl_verify=True):
        return {"view_url": good_url, "report_id": f"report-{artifact_id}", "md5": "good"}

    monkeypatch.setenv("ANTON_MINDS_API_KEY", "test-key")
    monkeypatch.setattr("anton.publisher.publish", publish_good)

    good_result = asyncio.run(
        anton_tools._cowork_publish_or_preview(
            object(),
            {"file_path": str(folder / "index.html"), "title": "Chat publish", "action": "publish"},
        )
    )
    assert good_url in good_result

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        good_deployment = session.exec(
            select(ArtifactDeployment).where(ArtifactDeployment.url == good_url)
        ).one()
        good_version_id = good_deployment.version_id
        artifact_internal_id = good_deployment.artifact_id

    (folder / "index.html").write_text("<h1>Chat broken</h1>\n", encoding="utf-8")

    def publish_failure(path, *, api_key, report_id=None, publish_url=None, ssl_verify=True):
        raise RuntimeError("chat upload failed")

    original_replace = ArtifactVersionService.replace_with_version

    def fail_live_rollback(self, version_id, target_dir, **kwargs):
        if Path(target_dir).resolve(strict=False) == folder.resolve():
            raise RuntimeError("chat rollback copy failed")
        return original_replace(self, version_id, target_dir, **kwargs)

    monkeypatch.setattr("anton.publisher.publish", publish_failure)
    monkeypatch.setattr(ArtifactVersionService, "replace_with_version", fail_live_rollback)

    failed_result = asyncio.run(
        anton_tools._cowork_publish_or_preview(
            object(),
            {"file_path": str(folder / "index.html"), "title": "Chat publish", "action": "publish"},
        )
    )

    assert "PUBLISH FAILED" in failed_result
    assert (folder / "index.html").read_text(encoding="utf-8") == "<h1>Chat broken</h1>\n"
    with Session(engine) as session:
        artifact = session.get(Artifact, artifact_internal_id)
        failed_deployment = session.exec(
            select(ArtifactDeployment)
            .where(ArtifactDeployment.artifact_id == artifact_internal_id)
            .where(ArtifactDeployment.status == "failed")
        ).one()
        failed_version = session.get(ArtifactVersion, failed_deployment.version_id)

    assert artifact is not None
    assert failed_version is not None
    assert artifact.current_version_id == failed_version.id
    assert artifact.last_known_good_version_id == good_version_id
    assert failed_version.publish_status == "failed"
    assert failed_deployment.details["error"] == "chat upload failed"
    assert failed_deployment.details["rollbackError"] == "chat rollback copy failed"


def test_failed_publish_without_rollback_target_keeps_failed_checkpoint_current():
    from cowork.harnesses.anton_harness import tools as anton_tools

    _artifact_id, folder = _make_artifact(
        files={"index.html": "<h1>Chat first publish</h1>\n"},
        artifact_type="html-app",
    )

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        version = ArtifactVersionService(session).snapshot_artifact(
            folder,
            operation_type="publish",
            label="Published version",
        )
        version_id = version.id
        artifact_id = version.artifact_id
        session.commit()

    anton_tools._record_publish_result(version_id, status="failed", details={"error": "first upload failed"})

    with Session(engine) as session:
        artifact = session.get(Artifact, artifact_id)
        failed_deployment = session.exec(
            select(ArtifactDeployment)
            .where(ArtifactDeployment.artifact_id == artifact_id)
            .where(ArtifactDeployment.status == "failed")
        ).one()
        failed_version = session.get(ArtifactVersion, failed_deployment.version_id)

    assert artifact is not None
    assert failed_version is not None
    assert artifact.current_version_id == failed_version.id
    assert artifact.last_known_good_version_id is None
    assert failed_version.publish_status == "failed"
    assert failed_deployment.details["error"] == "first upload failed"


def test_artifact_handoff_creates_conversation_with_version_and_comment_context(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    captured = []

    async def fake_handle(self, request):
        captured.append(request)
        return object()

    monkeypatch.setattr("cowork.services.artifact_handoff.ResponsesHandler.handle", fake_handle)
    artifact_id, folder = _make_artifact(
        files={"report.md": "# Draft\nSummary: rough\n"},
        artifact_type="document",
    )
    checkpoint = _checkpoint(client, artifact_id, path=str(folder), label="Review baseline")
    (folder / "report.md").write_text("# Draft\nSummary: live drift\n", encoding="utf-8")
    comment_response = client.post(
        "/api/v1/artifacts/comments",
        json={"path": str(folder), "body": "Tighten the summary.", "kind": "suggestion"},
    )
    assert comment_response.status_code == 201, comment_response.text
    comment = comment_response.json()["comment"]

    response = client.post(
        "/api/v1/artifacts/handoff",
        json={
            "path": str(folder),
            "versionId": checkpoint["version"]["id"],
            "commentId": comment["id"],
            "prompt": "Revise this artifact from the note.",
            "title": "Revise artifact",
        },
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["started"] is True
    assert payload["conversationId"]
    assert payload["version"]["id"] == checkpoint["version"]["id"]
    assert payload["comment"]["id"] == comment["id"]
    assert payload["materializedVersion"] is True
    assert payload["handoffPath"] != str(folder)
    handoff_folder = Path(payload["handoffPath"])
    assert handoff_folder.parent.name == "artifacts"
    assert (handoff_folder / "report.md").read_text(encoding="utf-8") == "# Draft\nSummary: rough\n"
    assert captured and captured[0].stream is True
    assert captured[0].conversation == payload["conversationId"]
    assert "Revise this artifact from the note." in captured[0].input
    assert "Selected version:" in captured[0].input
    assert "Selected review note:" in captured[0].input
    assert payload["handoffPath"] in captured[0].input

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        conversation = session.get(Conversation, UUID(payload["conversationId"]))
        assert conversation is not None
        assert conversation.topic == "Revise artifact"
        copied_artifact = session.exec(
            select(Artifact).where(Artifact.path == str(handoff_folder.resolve(strict=False)))
        ).one()
        copied_versions = session.exec(
            select(ArtifactVersion)
            .where(ArtifactVersion.artifact_id == copied_artifact.id)
            .order_by(ArtifactVersion.version_number)
        ).all()

    assert copied_artifact.slug == handoff_folder.name
    assert copied_artifact.current_version_id == copied_versions[-1].id
    assert len(copied_versions) == 1
    assert copied_versions[0].operation_type == "fork"
    assert copied_versions[0].forked_from_version_id == UUID(checkpoint["version"]["id"])
    assert copied_versions[0].source_conversation_id == UUID(payload["conversationId"])


def test_artifact_handoff_allows_source_viewer_and_records_activity(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    captured = []

    async def fake_handle(self, request):
        captured.append(request)
        return object()

    monkeypatch.setattr("cowork.services.artifact_handoff.ResponsesHandler.handle", fake_handle)
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifacts.principal_from_authorization_header",
        _fake_version_principal,
    )
    _own_general_project_for_versions()
    artifact_id, folder = _make_artifact(files={"report.md": "# Viewer source\n"}, artifact_type="document")

    response = client.post(
        "/api/v1/artifacts/handoff",
        headers={"Authorization": "Bearer viewer"},
        json={
            "path": str(folder),
            "prompt": "Turn this review into a follow-up task.",
            "title": "Viewer follow-up",
        },
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["conversation"]["title"] == "Viewer follow-up"
    assert captured and captured[0].conversation == payload["conversationId"]

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        artifact = session.exec(select(Artifact).where(Artifact.path == str(folder.resolve()))).one()
        event = session.exec(
            select(ArtifactActivityEvent)
            .where(ArtifactActivityEvent.artifact_id == artifact.id)
            .where(ArtifactActivityEvent.event_type == "handoff")
        ).one()
    assert event.actor_name == "Viewer"
    assert event.details["conversationId"] == payload["conversationId"]
    assert event.details["actorEmail"] == "viewer@example.com"
    assert event.details["materializedVersion"] is False


def test_artifact_handoff_rejects_comment_from_another_artifact(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_handle(self, request):
        return object()

    monkeypatch.setattr("cowork.services.artifact_handoff.ResponsesHandler.handle", fake_handle)
    _artifact_id, first = _make_artifact(files={"report.md": "first"}, artifact_type="document")
    _other_id, second = _make_artifact(files={"report.md": "second"}, artifact_type="document")
    comment_response = client.post(
        "/api/v1/artifacts/comments",
        json={"path": str(first), "body": "Only applies to the first artifact."},
    )
    assert comment_response.status_code == 201, comment_response.text

    response = client.post(
        "/api/v1/artifacts/handoff",
        json={"path": str(second), "commentId": comment_response.json()["comment"]["id"]},
    )

    assert response.status_code == 404


def test_artifact_handoff_uses_target_project_name(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    captured = []

    async def fake_handle(self, request):
        captured.append(request)
        return object()

    monkeypatch.setattr("cowork.services.artifact_handoff.ResponsesHandler.handle", fake_handle)
    artifact_id, folder = _make_artifact(files={"report.md": "target me\n"}, artifact_type="document")
    checkpoint = _checkpoint(client, artifact_id, path=str(folder), label="Target baseline")
    target_path = tmp_path / "named-target"
    target_path.mkdir()
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        target = Project(name=f"named-target-{uuid4().hex}", path=str(target_path))
        session.add(target)
        session.commit()
        session.refresh(target)
        target_id = target.id
        target_name = target.name

    response = client.post(
        "/api/v1/artifacts/handoff",
        json={
            "path": str(folder),
            "versionId": checkpoint["version"]["id"],
            "project": target_name,
            "title": "Named target handoff",
        },
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    handoff_folder = Path(payload["handoffPath"])
    assert handoff_folder.parent == target_path / ".anton" / "artifacts"
    assert captured and captured[0].project == target_name
    assert captured[0].project_id == target_id

    with Session(engine) as session:
        conversation = session.get(Conversation, UUID(payload["conversationId"]))
        copied_artifact = session.exec(
            select(Artifact).where(Artifact.path == str(handoff_folder.resolve(strict=False)))
        ).one()

    assert conversation is not None
    assert conversation.project_id == target_id
    assert copied_artifact.project_id == target_id


def test_artifact_handoff_requires_edit_access_to_target_project(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    async def fail_if_called(self, request):
        raise AssertionError("handoff should not start a response without target project access")

    monkeypatch.setattr("cowork.services.artifact_handoff.ResponsesHandler.handle", fail_if_called)
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.artifacts.principal_from_authorization_header",
        _fake_version_principal,
    )
    _artifact_id, folder = _make_artifact(files={"report.md": "# Draft\n"}, artifact_type="document")
    _own_general_project_for_versions()

    target_path = tmp_path / "handoff-target"
    target_path.mkdir()
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        target = Project(name=f"version-test-target-{uuid4().hex[:8]}", path=str(target_path))
        session.add(target)
        session.commit()
        session.refresh(target)
        target_id = target.id
        session.add(ProjectCollaborator(project_id=target_id, email="target-owner@example.com", role="owner"))
        session.commit()

    response = client.post(
        "/api/v1/artifacts/handoff",
        headers={"Authorization": "Bearer editor"},
        json={
            "path": str(folder),
            "projectId": str(target_id),
            "prompt": "Continue this in the target project.",
        },
    )

    assert response.status_code == 403, response.text
    with Session(engine) as session:
        conversations = session.exec(select(Conversation).where(Conversation.project_id == target_id)).all()
        for collaborator in session.exec(
            select(ProjectCollaborator).where(ProjectCollaborator.project_id == target_id)
        ).all():
            session.delete(collaborator)
        project = session.get(Project, target_id)
        if project is not None:
            session.delete(project)
        session.commit()
    assert conversations == []


def test_pathless_duplicate_slug_identifier_is_rejected_for_restore(
    client: TestClient,
    tmp_path: Path,
):
    shared_slug = f"shared-slug-{uuid4().hex}"
    folders: list[Path] = []
    version_ids: list[UUID] = []
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        for index in range(2):
            project_path = tmp_path / f"project-{index}"
            project_path.mkdir(parents=True)
            project = Project(name=f"duplicate-slug-project-{uuid4().hex}", path=str(project_path))
            session.add(project)
            session.commit()
            session.refresh(project)
            folder = project_path / ".anton" / "artifacts" / shared_slug
            folder.mkdir(parents=True)
            (folder / "metadata.json").write_text(
                json.dumps(
                    {
                        "id": f"duplicate-external-{index}-{uuid4().hex}",
                        "slug": shared_slug,
                        "name": f"Duplicate {index}",
                        "type": "document",
                        "primary": "report.md",
                    }
                ),
                encoding="utf-8",
            )
            (folder / "report.md").write_text(f"project {index}\n", encoding="utf-8")
            folders.append(folder)
            version = ArtifactVersionService(session).snapshot_artifact(
                folder,
                project_id=project.id,
            )
            version_ids.append(version.id)

    ambiguous = client.post(
        "/api/v1/artifacts/versions/restore",
        json={"artifactId": shared_slug, "versionId": str(version_ids[0])},
    )
    assert ambiguous.status_code == 400
    assert "ambiguous" in ambiguous.json()["detail"].lower()

    scoped = client.post(
        "/api/v1/artifacts/versions/restore",
        json={"path": str(folders[0]), "versionId": str(version_ids[0])},
    )
    assert scoped.status_code == 200, scoped.text
