from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from cowork.api.v1.router import api_router
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.artifact import Artifact, ArtifactVersion
from cowork.models.project import Project
from cowork.services.request_identity import RequestPrincipal
from cowork.services.projects import GENERAL_PROJECT, GENERAL_PROJECT_ID


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(api_router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_project_file_artifacts():
    project = _general_project_path()
    artifact_root = project / ".anton" / "artifacts"
    if artifact_root.is_dir():
        for folder in artifact_root.glob("project-file-test-*"):
            shutil.rmtree(folder, ignore_errors=True)
    yield
    if artifact_root.is_dir():
        for folder in artifact_root.glob("project-file-test-*"):
            shutil.rmtree(folder, ignore_errors=True)


def _general_project_path() -> Path:
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        project = session.get(Project, GENERAL_PROJECT_ID)
        assert project is not None
        return Path(project.path)


def _make_artifact() -> Path:
    folder = _general_project_path() / ".anton" / "artifacts" / f"project-file-test-{uuid4().hex}"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(
        json.dumps({"slug": folder.name, "name": "Project file artifact", "type": "document"}),
        encoding="utf-8",
    )
    (folder / "report.md").write_text("before", encoding="utf-8")
    return folder


def _versions_for(folder: Path) -> list[ArtifactVersion]:
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        artifact = session.exec(select(Artifact).where(Artifact.path == str(folder.resolve()))).one()
        return session.exec(
            select(ArtifactVersion)
            .where(ArtifactVersion.artifact_id == artifact.id)
            .order_by(ArtifactVersion.version_number)
        ).all()


def _events_for(folder: Path):
    from cowork.models.artifact import ArtifactActivityEvent

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        artifact = session.exec(select(Artifact).where(Artifact.path == str(folder.resolve()))).one()
        return session.exec(
            select(ArtifactActivityEvent)
            .where(ArtifactActivityEvent.artifact_id == artifact.id)
            .order_by(ArtifactActivityEvent.created_at)
        ).all()


def _fake_principal(authorization: str | None):
    if authorization == "Bearer editor":
        return RequestPrincipal("editor-subject", "editor@example.com", "Editor", "test", {})
    return None


def test_project_file_write_snapshots_artifact_folder(client: TestClient):
    folder = _make_artifact()
    rel = folder.relative_to(_general_project_path()).as_posix()

    response = client.put(
        f"/api/v1/projects/{GENERAL_PROJECT}/files/{rel}/report.md",
        json={"content": "after"},
    )

    assert response.status_code == 200, response.text
    assert (folder / "report.md").read_text(encoding="utf-8") == "after"
    assert [version.operation_type for version in _versions_for(folder)] == ["pre_edit", "edit"]


def test_project_file_write_records_editor_identity(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    from cowork.api.v1.endpoints import project_files

    monkeypatch.setattr(project_files, "principal_from_authorization_header", _fake_principal)
    folder = _make_artifact()
    rel = folder.relative_to(_general_project_path()).as_posix()

    response = client.put(
        f"/api/v1/projects/{GENERAL_PROJECT}/files/{rel}/report.md",
        headers={"Authorization": "Bearer editor"},
        json={"content": "after"},
    )

    assert response.status_code == 200, response.text
    versions = _versions_for(folder)
    assert [version.operation_type for version in versions] == ["pre_edit", "edit"]
    events = _events_for(folder)
    assert [event.actor_name for event in events] == ["Editor", "Editor"]
    assert [event.details["actorEmail"] for event in events] == ["editor@example.com", "editor@example.com"]
    assert [event.details["actorSubject"] for event in events] == ["editor-subject", "editor-subject"]


def test_project_file_write_rolls_back_when_post_edit_checkpoint_fails(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    from cowork.api.v1.endpoints import project_files

    folder = _make_artifact()
    rel = folder.relative_to(_general_project_path()).as_posix()
    original_snapshot = project_files._snapshot_artifact
    calls = {"count": 0}

    def flaky_snapshot(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("post edit checkpoint failed")
        return original_snapshot(*args, **kwargs)

    monkeypatch.setattr(project_files, "_snapshot_artifact", flaky_snapshot)

    response = client.put(
        f"/api/v1/projects/{GENERAL_PROJECT}/files/{rel}/report.md",
        json={"content": "after"},
    )

    assert response.status_code == 500
    assert (folder / "report.md").read_text(encoding="utf-8") == "before"
    assert [version.operation_type for version in _versions_for(folder)] == ["pre_edit"]


def test_project_file_delete_snapshots_artifact_folder(client: TestClient):
    folder = _make_artifact()
    rel = folder.relative_to(_general_project_path()).as_posix()

    response = client.delete(f"/api/v1/projects/{GENERAL_PROJECT}/files/{rel}/report.md")

    assert response.status_code == 200, response.text
    assert not (folder / "report.md").exists()
    assert [version.operation_type for version in _versions_for(folder)] == ["pre_delete_file", "delete_file"]


def test_project_file_delete_restores_file_when_post_delete_checkpoint_fails(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    from cowork.api.v1.endpoints import project_files

    folder = _make_artifact()
    rel = folder.relative_to(_general_project_path()).as_posix()
    original_snapshot = project_files._snapshot_artifact
    calls = {"count": 0}

    def flaky_snapshot(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("post delete checkpoint failed")
        return original_snapshot(*args, **kwargs)

    monkeypatch.setattr(project_files, "_snapshot_artifact", flaky_snapshot)

    response = client.delete(f"/api/v1/projects/{GENERAL_PROJECT}/files/{rel}/report.md")

    assert response.status_code == 500
    assert (folder / "report.md").read_text(encoding="utf-8") == "before"
    assert [version.operation_type for version in _versions_for(folder)] == ["pre_delete_file"]


def test_project_file_upload_cannot_overwrite_artifact_file_by_filename_path(client: TestClient):
    folder = _make_artifact()
    project = _general_project_path()
    rel = folder.relative_to(project).as_posix()
    root_upload = project / "report.md"
    root_upload.unlink(missing_ok=True)

    response = client.post(
        f"/api/v1/projects/{GENERAL_PROJECT}/files/upload",
        files=[("files", (f"{rel}/report.md", b"uploaded", "text/markdown"))],
    )

    assert response.status_code == 200, response.text
    assert response.json()["results"] == [{"name": "report.md", "ok": True, "size": len(b"uploaded")}]
    assert (folder / "report.md").read_text(encoding="utf-8") == "before"
    assert root_upload.read_text(encoding="utf-8") == "uploaded"
