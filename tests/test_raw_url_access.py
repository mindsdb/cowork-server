from __future__ import annotations

import hashlib
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
from cowork.models.project import Project
from cowork.models.project_collaboration import ProjectCollaborator
from cowork.services.preview_proxy import _build_upstream_headers
from cowork.services.projects import GENERAL_PROJECT, GENERAL_PROJECT_ID
from cowork.services.request_identity import RequestPrincipal


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(api_router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_raw_url_rows():
    engine = get_engine(get_app_settings().database.uri)
    project = _general_project_path()
    for folder in project.glob("raw-url-test-*"):
        shutil.rmtree(folder, ignore_errors=True)
    with Session(engine) as session:
        for row in session.exec(select(ProjectCollaborator)).all():
            session.delete(row)
        session.commit()
    yield
    for folder in project.glob("raw-url-test-*"):
        shutil.rmtree(folder, ignore_errors=True)
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


def _make_project_html_file() -> tuple[Path, str, str]:
    folder = _general_project_path() / f"raw-url-test-{uuid4().hex}"
    folder.mkdir(parents=True)
    (folder / "index.html").write_text(
        "<!doctype html><link rel=\"stylesheet\" href=\"style.css\"><h1>Private</h1>",
        encoding="utf-8",
    )
    (folder / "style.css").write_text("h1 { color: tomato; }", encoding="utf-8")
    rel_html = folder.relative_to(_general_project_path() / "").as_posix() + "/index.html"
    rel_css = folder.relative_to(_general_project_path() / "").as_posix() + "/style.css"
    return folder, rel_html, rel_css


def _own_general_project() -> None:
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        session.add(ProjectCollaborator(project_id=GENERAL_PROJECT_ID, email="owner@example.com", role="owner"))
        session.add(ProjectCollaborator(project_id=GENERAL_PROJECT_ID, email="viewer@example.com", role="viewer"))
        session.commit()


def _fake_principal(authorization: str | None):
    if authorization == "Bearer viewer":
        return RequestPrincipal("viewer", "viewer@example.com", "Viewer", "test", {})
    if authorization == "Bearer owner":
        return RequestPrincipal("owner", "owner@example.com", "Owner", "test", {})
    return None


def test_owned_project_file_preview_and_download_use_short_lived_grants(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    _own_general_project()
    folder, rel_html, rel_css = _make_project_html_file()
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.project_files.principal_from_authorization_header",
        _fake_principal,
    )

    anonymous_list = client.get(f"/api/v1/projects/{GENERAL_PROJECT}/files")
    assert anonymous_list.status_code == 401

    viewer_list = client.get(
        f"/api/v1/projects/{GENERAL_PROJECT}/files",
        headers={"Authorization": "Bearer viewer"},
    )
    assert viewer_list.status_code == 200, viewer_list.text
    assert rel_html in {item["path"] for item in viewer_list.json()["files"]}

    anonymous_mount = client.post(
        "/api/v1/projects/preview-mount-file",
        json={"name": GENERAL_PROJECT, "path": rel_html},
    )
    assert anonymous_mount.status_code == 401

    mounted = client.post(
        "/api/v1/projects/preview-mount-file",
        headers={"Authorization": "Bearer viewer"},
        json={"name": GENERAL_PROJECT, "path": rel_html},
    )
    assert mounted.status_code == 200, mounted.text
    token = mounted.json()["token"]
    assert token != hashlib.sha256(str(folder.resolve()).encode("utf-8")).hexdigest()[:16]

    html = client.get(f"/api/v1{mounted.json()['relUrl']}")
    assert html.status_code == 200, html.text
    assert "<h1>Private</h1>" in html.text

    css = client.get(f"/api/v1/projects/preview-asset/{token}/style.css")
    assert css.status_code == 200, css.text
    assert "tomato" in css.text

    deterministic_token = hashlib.sha256(str(folder.resolve()).encode("utf-8")).hexdigest()[:16]
    deterministic = client.get(f"/api/v1/projects/preview-asset/{deterministic_token}/index.html")
    assert deterministic.status_code == 404

    unsigned = client.get(f"/api/v1/projects/{GENERAL_PROJECT}/files-raw/{rel_html}")
    assert unsigned.status_code == 401

    minted = client.post(
        f"/api/v1/projects/{GENERAL_PROJECT}/files-raw-token",
        headers={"Authorization": "Bearer viewer"},
        json={"path": rel_html},
    )
    assert minted.status_code == 200, minted.text
    signed_url = minted.json()["url"]
    assert "?token=" in signed_url

    signed = client.get(signed_url)
    assert signed.status_code == 200, signed.text
    assert "<h1>Private</h1>" in signed.text

    mismatched = client.get(signed_url.replace(rel_html, rel_css))
    assert mismatched.status_code == 401


def test_preview_proxy_does_not_forward_host_credentials():
    class Req:
        headers = {
            "Authorization": "Bearer secret",
            "Cookie": "sid=secret",
            "X-Api-Key": "secret",
            "Accept": "text/html",
        }

    headers = dict(_build_upstream_headers(Req(), 4321))

    assert headers["host"] == "127.0.0.1:4321"
    assert headers["Accept"] == "text/html"
    assert "Authorization" not in headers
    assert "Cookie" not in headers
    assert "X-Api-Key" not in headers
