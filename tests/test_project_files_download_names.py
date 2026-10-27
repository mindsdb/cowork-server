"""`files-raw` serves any file name, not only names that fit in Latin-1."""
from __future__ import annotations

import uuid
from urllib.parse import unquote

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.project import Project

pytestmark = pytest.mark.usefixtures("cleanup_tmp_projects")

RFC5987_PREFIX = "filename*=UTF-8''"


@pytest.fixture
def local_project(tmp_path, monkeypatch):
    """A desktop-style project (local tenancy, no organization) and a client."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    try:
        name = f"download-names-{uuid.uuid4().hex[:8]}"
        root = tmp_path / name
        root.mkdir()
        engine = get_engine(get_app_settings().database.uri)
        with Session(engine) as session:
            session.add(Project(id=uuid.uuid4(), name=name, path=str(root), org_id=None))
            session.commit()

        from cowork.server import create_app

        yield TestClient(create_app()), name, root
    finally:
        get_app_settings.cache_clear()


@pytest.mark.parametrize(
    "filename",
    ["report.pdf", "Отчёт.pdf", "報告.xlsx", "plan – v2.docx", 'q"uote.txt'],
)
def test_files_raw_serves_any_name(local_project, filename):
    client, project, root = local_project
    body = f"bytes of {filename}".encode()
    (root / filename).write_bytes(body)

    response = client.get(f"/api/v1/projects/{project}/files-raw/{filename}")

    assert response.status_code == 200, response.text
    assert response.content == body
    disposition = response.headers["content-disposition"]
    assert unquote(disposition.split(RFC5987_PREFIX, 1)[1]) == filename


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("report.pdf", "attachment; filename=\"report.pdf\"; filename*=UTF-8''report.pdf"),
        (
            "Отчёт.pdf",
            "attachment; filename=\"download.pdf\"; "
            "filename*=UTF-8''%D0%9E%D1%82%D1%87%D1%91%D1%82.pdf",
        ),
    ],
)
def test_files_raw_header_literal(local_project, filename, expected):
    client, project, root = local_project
    (root / filename).write_bytes(b"x")

    response = client.get(f"/api/v1/projects/{project}/files-raw/{filename}")

    assert response.status_code == 200, response.text
    assert response.headers["content-disposition"] == expected
