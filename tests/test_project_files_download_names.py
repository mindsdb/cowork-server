"""`files-raw` serves any file name, not only names that fit in Latin-1."""
from __future__ import annotations

import asyncio
import os
import uuid
from urllib.parse import unquote

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session
from starlette.requests import ClientDisconnect

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

        client = TestClient(create_app())
        try:
            yield client, name, root
        finally:
            client.close()
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
    assert RFC5987_PREFIX in disposition, disposition
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


def _track_pinned_files(monkeypatch, project_files):
    """Record (fd, inode) for every descriptor `_pinned_stream` pins."""
    opened = []
    original = project_files._pinned_regular_file

    def tracking(*args, **kwargs):
        cm, fd, st = original(*args, **kwargs)
        opened.append((fd, st.st_ino))
        return cm, fd, st

    monkeypatch.setattr(project_files, "_pinned_regular_file", tracking)
    return opened


def _still_open(fd: int, inode: int) -> bool:
    # A closed number may be reused by another open, so compare the inode too.
    try:
        return os.fstat(fd).st_ino == inode
    except OSError:
        return False


def test_pinned_stream_closes_descriptor_when_response_cannot_be_built(tmp_path, monkeypatch):
    from cowork.api.v1.endpoints import project_files

    target = tmp_path / "report.txt"
    target.write_bytes(b"x")
    opened = _track_pinned_files(monkeypatch, project_files)

    class FailingResponse:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("response could not be built")

    monkeypatch.setattr(project_files, "PinnedStreamingResponse", FailingResponse)

    with pytest.raises(RuntimeError):
        project_files._pinned_stream(target, tmp_path, media_type="text/plain", headers={})

    assert len(opened) == 1
    assert not _still_open(*opened[0])


def test_pinned_stream_closes_descriptor_when_client_leaves_before_the_body(tmp_path, monkeypatch):
    from cowork.api.v1.endpoints import project_files

    target = tmp_path / "report.txt"
    target.write_bytes(b"x")
    opened = _track_pinned_files(monkeypatch, project_files)
    response = project_files._pinned_stream(
        target, tmp_path, media_type="text/plain", headers={}
    )

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        raise OSError("client went away")

    scope = {"type": "http", "asgi": {"spec_version": "2.4"}}
    with pytest.raises(ClientDisconnect):
        asyncio.run(response(scope, receive, send))

    # `response` still holds the unstarted body generator, so only the
    # response's own cleanup can have closed the descriptor here.
    assert len(opened) == 1
    assert not _still_open(*opened[0])
