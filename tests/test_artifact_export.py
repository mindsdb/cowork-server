"""Tests for the artifact export pipeline (markdown/HTML → PDF/Word/HTML).

Covers the pure-Python converter (cowork.services.artifact_export) for each
target format, its rejection of unsupported sources, and the `/artifacts/export`
endpoint contract — in particular that it returns a signed origin-relative
`serveUrl` so the web client can download the result (the desktop client opens
it by path).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork.api.v1.router import api_router
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.project import Project
from cowork.services.artifact_export import ExportError, export_artifact
from cowork.services.projects import GENERAL_PROJECT_ID
from sqlmodel import Session


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(api_router)
    return TestClient(app)


def _general_project_path() -> Path:
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        project = session.get(Project, GENERAL_PROJECT_ID)
        assert project is not None
        return Path(project.path)


@pytest.fixture(autouse=True)
def clean_export_test_artifacts():
    root = _general_project_path() / ".anton" / "artifacts"
    pattern = "export-test-*"
    if root.is_dir():
        for folder in root.glob(pattern):
            shutil.rmtree(folder, ignore_errors=True)
    yield
    if root.is_dir():
        for folder in root.glob(pattern):
            shutil.rmtree(folder, ignore_errors=True)


def _make_artifact(*, files: dict[str, str], artifact_type: str = "document") -> tuple[str, Path]:
    """Write an artifact folder under the GENERAL project's artifacts tree."""
    project = _general_project_path()
    slug = f"export-test-{uuid4().hex}"
    artifact_id = f"artifact-{uuid4().hex}"
    folder = project / ".anton" / "artifacts" / slug
    folder.mkdir(parents=True)
    primary = next(iter(files), "")
    (folder / "metadata.json").write_text(
        json.dumps(
            {
                "id": artifact_id,
                "slug": slug,
                "name": "Export Sample",
                "type": artifact_type,
                "primary": primary,
            }
        ),
        encoding="utf-8",
    )
    for rel_path, content in files.items():
        (folder / rel_path).write_text(content, encoding="utf-8")
    return artifact_id, folder


_MARKDOWN = "# Title\n\nA paragraph with **bold** text.\n\n- one\n- two\n"


@pytest.mark.parametrize(
    ("fmt", "magic"),
    [
        ("pdf", b"%PDF"),
        ("docx", b"PK"),  # docx is a zip container
    ],
)
def test_export_markdown_to_binary_formats(fmt: str, magic: bytes):
    _, folder = _make_artifact(files={"report.md": _MARKDOWN})
    source = folder / "report.md"

    out = export_artifact(source, fmt)

    assert out == source.with_suffix(f".{fmt}")
    assert out.is_file()
    assert out.read_bytes()[: len(magic)] == magic


def test_export_markdown_to_html_wraps_document():
    _, folder = _make_artifact(files={"report.md": _MARKDOWN})
    source = folder / "report.md"

    out = export_artifact(source, "html")

    assert out == source.with_suffix(".html")
    text = out.read_text(encoding="utf-8")
    assert text.lstrip().startswith("<!DOCTYPE html>")
    # Markdown was rendered, not embedded verbatim.
    assert "<strong>bold</strong>" in text
    assert "<title>report</title>" in text


def test_export_html_source_to_pdf():
    _, folder = _make_artifact(
        files={"page.html": "<h1>Hello</h1><p>World</p>"},
        artifact_type="html-app",
    )
    out = export_artifact(folder / "page.html", "pdf")
    assert out.is_file()
    assert out.read_bytes()[:4] == b"%PDF"


def test_export_rejects_unsupported_format():
    _, folder = _make_artifact(files={"report.md": _MARKDOWN})
    with pytest.raises(ExportError):
        export_artifact(folder / "report.md", "rtf")


def test_export_rejects_unsupported_source():
    _, folder = _make_artifact(files={"data.bin": "not a document"})
    with pytest.raises(ExportError):
        export_artifact(folder / "data.bin", "pdf")


def test_export_endpoint_returns_serve_url(client: TestClient):
    _, folder = _make_artifact(files={"report.md": _MARKDOWN})
    source = folder / "report.md"

    resp = client.post(
        "/api/v1/artifacts/export",
        json={"path": str(source), "format": "html"},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["filename"] == "report.html"
    assert Path(body["path"]).is_file()
    # The signed, origin-relative URL the web client uses to download the file.
    assert body["serveUrl"].startswith("/api/v1/artifacts/serve/")
    assert "token=" in body["serveUrl"]


def test_export_endpoint_rejects_unsupported_format(client: TestClient):
    _, folder = _make_artifact(files={"report.md": _MARKDOWN})
    resp = client.post(
        "/api/v1/artifacts/export",
        json={"path": str(folder / "report.md"), "format": "rtf"},
    )
    assert resp.status_code == 400, resp.text
