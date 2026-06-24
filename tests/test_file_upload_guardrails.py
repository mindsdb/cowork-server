"""Upload guardrails: a global size cap and a MIME/extension allow-list.

Both the OpenAI-style ``POST /api/v1/files/`` endpoint and the attachments
compat bridge (``POST /api/v1/attachments/{project}/{session}/upload`` — the
path the renderer actually hits) route through ``FileService.create_file``, so
the rules are enforced in one place and surfaced as a 400 with a human-readable
reason ("PNG is 41 MB — max is 25 MB").

Storage is pointed at a temp dir by conftest (COWORK_FILES_DIR); these tests
never touch the real ~/.cowork.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork.api.v1.router import api_router
from cowork.common.settings.app_settings import get_app_settings
from cowork.services.files import (
    FileValidationError,
    _humanize_bytes,
    validate_upload,
)


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(api_router)
    return TestClient(app)


@pytest.fixture()
def max_bytes() -> int:
    return get_app_settings().file.max_upload_bytes


# ── Unit: validate_upload ────────────────────────────────────────────────


def test_validate_upload_accepts_allowed_mime(max_bytes):
    # No extension, but an allowed MIME (e.g. a pasted screenshot) passes.
    validate_upload("blob", "image/png", 1024)


def test_validate_upload_accepts_allowed_extension_with_octet_stream(max_bytes):
    # Browsers often send a bare octet-stream for a .csv — the extension
    # rescues it.
    validate_upload("data.csv", "application/octet-stream", 1024)


def test_validate_upload_rejects_oversize_with_human_message(max_bytes):
    with pytest.raises(FileValidationError) as exc:
        validate_upload("huge.png", "image/png", max_bytes + 1)
    msg = str(exc.value)
    assert "huge.png" in msg
    # Human-readable size + cap, both rendered in MB for a 25 MiB default.
    assert "MB" in msg
    assert _humanize_bytes(max_bytes) in msg


def test_validate_upload_rejects_disallowed_type(max_bytes):
    with pytest.raises(FileValidationError) as exc:
        validate_upload("payload.exe", "application/x-msdownload", 16)
    msg = str(exc.value)
    assert "payload.exe" in msg
    assert "supported" in msg.lower()


def test_humanize_bytes_renders_mb():
    assert _humanize_bytes(41 * 1024 * 1024) == "41 MB"
    assert _humanize_bytes(25 * 1024 * 1024) == "25 MB"


# ── Endpoint: attachments compat bridge (the renderer's path) ─────────────

ATTACH_URL = "/api/v1/attachments/general/11111111-1111-1111-1111-111111111111/upload"


def test_attachment_upload_accepts_small_allowed_file(client):
    resp = client.post(
        ATTACH_URL,
        files=[("files", ("notes.txt", b"hello world", "text/plain"))],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body[0]["name"] == "notes.txt"


def test_attachment_upload_rejects_oversize_with_400(client, max_bytes):
    big = b"\x00" * (max_bytes + 1)
    resp = client.post(
        ATTACH_URL,
        files=[("files", ("big.png", big, "image/png"))],
    )
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert "big.png" in detail
    assert "MB" in detail  # "... is 25 MB — max is 25 MB."


def test_attachment_upload_rejects_disallowed_type_with_400(client):
    resp = client.post(
        ATTACH_URL,
        files=[("files", ("evil.exe", b"MZ\x90\x00", "application/x-msdownload"))],
    )
    assert resp.status_code == 400, resp.text
    assert "evil.exe" in resp.json()["detail"]


# ── Endpoint: OpenAI-style /files/ ────────────────────────────────────────


def test_files_endpoint_rejects_disallowed_type_with_400(client):
    resp = client.post(
        "/api/v1/files/",
        data={"purpose": "assistants"},
        files={"file": ("archive.zip", b"PK\x03\x04", "application/zip")},
    )
    assert resp.status_code == 400, resp.text
    assert "archive.zip" in resp.json()["detail"]


def test_files_endpoint_accepts_allowed_file(client):
    resp = client.post(
        "/api/v1/files/",
        data={"purpose": "assistants"},
        files={"file": ("report.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["filename"] == "report.pdf"
