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

import asyncio
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork.api.v1.router import api_router
from cowork.common.settings.app_settings import get_app_settings
from cowork.services.files import (
    FileValidationError,
    _humanize_bytes,
    reject_if_content_length_over_cap,
    stream_upload_to_path,
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


# ── Streaming ingest: cap is enforced mid-stream, never fully buffered ─────


class _FakeStreamUpload:
    """Stands in for an UploadFile backed by a stream far larger than the
    cap. Serves a fixed chunk per ``read(size)`` call up to ``total`` bytes
    and records how much was actually pulled — so a test can prove the cap
    check stops reading early instead of draining (and buffering) the lot."""

    def __init__(self, filename: str, content_type: str, total: int):
        self.filename = filename
        self.content_type = content_type
        self._remaining = total
        self.bytes_served = 0

    async def read(self, size: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        n = self._remaining if size is None or size < 0 else min(size, self._remaining)
        self._remaining -= n
        self.bytes_served += n
        return b"\x00" * n


def test_stream_upload_aborts_early_without_buffering(tmp_path, max_bytes):
    # A "10x the cap" stream: if the writer drained it all we'd both buffer
    # and write 10x the cap. It must stop just past the cap instead.
    total = max_bytes * 10
    fake = _FakeStreamUpload("huge.png", "image/png", total)
    dest = tmp_path / "huge.png"

    with pytest.raises(FileValidationError) as exc:
        asyncio.run(stream_upload_to_path(fake, dest))  # type: ignore[arg-type]

    assert "max is" in str(exc.value)
    # Pulled only ~one chunk past the cap — nowhere near the full payload.
    assert fake.bytes_served <= max_bytes + (1024 * 1024)
    assert fake.bytes_served < total
    # Partial file cleaned up — no truncated artifact left behind.
    assert not dest.exists()


def test_stream_upload_under_cap_writes_full_file(tmp_path):
    payload = b"hello streamed world"
    fake = _FakeStreamUpload("notes.txt", "text/plain", len(payload))

    dest = tmp_path / "notes.txt"
    written = asyncio.run(stream_upload_to_path(fake, dest))  # type: ignore[arg-type]

    assert written == len(payload)
    assert dest.exists()
    assert dest.stat().st_size == len(payload)


def test_attachment_upload_oversize_leaves_no_file_on_disk(client, max_bytes):
    # End-to-end through the renderer's path: an over-cap body is rejected and
    # nothing is committed to the files store (no orphaned partial upload).
    files_root = Path(get_app_settings().file.root_dir)
    before = set(files_root.glob("*")) if files_root.exists() else set()

    big = b"\x00" * (max_bytes + 1)
    resp = client.post(ATTACH_URL, files=[("files", ("big.png", big, "image/png"))])
    assert resp.status_code == 400, resp.text

    after = set(files_root.glob("*")) if files_root.exists() else set()
    assert after == before, "over-cap upload left a file behind in the store"


# ── Content-Length fast-reject ─────────────────────────────────────────────


def test_content_length_over_cap_raises(max_bytes):
    from cowork.services.files import UploadTooLarge

    with pytest.raises(UploadTooLarge):
        # Well past cap + the 1 MiB multipart allowance.
        reject_if_content_length_over_cap(max_bytes + (5 * 1024 * 1024))


def test_content_length_under_cap_passes(max_bytes):
    # Under cap — no raise. (None / garbage headers are also ignored.)
    reject_if_content_length_over_cap(max_bytes - 1)
    reject_if_content_length_over_cap(None)
    reject_if_content_length_over_cap("not-a-number")


def test_attachment_upload_huge_content_length_returns_413(client, max_bytes):
    # A declared body far over the cap is rejected up front as 413, before the
    # stream is consumed.
    big = b"\x00" * (max_bytes + (3 * 1024 * 1024))
    resp = client.post(ATTACH_URL, files=[("files", ("big.png", big, "image/png"))])
    assert resp.status_code == 413, resp.text
    assert "max is" in resp.json()["detail"]
