"""Downloading an HTML artifact must return the file, byte for byte.

Regression coverage for the `/serve` and `/preview-asset` injection gate: both
routes now honour `?download=1` the same way the drafts route already did
(`serve_private_draft` in artifact_workspace.py), so "Download" in the
renderer never saves a file with the preview shim spliced into it.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cowork.server import create_app
from cowork.services.artifacts import _PREVIEW_MOUNTS


@pytest.fixture()
def projects_root(tmp_path, monkeypatch):
    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(root))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path))
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    yield root
    get_app_settings.cache_clear()


def _client():
    return TestClient(
        create_app(), base_url="http://127.0.0.1:26868", client=("127.0.0.1", 54323)
    )


# Non-ASCII on purpose: it is also the payload BLOCKING 1 names for the
# mojibake harm (a <meta charset> pushed past the 1024-byte prescan).
_HTML = (
    "<html><head><meta charset=\"utf-8\"></head><body>café — naïve</body></html>"
).encode("utf-8")


def test_serve_download_is_byte_identical_to_disk(projects_root):
    project = projects_root / "proj"
    artifact = project / ".anton" / "artifacts" / "dash"
    artifact.mkdir(parents=True)
    on_disk = artifact / "index.html"
    on_disk.write_bytes(_HTML)

    client = _client()
    plain = client.get("/api/v1/artifacts/serve/proj/dash/index.html")
    assert plain.status_code == 200
    # Sanity: without the flag, the shim is still injected as before.
    assert b"anton-preview" in plain.content

    downloaded = client.get("/api/v1/artifacts/serve/proj/dash/index.html?download=1")
    assert downloaded.status_code == 200
    assert downloaded.content == on_disk.read_bytes() == _HTML
    assert b"anton-preview" not in downloaded.content


def test_preview_asset_download_is_byte_identical_to_disk(projects_root):
    artifact = projects_root / "mount-source"
    artifact.mkdir()
    on_disk = artifact / "index.html"
    on_disk.write_bytes(_HTML)

    token = "download-gate-test-token"
    _PREVIEW_MOUNTS[token] = artifact
    try:
        client = _client()
        plain = client.get(f"/api/v1/artifacts/preview-asset/{token}/index.html")
        assert plain.status_code == 200
        assert b"anton-preview" in plain.content

        downloaded = client.get(
            f"/api/v1/artifacts/preview-asset/{token}/index.html?download=1"
        )
        assert downloaded.status_code == 200
        assert downloaded.content == on_disk.read_bytes() == _HTML
        assert b"anton-preview" not in downloaded.content
    finally:
        _PREVIEW_MOUNTS.pop(token, None)
