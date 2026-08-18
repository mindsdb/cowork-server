from pathlib import Path

import pytest

from cowork.services import artifacts


@pytest.fixture
def org_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    from cowork.common.settings.app_settings import get_app_settings
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.mark.asyncio
async def test_backend_launch_refused_in_org_mode(org_mode, tmp_path, monkeypatch):
    """No subprocess, no port probe, no anton import. Just a refusal."""
    def _boom(*_a, **_k):
        raise AssertionError("_probe_port must not run in org mode")

    monkeypatch.setattr(artifacts, "_probe_port", _boom)

    running, detail, port = await artifacts._ensure_backend_running(tmp_path / "slug", 8000)

    assert running is False
    assert port == 0
    assert "not available" in detail.lower()
    assert artifacts._LAUNCHED_BACKENDS == {}


@pytest.mark.asyncio
async def test_backend_launch_still_works_on_desktop(monkeypatch, tmp_path):
    """Desktop keeps the launcher. Guard against the kill switch being unconditional."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    from cowork.common.settings.app_settings import get_app_settings
    get_app_settings.cache_clear()
    monkeypatch.setattr(artifacts, "_probe_port", lambda _p: True)

    running, detail, port = await artifacts._ensure_backend_running(tmp_path / "slug", 8000)

    assert running is True
    assert detail == "already_running"
    get_app_settings.cache_clear()


def test_reveal_in_file_manager_refused_in_org_mode(org_mode, tmp_path):
    with pytest.raises(RuntimeError, match="not available"):
        artifacts.reveal_in_file_manager(tmp_path)


# ─── I1: open_artifact endpoint ────────────────────────────────────

@pytest.mark.asyncio
async def test_open_artifact_endpoint_refused_in_org_mode(org_mode):
    """Regression guard: this endpoint's guard had no test, so deleting it
    changed nothing in the full suite."""
    from fastapi import HTTPException

    from cowork.api.v1.endpoints.artifacts import _PathBody, open_artifact

    with pytest.raises(HTTPException) as exc:
        await open_artifact(_PathBody(path="/tmp/whatever"))
    assert exc.value.status_code == 403


# ─── I4: mount_preview's proxy branch must not register a token ───

@pytest.mark.asyncio
async def test_mount_preview_proxy_refused_in_org_mode(org_mode, tmp_path):
    """A fullstack artifact's `port` is read fresh from its own (agent-
    writable) metadata.json by the proxy route on every request. Registering
    the token before refusing would leave that SSRF pivot live even though
    `_ensure_backend_running` itself refuses to launch anything."""
    root = tmp_path / "artifact_root"
    root.mkdir()
    (root / "metadata.json").write_text('{"port": 54321}', encoding="utf-8")
    primary = root / "index.html"
    primary.write_text("<html></html>", encoding="utf-8")

    before = dict(artifacts._PREVIEW_MOUNTS)
    with pytest.raises(ValueError, match="not available"):
        await artifacts.mount_preview(primary)
    assert artifacts._PREVIEW_MOUNTS == before


@pytest.mark.asyncio
async def test_mount_preview_proxy_still_registers_on_desktop(monkeypatch, tmp_path):
    """Guard against the I4 fix being unconditional: desktop still needs the
    token registered so the proxy route (and a subsequent backend launch)
    can find the artifact root."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    from cowork.common.settings.app_settings import get_app_settings
    get_app_settings.cache_clear()

    root = tmp_path / "artifact_root"
    root.mkdir()
    (root / "metadata.json").write_text('{"port": 54321}', encoding="utf-8")
    primary = root / "index.html"
    primary.write_text("<html></html>", encoding="utf-8")

    payload = await artifacts.mount_preview(primary)

    assert payload["kind"] == "proxy"
    assert payload["token"] in artifacts._PREVIEW_MOUNTS
    get_app_settings.cache_clear()


# ─── I5: PDF/Word export can SSRF via the source HTML's own URIs ──

@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", ["pdf", "docx", "PDF", ".docx"])
async def test_export_pdf_docx_refused_in_org_mode(org_mode, fmt):
    from fastapi import HTTPException

    from cowork.api.v1.endpoints.artifacts import _ExportBody, export_artifact_endpoint

    with pytest.raises(HTTPException) as exc:
        await export_artifact_endpoint(_ExportBody(path="/tmp/whatever.md", format=fmt))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_export_html_not_refused_in_org_mode(org_mode, tmp_path):
    """Guard against the I5 fix being a blanket export ban: HTML export does
    no URI resolution (no fetch, no local read beyond the source itself), so
    it stays available. The 404 below (not 403) proves the request reached
    path resolution instead of being stopped by the export guard."""
    from fastapi import HTTPException

    from cowork.api.v1.endpoints.artifacts import _ExportBody, export_artifact_endpoint

    with pytest.raises(HTTPException) as exc:
        await export_artifact_endpoint(
            _ExportBody(path=str(tmp_path / "missing.md"), format="html")
        )
    assert exc.value.status_code == 404


# ─── C2: apply_env_to_process would pollute this whole process's env ──

def test_apply_workspace_env_refused_in_org_mode(org_mode):
    from unittest.mock import Mock

    from cowork.harnesses.anton_harness.harness import _apply_workspace_env_if_safe

    fake_workspace = Mock()
    applied = _apply_workspace_env_if_safe(fake_workspace)

    assert applied is False
    fake_workspace.apply_env_to_process.assert_not_called()


def test_apply_workspace_env_still_works_on_desktop(monkeypatch):
    """Desktop still needs its own .env loaded into the process (e.g. a
    locally-set API key). Guard against the kill switch being unconditional."""
    from unittest.mock import Mock

    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    from cowork.common.settings.app_settings import get_app_settings
    get_app_settings.cache_clear()

    from cowork.harnesses.anton_harness.harness import _apply_workspace_env_if_safe

    fake_workspace = Mock()
    applied = _apply_workspace_env_if_safe(fake_workspace)

    assert applied is True
    fake_workspace.apply_env_to_process.assert_called_once()
    get_app_settings.cache_clear()


# ─── C3: stream_response is the single choke point for in-process turns ──

@pytest.mark.asyncio
async def test_stream_response_refused_in_org_mode(org_mode):
    """AntonHarness.stream_response is reachable in org mode via three paths
    that all bypass the remote-worker routing gate: the legacy non-streaming
    branch in handlers/responses.py (ResponsesRequest.stream defaults to
    False), _produce/_run_turn's in-process fallback whenever
    COWORK_TURN_BACKEND isn't "remote", and the channel-ingress runtime.
    Refusing here, before `conversation` is ever touched, closes all three at
    once. `conversation=None` proves the refusal precedes any use of it."""
    from cowork.harnesses.anton_harness.harness import AntonHarness

    harness = AntonHarness()
    stream = harness.stream_response(conversation=None, input=[])
    with pytest.raises(RuntimeError, match="remote worker"):
        await stream.__anext__()
