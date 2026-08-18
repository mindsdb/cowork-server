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
