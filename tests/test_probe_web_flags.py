"""The connector probe runs a real ChatSession, so it must honor the same
ANTON_WEB_SEARCH_ENABLED / ANTON_WEB_FETCH_ENABLED env flags as a user turn
instead of silently defaulting both to on.
"""
from __future__ import annotations

import pytest

from cowork.services.connectors import probe as probe_module
from cowork.services.connectors.probe import CredentialProbe


def _probe():
    return CredentialProbe(
        engine="postgres",
        credentials={},
        llm_client=None,
        workspace=None,
    )


async def _captured_config(monkeypatch):
    """Run the probe just far enough to capture the ChatSessionConfig it
    builds, before any real LLM call would happen."""
    captured = []

    def _capture(config):
        captured.append(config)
        raise RuntimeError("stop before a real turn")

    monkeypatch.setattr(probe_module, "build_chat_session", _capture)

    async for _kind, outcome in _probe().run():
        assert outcome.status == "failure"

    return captured[0]


@pytest.mark.asyncio
async def test_web_fetch_disabled_reaches_the_probe_session(monkeypatch):
    monkeypatch.setenv("ANTON_WEB_FETCH_ENABLED", "false")
    config = await _captured_config(monkeypatch)
    assert config.web_fetch_enabled is False
    assert config.web_search_enabled is True


@pytest.mark.asyncio
async def test_web_search_disabled_reaches_the_probe_session(monkeypatch):
    monkeypatch.setenv("ANTON_WEB_SEARCH_ENABLED", "false")
    config = await _captured_config(monkeypatch)
    assert config.web_search_enabled is False
    assert config.web_fetch_enabled is True


@pytest.mark.asyncio
async def test_defaults_stay_true(monkeypatch):
    monkeypatch.delenv("ANTON_WEB_FETCH_ENABLED", raising=False)
    monkeypatch.delenv("ANTON_WEB_SEARCH_ENABLED", raising=False)
    config = await _captured_config(monkeypatch)
    assert config.web_fetch_enabled is True
    assert config.web_search_enabled is True
