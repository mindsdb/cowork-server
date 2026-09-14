"""Connection-test / credential-validator helper for channels.

`_is_configured` only checks that required fields have SOME stored value — a
garbage bot token reads as "configured" today, on desktop and in org mode
alike, since building a live adapter never rejects a bad token on its own.
This is the live check: does the platform actually accept it. A separate,
explicit action (not folded into `configured`/`status`), so an ordinary
status read never spends the org's credential against a live API call.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.channels.plugin import (
    ChannelCapabilities,
    ChannelPlugin,
    CredentialField,
    CredentialSchema,
    VerifyResult,
)
from cowork.channels.registry import PluginRegistry
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.services.channels import ChannelConfigService, UnknownChannelError


def _run(coro):
    return asyncio.run(coro)


# --- Slack: auth.test is the cheapest real proof a token works -------------

def test_verify_slack_credentials_ok(monkeypatch):
    from cowork.channels.plugins.slack import verify_slack_credentials

    async def fake_post(self, url, headers=None, **kw):
        assert url.endswith("/auth.test")
        assert headers["Authorization"] == "Bearer xoxb-real"

        class R:
            def json(self):
                return {"ok": True, "team": "Acme Corp"}
        return R()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    result = _run(verify_slack_credentials({"bot_token": "xoxb-real"}))
    assert result.ok is True
    assert "Acme Corp" in result.detail


def test_verify_slack_credentials_rejected(monkeypatch):
    from cowork.channels.plugins.slack import verify_slack_credentials

    async def fake_post(self, url, headers=None, **kw):
        class R:
            def json(self):
                return {"ok": False, "error": "invalid_auth"}
        return R()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    result = _run(verify_slack_credentials({"bot_token": "garbage"}))
    assert result.ok is False
    assert "invalid_auth" in result.detail


def test_verify_slack_credentials_missing_token():
    from cowork.channels.plugins.slack import verify_slack_credentials

    result = _run(verify_slack_credentials({}))
    assert result.ok is False


def test_verify_slack_credentials_network_error(monkeypatch):
    from cowork.channels.plugins.slack import verify_slack_credentials

    async def fake_post(self, url, headers=None, **kw):
        raise httpx.ConnectTimeout("boom")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    result = _run(verify_slack_credentials({"bot_token": "xoxb-real"}))
    assert result.ok is False


# --- Discord: GET /oauth2/applications/@me provides the application_id ------

def test_verify_discord_credentials_ok(monkeypatch):
    from cowork.channels.plugins.discord import verify_discord_credentials

    async def fake_get(self, url, headers=None, **kw):
        assert url.endswith("/oauth2/applications/@me")
        assert headers["Authorization"] == "Bot real-token"

        class R:
            status_code = 200

            def json(self):
                return {"id": "1234567890", "name": "cowork-bot"}
        return R()

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    result = _run(verify_discord_credentials({"bot_token": "real-token"}))
    assert result.ok is True
    assert "cowork-bot" in result.detail
    assert result.routing_key == "1234567890"


def test_verify_discord_credentials_rejected(monkeypatch):
    from cowork.channels.plugins.discord import verify_discord_credentials

    async def fake_get(self, url, headers=None, **kw):
        class R:
            status_code = 401

            def json(self):
                return {}
        return R()

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    result = _run(verify_discord_credentials({"bot_token": "garbage"}))
    assert result.ok is False
    assert "401" in result.detail


def test_verify_discord_credentials_missing_token():
    from cowork.channels.plugins.discord import verify_discord_credentials

    result = _run(verify_discord_credentials({}))
    assert result.ok is False


def test_verify_discord_credentials_network_error(monkeypatch):
    from cowork.channels.plugins.discord import verify_discord_credentials

    async def fake_get(self, url, headers=None, **kw):
        raise httpx.ConnectTimeout("boom")

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    result = _run(verify_discord_credentials({"bot_token": "real-token"}))
    assert result.ok is False


# --- Both plugins actually declare the hook ---------------------------------

def test_slack_plugin_declares_verify():
    from cowork.channels.plugins.slack import plugin, verify_slack_credentials

    assert plugin.verify is verify_slack_credentials
    assert plugin.capabilities.supports_verify is True


def test_discord_plugin_declares_verify():
    from cowork.channels.plugins.discord import plugin, verify_discord_credentials

    assert plugin.verify is verify_discord_credentials
    assert plugin.capabilities.supports_verify is True


# --- Service layer: dispatches to the plugin's hook against STORED creds ---

async def _stub_factory(creds):
    return None


def _plugin_with_verify(outcome: VerifyResult) -> ChannelPlugin:
    async def verify(creds):
        return outcome

    return ChannelPlugin(
        channel_type="slack",
        display_name="Slack",
        factory=_stub_factory,
        credentials=CredentialSchema(fields=(CredentialField(name="bot_token", label="Bot token"),)),
        capabilities=ChannelCapabilities(supports_verify=True),
        verify=verify,
    )


def _plugin_without_verify() -> ChannelPlugin:
    return ChannelPlugin(
        channel_type="telegram",
        display_name="Telegram",
        factory=_stub_factory,
        credentials=CredentialSchema(fields=(CredentialField(name="bot_token", label="Bot token"),)),
    )


@pytest.fixture()
def engine():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    return eng


def test_service_test_connection_calls_the_plugin_verify_hook_with_stored_creds(engine):
    registry = PluginRegistry()
    registry.register(_plugin_with_verify(VerifyResult(ok=True, detail="Connected to Acme")))
    scoped = ScopedSession(Session(engine), LOCAL_SCOPE)
    svc = ChannelConfigService(scoped, registry=registry)
    svc.set_config("slack", {"bot_token": "xoxb-real"})

    result = _run(svc.test_connection("slack"))

    assert result.ok is True
    assert result.detail == "Connected to Acme"


def test_service_test_connection_reports_unsupported_without_a_verify_hook(engine):
    registry = PluginRegistry()
    registry.register(_plugin_without_verify())
    scoped = ScopedSession(Session(engine), LOCAL_SCOPE)
    svc = ChannelConfigService(scoped, registry=registry)

    result = _run(svc.test_connection("telegram"))

    assert result.ok is False


def test_service_test_connection_unknown_channel_raises(engine):
    registry = PluginRegistry()
    scoped = ScopedSession(Session(engine), LOCAL_SCOPE)
    svc = ChannelConfigService(scoped, registry=registry)

    with pytest.raises(UnknownChannelError):
        _run(svc.test_connection("ghost"))
