"""Channels actually dispatching in org (SaaS) mode.

Everything here was already built org-aware (installations, credentials,
webhook resolution, per-org adapters) but unreachable: `_install_channels`
loaded no plugins and mounted no routes outside local mode, and
`_require_local_channels()` 501-gated the config/lifecycle endpoints. This
file proves the activation itself, not the underlying mechanisms (see
test_channels_tenancy.py / test_channel_webhooks.py for those).

Streaming/polling ingress (Discord Gateway, Slack Socket Mode) deliberately
stays inert in org mode even after this — see the ingress test at the bottom.
"""
from __future__ import annotations

import os

import pytest

from cowork.common.settings.app_settings import get_app_settings

ORG_A = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ORG_B = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"
USER_A = "11111111-1111-4111-8111-111111111111"
USER_B = "22222222-2222-4222-8222-222222222222"

A = {"X-User-Id": USER_A, "X-Organization-Id": ORG_A}
B = {"X-User-Id": USER_B, "X-Organization-Id": ORG_B}


@pytest.fixture(scope="module")
def client():
    saved = {k: os.environ.get(k) for k in ("COWORK_TENANCY_MODE", "COWORK_IDENTITY_ENFORCE")}
    os.environ["COWORK_TENANCY_MODE"] = "org"
    os.environ["COWORK_IDENTITY_ENFORCE"] = "enforce"
    get_app_settings.cache_clear()
    try:
        from fastapi.testclient import TestClient
        from cowork.server import create_app

        # No `with`: skipping the lifespan skips boot migrations (they would
        # collide with the schema conftest already created) and never starts
        # background workers in this suite. Mirrors test_org_isolation_e2e.py.
        test_client = TestClient(create_app())
        try:
            yield test_client
        finally:
            test_client.close()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        get_app_settings.cache_clear()


# --- activation: plugins load and webhook routes mount in org mode too -----

def test_plugins_endpoint_lists_first_party_plugins_in_org_mode(client):
    res = client.get("/api/v1/channels/plugins", headers=A)
    assert res.status_code == 200
    types = {p["channel_type"] for p in res.json()}
    assert {"slack", "discord"} <= types


def test_slack_webhook_route_is_mounted_in_org_mode(client):
    # Nothing claims this team_id, so the route exists but safely ACK-ignores.
    # A 404 here would mean the route never got mounted at all.
    res = client.post(
        "/api/v1/channels/slack/events",
        json={"type": "event_callback", "team_id": "T-UNMATCHED", "event": {}},
    )
    assert res.status_code == 204


def test_webhook_paths_are_exempt_from_identity_enforcement(client):
    # No identity headers — a normal API path would 401 here (see
    # test_no_identity_is_401 in test_org_isolation_e2e.py); the webhook path
    # must not, since external platforms authenticate with their own signature.
    res = client.post("/api/v1/channels/discord/interactions", json={"type": 1})
    assert res.status_code != 401


# --- config/status/lifecycle: no longer gated by tenancy mode alone --------

def test_channel_status_available_in_org_mode(client):
    res = client.get("/api/v1/channels/status", headers=A)
    assert res.status_code == 200


def test_channel_config_get_available_in_org_mode(client):
    res = client.get("/api/v1/channels/slack/config", headers=A)
    assert res.status_code == 200
    assert res.json()["configured"] is False


def test_channel_config_put_and_get_are_org_scoped(client):
    put = client.put(
        "/api/v1/channels/slack/config", headers=A,
        json={"values": {"bot_token": "xoxb-a", "signing_secret": "secret-a"}},
    )
    assert put.status_code == 200, put.text
    try:
        # Org B never configured slack — its own state, not org A's leaking in.
        other = client.get("/api/v1/channels/slack/config", headers=B)
        assert other.json()["configured"] is False

        mine = client.get("/api/v1/channels/slack/config", headers=A)
        assert mine.json()["configured"] is True
    finally:
        client.delete("/api/v1/channels/slack/config", headers=A)


def test_channel_setup_available_in_org_mode(client):
    # Slack has no setup/teardown lifecycle at all, so this still 501s — but
    # for THAT reason, not the tenancy gate ("not available in org deployments").
    res = client.post("/api/v1/channels/slack/setup", headers=A)
    assert res.status_code == 501
    assert "not implemented" in res.json()["detail"]


# --- the channel-agent (harness) setting is per-org, not one shared row ----

def test_channel_agent_endpoints_available_in_org_mode(client):
    res = client.get("/api/v1/channels/agent", headers=A)
    assert res.status_code == 200

    put = client.put("/api/v1/channels/agent", headers=A, json={"harness": "anton"})
    assert put.status_code == 200, put.text


def test_set_channel_agent_write_does_not_leak_to_other_orgs(monkeypatch):
    """The precise thing this change adds: set_channel_agent must write the
    caller's org row, not the one global row every org would otherwise share."""
    import cowork.api.v1.endpoints.channels as channels_ep
    from cowork.common.settings.user_settings import get_user_settings
    from cowork.db.scoped import ScopedSession, TenantScope
    from cowork.db.session import get_open_session
    from cowork.schemas.channels import ChannelAgentUpdateRequest
    from cowork.services.settings import SettingService

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    monkeypatch.setattr(channels_ep, "available_harness_ids", lambda: ["anton", "hermes"])

    session = get_open_session()
    scope_a = TenantScope(org_mode=True, org_id=ORG_A)
    scoped_a = ScopedSession(session, scope_a)
    try:
        result = channels_ep.set_channel_agent(
            ChannelAgentUpdateRequest(harness="hermes"), session, scoped_a
        )
        assert result.harness == "hermes"
        assert get_user_settings(TenantScope(org_mode=True, org_id=ORG_A)).channels_harness == "hermes"
        assert get_user_settings(TenantScope(org_mode=True, org_id=ORG_B)).channels_harness != "hermes"
    finally:
        SettingService(session, scope_a).delete_setting("channels_harness")
        session.close()
        get_app_settings.cache_clear()


# --- streaming/polling ingress stays inert in org mode, like Telegram/
# WhatsApp already are — reconciling it per-request would need a per-org
# IngressManager, which is its own deferred design, not part of this change.

def test_channel_config_put_does_not_start_ingress_in_org_mode(client, monkeypatch):
    import cowork.api.v1.endpoints.channels as channels_ep

    called = []

    async def spy(*args, **kwargs):
        called.append(True)

    monkeypatch.setattr(channels_ep, "sync_channel_ingress", spy)
    try:
        res = client.put(
            "/api/v1/channels/discord/config", headers=A,
            json={"values": {"bot_token": "discord-bot-token"}},
        )
        assert res.status_code == 200, res.text
        assert called == []
    finally:
        client.delete("/api/v1/channels/discord/config", headers=A)
