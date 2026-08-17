"""Bridge resolution for inbound webhooks: falls back to the plain
channel_type-only resolver whenever a plugin doesn't declare a routing-key
extractor, the extractor finds nothing, or the key doesn't resolve to a known
installation — so plugins that don't participate (today: everyone but Slack)
are completely unaffected."""
from __future__ import annotations

import hashlib
import hmac
import json
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import select

from cowork.channels.plugin import ChannelPlugin, CredentialSchema
from cowork.channels.registry import PluginRegistry
from cowork.channels.runtime import LiveAdapterRegistry
from cowork.channels.webhooks import build_channel_webhook_router, intake_events, resolve_bridge
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.db.session import get_open_session
from cowork.services.channel_events import ChannelEventService
from cowork.services.channels import ChannelConfigService

ORG_A = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ORG_B = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"


class _FakeBridge:
    def dedupe_key(self, event):
        return event["key"]


async def _fake_factory(creds):
    return None


def _plugin(*, extract_routing_key=None) -> ChannelPlugin:
    return ChannelPlugin(
        channel_type="slack",
        display_name="Slack",
        factory=_fake_factory,
        credentials=CredentialSchema(),
        extract_routing_key=extract_routing_key,
    )


async def test_falls_back_when_plugin_has_no_extractor():
    plugin = _plugin()
    bridge, org_id = await resolve_bridge(
        channel_type="slack", plugin=plugin, body=b"{}", headers={},
        resolver=lambda ct: "local-bridge", org_resolver=None,
    )
    assert bridge == "local-bridge"
    assert org_id is None


async def test_falls_back_when_extractor_finds_no_routing_key():
    plugin = _plugin(extract_routing_key=lambda body, headers: None)
    calls = []

    async def org_resolver(ct, key):
        calls.append((ct, key))
        return ("org-bridge", "org-x")

    bridge, org_id = await resolve_bridge(
        channel_type="slack", plugin=plugin, body=b"{}", headers={},
        resolver=lambda ct: "local-bridge", org_resolver=org_resolver,
    )
    assert bridge == "local-bridge"
    assert org_id is None
    assert calls == []  # never asked — nothing to resolve


async def test_resolves_org_bridge_when_routing_key_found():
    plugin = _plugin(extract_routing_key=lambda body, headers: "T-A")

    async def org_resolver(ct, key):
        assert (ct, key) == ("slack", "T-A")
        return ("org-a-bridge", "org-a")

    bridge, org_id = await resolve_bridge(
        channel_type="slack", plugin=plugin, body=b"{}", headers={},
        resolver=lambda ct: "local-bridge", org_resolver=org_resolver,
    )
    assert bridge == "org-a-bridge"
    assert org_id == "org-a"


async def test_falls_back_when_routing_key_does_not_resolve():
    plugin = _plugin(extract_routing_key=lambda body, headers: "T-unknown")

    async def org_resolver(ct, key):
        return None  # nobody claims this account

    bridge, org_id = await resolve_bridge(
        channel_type="slack", plugin=plugin, body=b"{}", headers={},
        resolver=lambda ct: "local-bridge", org_resolver=org_resolver,
    )
    assert bridge == "local-bridge"
    assert org_id is None


async def test_extractor_receives_the_actual_body_and_headers():
    seen = {}

    def extract(body, headers):
        seen["body"] = body
        seen["headers"] = headers
        return None

    plugin = _plugin(extract_routing_key=extract)
    await resolve_bridge(
        channel_type="slack", plugin=plugin, body=b'{"team_id":"T-A"}',
        headers={"x-slack-signature": "v0=abc"},
        resolver=lambda ct: "local-bridge", org_resolver=lambda ct, key: None,
    )
    assert seen["body"] == b'{"team_id":"T-A"}'
    assert seen["headers"] == {"x-slack-signature": "v0=abc"}


# --- intake_events: dedupe/event-log land in the resolved org's scope ---

def _capturing_scheduler(store):
    def sched(coro):
        store.append(coro)
    return sched


def _delete_channel_events(*dedupe_keys: str) -> None:
    # These tests hit the shared session-scoped test DB (get_open_session),
    # not an isolated engine — clean up so other tests' own counts stay valid.
    from cowork.models.channel import ChannelEvent

    session = get_open_session()
    try:
        for row in session.exec(
            select(ChannelEvent).where(ChannelEvent.dedupe_key.in_(dedupe_keys))
        ).all():
            session.delete(row)
        session.commit()
    finally:
        session.close()


async def test_intake_events_stamps_the_resolved_org(monkeypatch):
    captured = []
    try:
        intake_events(
            "slack", _FakeBridge(), [{"key": "evt-a"}],
            sink=lambda ct, evt: None, scheduler=_capturing_scheduler(captured), org_id=ORG_A,
        )
        for coro in captured:
            await coro

        session = get_open_session()
        try:
            svc = ChannelEventService(ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A)))
            assert svc.is_duplicate_inbound("slack", "evt-a") is True
        finally:
            session.close()
    finally:
        _delete_channel_events("evt-a")


async def test_intake_events_isolates_dedupe_across_orgs():
    # The real signal isn't "is_duplicate_inbound returns True" — org B seeing
    # its OWN row and org B incorrectly seeing org A's leaked row both look
    # like that. The real proof is that org B's intake actually scheduled
    # processing instead of silently treating a same-key event as a dup.
    captured_a: list = []
    captured_b: list = []
    try:
        intake_events(
            "slack", _FakeBridge(), [{"key": "evt-shared"}],
            sink=lambda ct, evt: None, scheduler=_capturing_scheduler(captured_a), org_id=ORG_A,
        )
        for coro in captured_a:
            await coro

        intake_events(
            "slack", _FakeBridge(), [{"key": "evt-shared"}],
            sink=lambda ct, evt: None, scheduler=_capturing_scheduler(captured_b), org_id=ORG_B,
        )

        assert len(captured_a) == 1
        assert len(captured_b) == 1  # would be 0 if org B's write got deduped against org A's
        for coro in captured_b:
            await coro
    finally:
        _delete_channel_events("evt-shared")


# --- Real plugins' extractors: pulled unverified, used only as a lookup key ---

def test_slack_extract_team_id():
    from cowork.channels.plugins.slack import extract_team_id

    body = b'{"type":"event_callback","team_id":"T0001","event":{}}'
    assert extract_team_id(body, {}) == "T0001"


def test_slack_extract_team_id_missing():
    from cowork.channels.plugins.slack import extract_team_id

    assert extract_team_id(b'{"type":"event_callback","event":{}}', {}) is None


def test_slack_extract_team_id_malformed_body():
    from cowork.channels.plugins.slack import extract_team_id

    assert extract_team_id(b"not json", {}) is None


def test_slack_plugin_declares_the_extractor():
    from cowork.channels.plugins.slack import extract_team_id, plugin

    assert plugin.extract_routing_key is extract_team_id


def test_discord_extract_application_id():
    from cowork.channels.plugins.discord import extract_application_id

    body = b'{"type":2,"application_id":"A0001","guild_id":"G0001"}'
    assert extract_application_id(body, {}) == "A0001"


def test_discord_extract_application_id_present_without_guild():
    # DM-context interactions have no guild_id — application_id is what's
    # actually reliable, which is exactly why it's the routing key, not guild_id.
    from cowork.channels.plugins.discord import extract_application_id

    body = b'{"type":2,"application_id":"A0001"}'
    assert extract_application_id(body, {}) == "A0001"


def test_discord_extract_application_id_missing():
    from cowork.channels.plugins.discord import extract_application_id

    assert extract_application_id(b'{"type":2}', {}) is None


def test_discord_plugin_declares_the_extractor():
    from cowork.channels.plugins.discord import extract_application_id, plugin

    assert plugin.extract_routing_key is extract_application_id


# --- End-to-end: local mode's real wiring, exactly as _install_channels builds it ---

def _signed_slack_headers(body: bytes, signing_secret: str) -> dict:
    ts = str(int(time.time()))
    base = f"v0:{ts}:".encode("utf-8") + body
    sig = "v0=" + hmac.new(signing_secret.encode("utf-8"), base, hashlib.sha256).hexdigest()
    return {"x-slack-request-timestamp": ts, "x-slack-signature": sig}


async def test_local_mode_webhook_falls_through_to_local_bridge_end_to_end():
    """The claim from the design conversation, proven rather than reasoned
    through: in local mode, a real inbound webhook whose team_id matches
    nothing (true today — nothing sets external_account_id on a local
    install) still resolves and verifies correctly, via the same plain-
    resolver path as before this feature existed. Uses the real Slack
    plugin, the real registry, and the real resolve_org_bridge — the actual
    production wiring _install_channels builds, not a stand-in for it."""
    from cowork.channels.plugins.slack import plugin as slack_plugin

    registry = PluginRegistry()
    registry.register(slack_plugin)

    signing_secret = "test-signing-secret"
    session = get_open_session()
    try:
        ChannelConfigService(ScopedSession(session, LOCAL_SCOPE), registry=registry).set_config(
            "slack", {"bot_token": "xoxb-test", "signing_secret": signing_secret},
        )
    finally:
        session.close()

    try:
        adapters = LiveAdapterRegistry(registry)
        assert await adapters.refresh_all() == ["slack"]  # mirrors the real lifespan bootstrap

        delivered = []

        async def sink(channel_type, event):
            delivered.append((channel_type, event))

        app = FastAPI()
        app.include_router(
            build_channel_webhook_router(
                slack_plugin, resolver=adapters.get, sink=sink,
                org_resolver=adapters.resolve_org_bridge,
            )
        )
        client = TestClient(app)

        body = json.dumps({
            "type": "event_callback",
            "team_id": "T-UNMATCHED",  # nothing sets external_account_id on a local install
            "event_id": "Ev0001",
            "event": {
                "type": "message", "channel": "C1", "user": "U1", "text": "hello",
                "ts": "1700000000.000000",
            },
        }).encode()

        response = client.post("/slack/events", content=body, headers=_signed_slack_headers(body, signing_secret))

        assert response.status_code == 200
        assert len(delivered) == 1
        assert delivered[0][0] == "slack"
    finally:
        session = get_open_session()
        try:
            ChannelConfigService(ScopedSession(session, LOCAL_SCOPE), registry=registry).delete_config("slack")
        finally:
            session.close()
        _delete_channel_events("slack:event:Ev0001")
