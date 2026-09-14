"""Chat actually works over Slack in org mode — proving the real user-facing
bug, not just its plumbing.

Every piece built earlier (webhook resolution, org-scoped dedupe, org-ready
gating, admin gating) correctly identified which org an inbound Slack message
belonged to — but `AntonChannelRuntime`, which actually runs the turn and
sends the reply, never received that org_id. It called
`scope_for_background_context()`, which fails closed unconditionally in org
mode, so every inbound message died silently after being ACKed. This file
drives a real signed Slack webhook through the real ASGI app and proves a
reply actually goes out, and that a second org gets nothing back.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time

import httpx
import pytest
from sqlmodel import select

import cowork.channels.runtime as runtime_mod
from cowork.channels.webhooks import drain_background_tasks
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.db.session import get_open_session
from cowork.models.channel import ChannelBinding, ChannelEvent, ChannelSession
from cowork.models.conversation import Conversation
from cowork.services.channels import ChannelConfigService

ORG_A = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
SIGNING_SECRET = "test-signing-secret-org-chat"


class FakeHarness:
    """One assistant delta, no LLM — records what it was asked to run."""

    def __init__(self):
        self.calls: list[dict] = []

    async def stream_response(self, *, conversation, input, channel_context=None, trace_metadata=None):
        self.calls.append({"conversation": conversation, "channel_context": channel_context})
        if False:
            yield

    async def formatter(self, stream, model, event_sink):
        async for _ in stream:
            pass
        event_sink("response.output_text.delta", {"delta": "pong from the org bot"})
        if False:
            yield


def _signed_slack_headers(body: bytes, signing_secret: str) -> dict:
    ts = str(int(time.time()))
    base = f"v0:{ts}:".encode() + body
    sig = "v0=" + hmac.new(signing_secret.encode(), base, hashlib.sha256).hexdigest()
    return {"x-slack-request-timestamp": ts, "x-slack-signature": sig}


def _slack_event_body(team_id: str, event_id: str, user: str = "U1") -> bytes:
    # A "D"-prefixed channel is a DM (SlackBridge.parse_inbound: is_group =
    # not channel.startswith("D")) — a fresh binding there defaults to
    # trigger_rule="always", so the turn runs without needing a mention.
    return json.dumps({
        "type": "event_callback", "team_id": team_id, "event_id": event_id,
        "event": {"type": "message", "channel": "D1", "user": user, "text": "hello bot",
                  "ts": "1700000000.000000"},
    }).encode()


def _delete_channel_events(*dedupe_keys: str) -> None:
    session = get_open_session()
    try:
        for row in session.exec(select(ChannelEvent).where(ChannelEvent.dedupe_key.in_(dedupe_keys))).all():
            session.delete(row)
        session.commit()
    finally:
        session.close()


def _delete_binding_and_its_rows(channel_type: str, external_group_id: str, org_id: str) -> None:
    # This suite shares the global test DB with test_channels_smoke.py, which
    # does an unfiltered select(ChannelBinding).one() — anything left behind
    # here breaks that test, not just this one. Goes through ConversationService
    # (not a raw delete) so message_events get dropped before their message —
    # the same order production deletion already uses.
    from cowork.services.conversations import ConversationService

    session = get_open_session()
    try:
        scoped = ScopedSession(session, TenantScope(org_mode=True, org_id=org_id))
        binding = session.exec(
            select(ChannelBinding).where(
                ChannelBinding.channel_type == channel_type,
                ChannelBinding.external_group_id == external_group_id,
            )
        ).first()
        if binding is None:
            return
        if binding.anton_conversation_id is not None:
            ConversationService(scoped).delete_conversation(binding.anton_conversation_id)
        for row in session.exec(select(ChannelSession).where(ChannelSession.binding_id == binding.id)).all():
            session.delete(row)
        session.delete(binding)
        session.commit()
    finally:
        session.close()


@pytest.fixture()
def org_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "enforce")
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


def test_slack_message_in_org_mode_gets_a_reply(monkeypatch, org_mode):
    fake_harness = FakeHarness()
    monkeypatch.setattr(runtime_mod, "get_harness", lambda _id: fake_harness)

    posted: list[tuple[str, dict | None]] = []
    real_post = httpx.AsyncClient.post

    async def fake_post(self, url, json=None, **kw):
        # Only Slack's own API is faked — the test's own client also POSTs
        # (over ASGITransport, to the webhook route) and must go through.
        if not str(url).startswith("https://slack.com/api"):
            return await real_post(self, url, json=json, **kw)
        method = str(url).rsplit("/", 1)[-1]
        posted.append((method, json))

        class R:
            def json(self):
                return {"ok": True, "ts": "1700000001.000000"}
        return R()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    from cowork.server import create_app
    app = create_app()
    adapters = app.state.channel_adapters

    session = get_open_session()
    try:
        svc = ChannelConfigService(ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A)))
        svc.set_config("slack", {"bot_token": "xoxb-org-a", "signing_secret": SIGNING_SECRET})
        svc.set_external_account_id("slack", "T-ORG-A")
    finally:
        session.close()

    try:
        async def flow():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                assert await adapters.get_or_refresh("slack", ORG_A) is not None
                body = _slack_event_body("T-ORG-A", "Ev-org-a-1")
                r = await client.post(
                    "/api/v1/channels/slack/events", content=body,
                    headers=_signed_slack_headers(body, SIGNING_SECRET),
                )
                assert r.status_code == 200
                await drain_background_tasks()

        asyncio.run(flow())

        # The turn actually ran — the old bug died before this, every time.
        assert len(fake_harness.calls) == 1, "the turn must run, not fail closed on scope"

        # The reply actually went out, to the right org's adapter.
        sends = [p for (m, p) in posted if m == "chat.postMessage"]
        assert len(sends) == 1
        assert "pong from the org bot" in sends[0]["text"]

        # The event log recorded success, not the old silent "failed".
        s = get_open_session()
        try:
            row = s.exec(
                select(ChannelEvent).where(ChannelEvent.dedupe_key == "slack:event:Ev-org-a-1")
            ).one()
            assert row.status == "routed", row.status

            org_a_scope = ScopedSession(s, TenantScope(org_mode=True, org_id=ORG_A))
            convs = org_a_scope.exec(org_a_scope.select(Conversation)).all()
            assert any(c.org_id == ORG_A for c in convs), "conversation must be stamped to the resolved org"
        finally:
            s.close()
    finally:
        session = get_open_session()
        try:
            ChannelConfigService(ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A))).delete_config("slack")
        finally:
            session.close()
        _delete_channel_events("slack:event:Ev-org-a-1")
        _delete_binding_and_its_rows("slack", "D1", ORG_A)


def test_org_mode_with_no_org_id_still_fails_closed(monkeypatch, org_mode):
    """Regression control: the fail-closed guard for a genuinely unresolved
    org in org mode must survive this fix, not just get bypassed."""
    from cowork.db.scoped import MissingTenantScopeError

    fake_harness = FakeHarness()
    monkeypatch.setattr(runtime_mod, "get_harness", lambda _id: fake_harness)
    runtime = runtime_mod.AntonChannelRuntime(runtime_mod.LiveAdapterRegistry())

    class _Addr:
        platform_id = "C1"
        thread_id = None

    class _Msg:
        content = "hi"
        is_mention = False
        is_group = False
        attachments = None
        sender_name = None

    class _Event:
        address = _Addr()
        message = _Msg()

    with pytest.raises(MissingTenantScopeError):
        asyncio.run(runtime.handle("slack", _Event(), org_id=None))

    assert fake_harness.calls == []
