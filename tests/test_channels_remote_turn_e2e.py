"""A real Slack message, in org mode, with COWORK_TURN_BACKEND=remote — the
exact combination staging runs — gets a reply, proving Tasks 1-3 compose.
Stubs only stream_remote_replies (the actual remote-worker call); every
other piece (webhook, dedupe, binding, AntonChannelRuntime, _turn_stream,
remote_turn_events, ChannelConfigService) is the real production code."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time

import httpx
import pytest
from sqlmodel import select

import cowork.turnqueue.remote_turn as remote_turn_mod
from cowork.channels.webhooks import drain_background_tasks
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.db.session import get_open_session
from cowork.models.channel import ChannelBinding, ChannelEvent, ChannelSession
from cowork.services.channels import ChannelConfigService

ORG_A = "3fa85f64-5717-4562-b3fc-2c963f66afa6"
SIGNING_SECRET = "test-signing-secret-remote-turn"


def _signed_slack_headers(body: bytes, signing_secret: str) -> dict:
    ts = str(int(time.time()))
    base = f"v0:{ts}:".encode() + body
    sig = "v0=" + hmac.new(signing_secret.encode(), base, hashlib.sha256).hexdigest()
    return {"x-slack-request-timestamp": ts, "x-slack-signature": sig}


def _slack_event_body(team_id: str, event_id: str) -> bytes:
    # "D"-prefixed channel = DM, so a fresh binding defaults to
    # trigger_rule="always" — the turn runs without needing a mention.
    return json.dumps({
        "type": "event_callback", "team_id": team_id, "event_id": event_id,
        "event": {"type": "message", "channel": "D1", "user": "U1", "text": "hello bot",
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
def org_mode_remote_backend(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "enforce")
    monkeypatch.setenv("COWORK_TURN_BACKEND", "remote")
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


def test_slack_message_gets_a_reply_via_the_remote_worker(monkeypatch, org_mode_remote_backend):
    # Sync, not async: asyncio.to_thread expects a plain callable — an async
    # def here would hand back an unawaited coroutine that silently no-ops.
    def fake_stage(session, conv_id):
        pass

    monkeypatch.setattr(
        "cowork.handlers.responses.ResponsesHandler._stage_remote_workspace_files",
        staticmethod(fake_stage),
    )
    monkeypatch.setattr(
        "cowork.handlers.responses.ResponsesHandler._remote_artifacts_context",
        staticmethod(lambda session, conv_id: None),
    )
    monkeypatch.setattr(
        "cowork.handlers.responses.ResponsesHandler._remote_history",
        staticmethod(lambda session, conv_id: []),
    )
    monkeypatch.setattr(
        "cowork.handlers.responses.ResponsesHandler._remote_workspace",
        staticmethod(lambda session, conv_id: {}),
    )
    monkeypatch.setattr(
        "cowork.handlers.responses.ResponsesHandler._persist_turn_memory",
        staticmethod(lambda session, conv_id, entries: None),
    )

    async def fake_replies(**kwargs):
        yield "turn_delta", {"text": "pong from the remote worker"}
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    posted: list[dict] = []
    real_post = httpx.AsyncClient.post

    async def fake_post(self, url, json=None, **kw):
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
        svc.set_config("slack", {"bot_token": "xoxb-org-a-remote", "signing_secret": SIGNING_SECRET})
        svc.set_external_account_id("slack", "T-ORG-A-REMOTE")
    finally:
        session.close()

    try:
        async def flow():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                assert await adapters.get_or_refresh("slack", ORG_A) is not None
                body = _slack_event_body("T-ORG-A-REMOTE", "Ev-remote-1")
                r = await client.post(
                    "/api/v1/channels/slack/events", content=body,
                    headers=_signed_slack_headers(body, SIGNING_SECRET),
                )
                assert r.status_code == 200
                await drain_background_tasks()

        asyncio.run(flow())

        sends = [p for (m, p) in posted if m == "chat.postMessage"]
        assert len(sends) == 1
        assert "pong from the remote worker" in sends[0]["text"]

        s = get_open_session()
        try:
            row = s.exec(
                select(ChannelEvent).where(ChannelEvent.dedupe_key == "slack:event:Ev-remote-1")
            ).one()
            assert row.status == "routed", row.status
        finally:
            s.close()
    finally:
        session = get_open_session()
        try:
            ChannelConfigService(ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A))).delete_config("slack")
        finally:
            session.close()
        _delete_channel_events("slack:event:Ev-remote-1")
        _delete_binding_and_its_rows("slack", "D1", ORG_A)


def test_a_remote_turn_failure_delivers_an_error_message_not_silence(monkeypatch, org_mode_remote_backend):
    monkeypatch.setattr(
        "cowork.handlers.responses.ResponsesHandler._stage_remote_workspace_files",
        staticmethod(lambda session, conv_id: None),
    )
    monkeypatch.setattr(
        "cowork.handlers.responses.ResponsesHandler._remote_artifacts_context",
        staticmethod(lambda session, conv_id: None),
    )
    monkeypatch.setattr(
        "cowork.handlers.responses.ResponsesHandler._remote_history",
        staticmethod(lambda session, conv_id: []),
    )
    monkeypatch.setattr(
        "cowork.handlers.responses.ResponsesHandler._remote_workspace",
        staticmethod(lambda session, conv_id: {}),
    )
    monkeypatch.setattr(
        "cowork.handlers.responses.ResponsesHandler._persist_turn_memory",
        staticmethod(lambda session, conv_id, entries: None),
    )

    async def fake_replies(**kwargs):
        yield "turn_failed", {"code": "anton_error", "message": "the worker is down"}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    posted: list[dict] = []
    real_post = httpx.AsyncClient.post

    async def fake_post(self, url, json=None, **kw):
        if not str(url).startswith("https://slack.com/api"):
            return await real_post(self, url, json=json, **kw)
        method = str(url).rsplit("/", 1)[-1]
        posted.append((method, json))

        class R:
            def json(self):
                return {"ok": True, "ts": "1700000002.000000"}
        return R()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    from cowork.server import create_app
    app = create_app()
    adapters = app.state.channel_adapters

    session = get_open_session()
    try:
        svc = ChannelConfigService(ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A)))
        svc.set_config("slack", {"bot_token": "xoxb-org-a-remote-fail", "signing_secret": SIGNING_SECRET})
        svc.set_external_account_id("slack", "T-ORG-A-REMOTE-FAIL")
    finally:
        session.close()

    try:
        async def flow():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                assert await adapters.get_or_refresh("slack", ORG_A) is not None
                body = _slack_event_body("T-ORG-A-REMOTE-FAIL", "Ev-remote-fail-1")
                r = await client.post(
                    "/api/v1/channels/slack/events", content=body,
                    headers=_signed_slack_headers(body, SIGNING_SECRET),
                )
                assert r.status_code == 200
                await drain_background_tasks()

        asyncio.run(flow())

        sends = [p for (m, p) in posted if m == "chat.postMessage"]
        assert len(sends) == 1
        assert "the worker is down" in sends[0]["text"]

        s = get_open_session()
        try:
            row = s.exec(
                select(ChannelEvent).where(ChannelEvent.dedupe_key == "slack:event:Ev-remote-fail-1")
            ).one()
            # The turn ran and produced a result (an error message) — this is
            # not the same "failed" as an unresolved org or an unhandled bug.
            assert row.status == "routed", row.status
        finally:
            s.close()
    finally:
        session = get_open_session()
        try:
            ChannelConfigService(ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A))).delete_config("slack")
        finally:
            session.close()
        _delete_channel_events("slack:event:Ev-remote-fail-1")
        _delete_binding_and_its_rows("slack", "D1", ORG_A)
