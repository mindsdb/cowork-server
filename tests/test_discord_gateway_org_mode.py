"""Discord chat actually works in org mode: a message arriving over the
Gateway (stubbed here — no real websocket) reaches AntonChannelRuntime with
the right org_id and gets a reply, through the exact same IngressManager +
lease + intake_events path a real Gateway connection would use."""
from __future__ import annotations

import asyncio

import httpx
import pytest
from sqlmodel import select

import cowork.channels.runtime as runtime_mod
from cowork.channels.ingress import IngressManager, sync_channel_ingress
from cowork.channels.webhooks import drain_background_tasks
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.db.session import get_open_session
from cowork.models.channel import ChannelBinding, ChannelEvent, ChannelInstallation, ChannelSession
from cowork.services.channels import ChannelConfigService

ORG_A = "8f14e45f-ceea-467e-adde-3fb5ba9302f0"


class FakeHarness:
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


async def _one_shot_stream(event):
    """Stands in for DiscordBridge.stream_events(): yields one batch of
    already-normalized InboundEvent(s) — real stream_events() normalizes
    each Gateway MESSAGE_CREATE via _normalize_message() before yielding,
    it never yields raw Gateway JSON — then hangs (like a real open Gateway
    connection) until cancelled."""
    yield [event]
    await asyncio.sleep(3600)


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


def _delete_installation(channel_type: str, org_id: str) -> None:
    session = get_open_session()
    try:
        row = session.exec(
            select(ChannelInstallation).where(
                ChannelInstallation.channel_type == channel_type,
                ChannelInstallation.org_id == org_id,
            )
        ).first()
        if row is not None:
            session.delete(row)
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


def test_discord_gateway_message_in_org_mode_gets_a_reply(monkeypatch, org_mode):
    fake_harness = FakeHarness()
    monkeypatch.setattr(runtime_mod, "get_harness", lambda _id: fake_harness)

    posted: list[dict] = []
    real_post = httpx.AsyncClient.post

    async def fake_post(self, url, json=None, **kw):
        if not str(url).startswith("https://discord.com/api"):
            return await real_post(self, url, json=json, **kw)
        posted.append(json or {})

        class R:
            status_code = 200

            def json(self):
                return {"id": "999999"}
        return R()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    # No real Redis in this test env — same fake as every other org-lease
    # test in test_channels_ingress.py, so the lease acquire can succeed.
    import fakeredis
    import fakeredis.aioredis as fakeaioredis

    from cowork.channels import ingress_lease

    fake_redis_client = fakeaioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    monkeypatch.setattr(ingress_lease, "get_redis", lambda: fake_redis_client)

    from cowork.server import create_app
    app = create_app()
    adapters = app.state.channel_adapters
    manager = IngressManager(sink=app.state.channel_runtime.handle)

    session = get_open_session()
    try:
        ChannelConfigService(ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A))).set_config(
            "discord", {"bot_token": "discord-bot-token-org-a"}
        )
    finally:
        session.close()

    try:
        async def flow():
            bridge = await adapters.get_or_refresh("discord", ORG_A)
            assert bridge is not None
            # No guild_id → DM (is_mention=True); a fresh binding defaults to
            # trigger_rule="always", so the turn runs without needing a mention.
            normalized = bridge._normalize_message({
                "id": "55", "channel_id": "D1", "content": "hello bot",
                "author": {"id": "7", "bot": False}, "mentions": [],
                "timestamp": "2026-08-20T00:00:00+00:00",
            })
            assert normalized is not None
            monkeypatch.setattr(bridge, "stream_events", lambda: _one_shot_stream(normalized))

            await sync_channel_ingress(manager, adapters, "discord", ORG_A)
            assert manager.is_running("discord", ORG_A)

            await asyncio.wait_for(_until_called(fake_harness), 1.0)
            await drain_background_tasks()

        asyncio.run(flow())

        assert len(fake_harness.calls) == 1, "the turn must actually run"

        assert len(posted) == 1
        assert "pong from the org bot" in posted[0]["content"]

        s = get_open_session()
        try:
            row = s.exec(
                select(ChannelEvent).where(ChannelEvent.dedupe_key == "discord:message:55")
            ).one()
            assert row.status == "routed", row.status
            assert row.org_id == ORG_A
        finally:
            s.close()
    finally:
        asyncio.run(manager.stop_all())
        session = get_open_session()
        try:
            ChannelConfigService(
                ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A))
            ).delete_config("discord")
        finally:
            session.close()
        _delete_channel_events("discord:message:55")
        _delete_binding_and_its_rows("discord", "D1", ORG_A)
        _delete_installation("discord", ORG_A)


async def _until_called(fake_harness, poll_interval: float = 0.02) -> None:
    while not fake_harness.calls:
        await asyncio.sleep(poll_interval)


def test_a_second_orgs_manager_cannot_also_run_the_same_orgs_gateway(monkeypatch, org_mode):
    """The lease, not just the adapter cache, is what a second replica would
    actually hit — prove a second IngressManager sharing the same lease
    backend is refused ownership of an org another manager already owns."""
    import fakeredis
    import fakeredis.aioredis as fakeaioredis

    from cowork.channels import ingress_lease

    server = fakeredis.FakeServer()
    client = fakeaioredis.FakeRedis(server=server, decode_responses=True)
    monkeypatch.setattr(ingress_lease, "get_redis", lambda: client)

    class _StubBridge:
        def __init__(self):
            self.opened = asyncio.Event()

        async def stream_events(self):
            self.opened.set()
            await asyncio.sleep(3600)
            yield []

        def dedupe_key(self, event):
            return None

    async def scenario():
        replica_1 = IngressManager(sink=lambda *a, **kw: asyncio.sleep(0))
        replica_2 = IngressManager(sink=lambda *a, **kw: asyncio.sleep(0))
        bridge_1, bridge_2 = _StubBridge(), _StubBridge()

        await replica_1.start("discord", bridge_1, org_id=ORG_A)
        await replica_2.start("discord", bridge_2, org_id=ORG_A)

        assert replica_1.is_running("discord", ORG_A)
        assert not replica_2.is_running("discord", ORG_A)
        assert not bridge_2.opened.is_set()

        await replica_1.stop_all()
        await replica_2.stop_all()

    asyncio.run(scenario())
