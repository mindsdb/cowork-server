"""Background ingress: IngressManager lifecycle (poll + stream shapes), the
start/stop reconcile decision, and the Discord Gateway message normaliser.

Async cases run via ``asyncio.run`` inside sync tests, matching the rest of the
channel suite (no pytest-asyncio dependency).
"""
import asyncio

import cowork.channels.plugins.discord as discord
import cowork.channels.plugins.slack as slack
import cowork.channels.plugins.telegram as telegram
from cowork.channels.ingress import IngressManager, sync_channel_ingress


async def _noop_sink(channel_type, event, org_id=None):
    return None


class _FakePollBridge:
    """Poll-shaped adapter (Telegram-style): one cycle returns no events."""

    def __init__(self):
        self.calls = 0
        self.polled = asyncio.Event()

    async def poll(self, *, offset):
        self.calls += 1
        self.polled.set()
        await asyncio.sleep(0.01)
        return [], offset

    def dedupe_key(self, event):
        return None


class _FakeStreamBridge:
    """Stream-shaped adapter (Discord-style): a persistent connection that
    signals it opened and then stays open until cancelled."""

    def __init__(self):
        self.opened = asyncio.Event()

    async def stream_events(self):
        self.opened.set()
        await asyncio.sleep(3600)
        yield []  # unreachable; makes this an async generator

    def dedupe_key(self, event):
        return None


class _FakeAdapters:
    def __init__(self, by_type):
        self._by_type = by_type

    def get(self, channel_type, org_id=None):
        return self._by_type.get(channel_type)


def test_ingress_manager_poll_lifecycle():
    async def scenario():
        mgr = IngressManager(sink=_noop_sink)
        bridge = _FakePollBridge()
        await mgr.start("telegram", bridge)
        assert mgr.is_running("telegram")
        await asyncio.wait_for(bridge.polled.wait(), 1.0)
        await mgr.start("telegram", bridge)  # idempotent — no second task
        await mgr.stop("telegram")
        assert not mgr.is_running("telegram")
        assert bridge.calls >= 1

    asyncio.run(scenario())


def test_ingress_manager_stream_lifecycle():
    async def scenario():
        mgr = IngressManager(sink=_noop_sink)
        bridge = _FakeStreamBridge()
        await mgr.start("discord", bridge)
        assert mgr.is_running("discord")
        await asyncio.wait_for(bridge.opened.wait(), 1.0)
        await mgr.stop("discord")
        assert not mgr.is_running("discord")

    asyncio.run(scenario())


def test_ingress_manager_ignores_non_ingestible_adapter():
    async def scenario():
        mgr = IngressManager(sink=_noop_sink)
        await mgr.start("slack", object())  # no poll() / stream_events()
        assert not mgr.is_running("slack")

    asyncio.run(scenario())


def test_sync_channel_ingress_decision(monkeypatch):
    def _public_url(value):
        monkeypatch.setattr(
            "cowork.channels.ingress.get_app_settings",
            lambda: type("S", (), {"public_base_url": value})(),
        )

    async def scenario():
        mgr = IngressManager(sink=_noop_sink)

        # Poll adapter: polls only when no public URL (else webhook owns ingress).
        poll_adapters = _FakeAdapters({"telegram": _FakePollBridge()})
        _public_url("")
        await sync_channel_ingress(mgr, poll_adapters, "telegram")
        assert mgr.is_running("telegram")
        _public_url("https://hooks.example.com")
        await sync_channel_ingress(mgr, poll_adapters, "telegram")
        assert not mgr.is_running("telegram")

        # Stream adapter (Gateway): runs whenever active, even with a public URL.
        stream_adapters = _FakeAdapters({"discord": _FakeStreamBridge()})
        _public_url("https://hooks.example.com")
        await sync_channel_ingress(mgr, stream_adapters, "discord")
        assert mgr.is_running("discord")

        # No live adapter → stopped.
        await sync_channel_ingress(mgr, _FakeAdapters({}), "discord")
        assert not mgr.is_running("discord")

        await mgr.stop_all()

    asyncio.run(scenario())


def test_discord_gateway_normalize_message():
    bridge = discord.DiscordBridge({"bot_token": "t"})
    bridge._bot_user_id = "999"

    # Normal guild message, bot not mentioned.
    ev = bridge._normalize_message({
        "id": "55", "channel_id": "42", "content": "hello",
        "author": {"id": "7", "bot": False}, "guild_id": "1", "mentions": [],
        "timestamp": "2026-06-08T00:00:00+00:00",
    })
    assert ev is not None
    assert ev.message.content == "hello"
    assert ev.address.platform_id == "42"
    assert ev.message.is_group is True
    assert ev.message.is_mention is False
    assert getattr(ev, "_dedupe_key") == "discord:message:55"

    # Bot-authored and our own messages are skipped (no echo loops).
    assert bridge._normalize_message(
        {"id": "1", "channel_id": "42", "content": "x", "author": {"id": "5", "bot": True}}
    ) is None
    assert bridge._normalize_message(
        {"id": "2", "channel_id": "42", "content": "x", "author": {"id": "999"}}
    ) is None

    # Explicit @-mention in a guild → mention; a DM (no guild) → always mention.
    mentioned = bridge._normalize_message({
        "id": "3", "channel_id": "42", "content": "hey", "author": {"id": "7"},
        "guild_id": "1", "mentions": [{"id": "999"}],
    })
    assert mentioned.message.is_mention is True
    dm = bridge._normalize_message(
        {"id": "4", "channel_id": "42", "content": "hi", "author": {"id": "7"}}
    )
    assert dm.message.is_mention is True


def test_slack_stream_events_bound_only_with_app_token():
    # Webhook-only install (no app_token) must NOT advertise stream_events, so
    # the ingress manager leaves it to the webhook and doesn't spin a loop.
    webhook_only = slack.SlackBridge({"signing_secret": "s", "bot_token": "xoxb-1"})
    assert not callable(getattr(webhook_only, "stream_events", None))

    # With an app-level token, Socket Mode ingress is enabled (duck-typed).
    socket_mode = slack.SlackBridge({"bot_token": "xoxb-1", "app_token": "xapp-1"})
    assert callable(getattr(socket_mode, "stream_events", None))


def test_telegram_poll_parses_and_advances_offset(monkeypatch):
    # Drives the real poll() to cover response parsing + offset advancement
    # (regression guard: a stuck offset re-fetches the same updates every cycle).
    calls = {"deleteWebhook": 0}

    class _Resp:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, params=None):
            return _Resp({"ok": True, "result": [{
                "update_id": 10,
                "message": {
                    "message_id": 1,
                    "chat": {"id": 99, "type": "private"},
                    "from": {"id": 5},
                    "text": "hi",
                    "date": 0,
                },
            }]})

        async def post(self, url, json=None):
            calls["deleteWebhook"] += 1
            return _Resp({"ok": True})

    monkeypatch.setattr(telegram.httpx, "AsyncClient", _FakeClient)
    bridge = telegram.TelegramBridge({"bot_token": "t", "secret_token": "s"})

    events, offset = asyncio.run(bridge.poll(offset=None))
    assert offset == 11                       # advanced past update_id 10
    assert len(events) == 1
    assert events[0].message.content == "hi"
    assert calls["deleteWebhook"] == 1        # first cycle clears any stale webhook

    # A subsequent cycle passes the advanced offset (and skips deleteWebhook).
    events2, offset2 = asyncio.run(bridge.poll(offset=offset))
    assert offset2 == 11
    assert calls["deleteWebhook"] == 1


def test_telegram_factory_needs_only_bot_token():
    run = lambda creds: asyncio.run(telegram._factory(creds))
    assert run({}) is None                                  # no bot_token
    assert run({"bot_token": "t"}) is not None              # secret_token NOT required (polling)
    assert run({"bot_token": "t", "secret_token": "s"}) is not None


def test_slack_factory_requires_bot_token_and_an_ingress_cred():
    run = lambda creds: asyncio.run(slack._factory(creds))
    # Missing bot_token, or no ingress credential at all → not configured.
    assert run({"signing_secret": "s"}) is None
    assert run({"bot_token": "xoxb-1"}) is None
    # Either a signing_secret (webhook) or an app_token (Socket Mode) suffices.
    assert run({"bot_token": "xoxb-1", "signing_secret": "s"}) is not None
    assert run({"bot_token": "xoxb-1", "app_token": "xapp-1"}) is not None


def test_slack_event_from_callback():
    bridge = slack.SlackBridge({"bot_token": "xoxb-1"})

    # A normal channel message → InboundEvent with the event_id dedupe key.
    # The same envelope arrives via the webhook and via Socket Mode.
    ev = bridge._event_from_callback({
        "type": "event_callback",
        "event_id": "Ev123",
        "event": {"type": "message", "channel": "C42", "user": "U7", "text": "hello", "ts": "100.5"},
    })
    assert ev is not None
    assert ev.message.content == "hello"
    assert ev.address.platform_id == "C42"
    assert ev.message.is_group is True
    assert getattr(ev, "_dedupe_key") == "slack:event:Ev123"

    # Bot-authored messages are skipped (no echo loops).
    assert bridge._event_from_callback({
        "type": "event_callback",
        "event": {"type": "message", "channel": "C42", "bot_id": "B1", "text": "x", "ts": "1"},
    }) is None

    # Non-event envelopes (e.g. a bare url_verification) are ignored here.
    assert bridge._event_from_callback({"type": "url_verification", "challenge": "abc"}) is None


def test_ingress_manager_org_scoped_lease_mutual_exclusion(monkeypatch):
    import fakeredis
    import fakeredis.aioredis as fakeaioredis

    from cowork.channels import ingress_lease

    server = fakeredis.FakeServer()
    client = fakeaioredis.FakeRedis(server=server, decode_responses=True)
    monkeypatch.setattr(ingress_lease, "get_redis", lambda: client)

    async def scenario():
        mgr1 = IngressManager(sink=_noop_sink)
        mgr2 = IngressManager(sink=_noop_sink)
        bridge1 = _FakeStreamBridge()
        bridge2 = _FakeStreamBridge()

        await mgr1.start("discord", bridge1, org_id="org-a")
        await mgr2.start("discord", bridge2, org_id="org-a")

        assert mgr1.is_running("discord", "org-a")
        assert not mgr2.is_running("discord", "org-a")
        assert not bridge2.opened.is_set()

        await mgr1.stop("discord", "org-a")
        assert not mgr1.is_running("discord", "org-a")

        # The lease was released on stop — mgr2 can now acquire it.
        await mgr2.start("discord", bridge2, org_id="org-a")
        assert mgr2.is_running("discord", "org-a")
        await mgr2.stop("discord", "org-a")

    asyncio.run(scenario())


def test_ingress_manager_org_scoped_lease_failover_on_renew_loss(monkeypatch):
    import fakeredis
    import fakeredis.aioredis as fakeaioredis

    from cowork.channels import ingress_lease

    server = fakeredis.FakeServer()
    client = fakeaioredis.FakeRedis(server=server, decode_responses=True)
    monkeypatch.setattr(ingress_lease, "get_redis", lambda: client)
    monkeypatch.setattr(ingress_lease, "RENEW_INTERVAL_S", 0.01)
    monkeypatch.setattr(ingress_lease, "LEASE_TTL_S", 0.03)

    async def scenario():
        mgr1 = IngressManager(sink=_noop_sink)
        bridge1 = _FakeStreamBridge()
        await mgr1.start("discord", bridge1, org_id="org-a")
        assert mgr1.is_running("discord", "org-a")

        # Simulate another replica stealing the lease after mgr1's TTL lapsed.
        await client.delete(ingress_lease._key("discord", "org-a"))
        await ingress_lease.acquire("discord", "org-a", "someone-else")

        # mgr1's next renewal tick must notice and stop its own stream loop.
        await asyncio.sleep(0.05)
        assert not mgr1.is_running("discord", "org-a")

        mgr2 = IngressManager(sink=_noop_sink)
        bridge2 = _FakeStreamBridge()
        await mgr2.start("discord", bridge2, org_id="org-a")
        assert mgr2.is_running("discord", "org-a")
        await mgr2.stop("discord", "org-a")

    asyncio.run(scenario())


def test_ingress_manager_local_mode_never_touches_the_lease(monkeypatch):
    # org_id=None must skip the lease path entirely — if it didn't, this
    # would blow up (get_redis is never patched in this test).
    async def scenario():
        mgr = IngressManager(sink=_noop_sink)
        bridge = _FakeStreamBridge()
        await mgr.start("discord", bridge)
        assert mgr.is_running("discord")
        await mgr.stop("discord")

    asyncio.run(scenario())


def test_reconcile_once_starts_configured_orgs_and_stops_deleted_ones(monkeypatch):
    import fakeredis
    import fakeredis.aioredis as fakeaioredis

    from cowork.channels import ingress_lease
    from cowork.channels.ingress import reconcile_once
    from cowork.db.session import get_open_session
    from cowork.models.channel import ChannelInstallation

    server = fakeredis.FakeServer()
    client = fakeaioredis.FakeRedis(server=server, decode_responses=True)
    monkeypatch.setattr(ingress_lease, "get_redis", lambda: client)

    session = get_open_session()
    try:
        row = ChannelInstallation(
            channel_type="discord", display_name="Discord", org_id="org-recon-1",
        )
        session.add(row)
        session.commit()
        row_id = row.id
    finally:
        session.close()

    class _FakeOrgAdapters:
        """No cache yet for this org — get() misses until get_or_refresh()
        populates it, exactly like the real LiveAdapterRegistry."""

        def __init__(self):
            self.bridge = _FakeStreamBridge()
            self._cached = False

        def get(self, channel_type, org_id=None):
            if self._cached and channel_type == "discord" and org_id == "org-recon-1":
                return self.bridge
            return None

        async def get_or_refresh(self, channel_type, org_id, *, session=None):
            if channel_type == "discord" and org_id == "org-recon-1":
                self._cached = True
                return self.bridge
            return None

    try:
        async def scenario():
            mgr = IngressManager(sink=_noop_sink)
            adapters = _FakeOrgAdapters()

            await reconcile_once(mgr, adapters)
            assert mgr.is_running("discord", "org-recon-1")

            cleanup = get_open_session()
            try:
                stored = cleanup.get(ChannelInstallation, row_id)
                cleanup.delete(stored)
                cleanup.commit()
            finally:
                cleanup.close()

            await reconcile_once(mgr, adapters)
            assert not mgr.is_running("discord", "org-recon-1")

        asyncio.run(scenario())
    finally:
        cleanup = get_open_session()
        try:
            stored = cleanup.get(ChannelInstallation, row_id)
            if stored is not None:
                cleanup.delete(stored)
                cleanup.commit()
        finally:
            cleanup.close()


def test_reconciler_task_can_be_started_and_cancelled():
    from cowork.channels.ingress import start_reconciler

    async def scenario():
        mgr = IngressManager(sink=_noop_sink)
        task = start_reconciler(mgr, _FakeAdapters({}), interval_s=0.01)
        await asyncio.sleep(0.03)
        assert not task.done()
        task.cancel()
        with __import__("contextlib").suppress(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_reconcile_once_with_multiple_orgs(monkeypatch):
    import fakeredis
    import fakeredis.aioredis as fakeaioredis

    from cowork.channels import ingress_lease
    from cowork.channels.ingress import reconcile_once
    from cowork.db.session import get_open_session
    from cowork.models.channel import ChannelInstallation

    server = fakeredis.FakeServer()
    client = fakeaioredis.FakeRedis(server=server, decode_responses=True)
    monkeypatch.setattr(ingress_lease, "get_redis", lambda: client)

    session = get_open_session()
    row_ids = []
    try:
        row1 = ChannelInstallation(
            channel_type="discord", display_name="Discord", org_id="org-multi-1",
        )
        row2 = ChannelInstallation(
            channel_type="slack", display_name="Slack", org_id="org-multi-2",
        )
        session.add(row1)
        session.add(row2)
        session.commit()
        row_ids = [row1.id, row2.id]
    finally:
        session.close()

    class _FakeMultiOrgAdapters:
        """Adapters for two distinct orgs."""

        def __init__(self):
            self.bridge1 = _FakeStreamBridge()
            self.bridge2 = _FakeStreamBridge()
            self._cached = set()

        def get(self, channel_type, org_id=None):
            if channel_type == "discord" and org_id == "org-multi-1" and "org-multi-1" in self._cached:
                return self.bridge1
            if channel_type == "slack" and org_id == "org-multi-2" and "org-multi-2" in self._cached:
                return self.bridge2
            return None

        async def get_or_refresh(self, channel_type, org_id, *, session=None):
            if channel_type == "discord" and org_id == "org-multi-1":
                self._cached.add("org-multi-1")
                return self.bridge1
            if channel_type == "slack" and org_id == "org-multi-2":
                self._cached.add("org-multi-2")
                return self.bridge2
            return None

    try:
        async def scenario():
            mgr = IngressManager(sink=_noop_sink)
            adapters = _FakeMultiOrgAdapters()

            await reconcile_once(mgr, adapters)
            assert mgr.is_running("discord", "org-multi-1")
            assert mgr.is_running("slack", "org-multi-2")

        asyncio.run(scenario())
    finally:
        cleanup = get_open_session()
        try:
            for row_id in row_ids:
                stored = cleanup.get(ChannelInstallation, row_id)
                if stored is not None:
                    cleanup.delete(stored)
            cleanup.commit()
        finally:
            cleanup.close()
