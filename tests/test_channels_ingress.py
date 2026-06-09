"""Background ingress: IngressManager lifecycle (poll + stream shapes), the
start/stop reconcile decision, and the Discord Gateway message normaliser.

Async cases run via ``asyncio.run`` inside sync tests, matching the rest of the
channel suite (no pytest-asyncio dependency).
"""
import asyncio

import cowork.channels.plugins.discord as discord
from cowork.channels.ingress import IngressManager, sync_channel_ingress


async def _noop_sink(channel_type, event):
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

    def get(self, channel_type):
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
