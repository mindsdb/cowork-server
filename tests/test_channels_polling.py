"""Server-side (tunnel-free) polling: PollManager lifecycle, the start/stop
reconcile decision, and Telegram's getUpdates poll cycle.

Async cases run via ``asyncio.run`` inside sync tests, matching the rest of the
channel suite (no pytest-asyncio dependency).
"""
import asyncio

import cowork.channels.plugins.telegram as tg
from cowork.channels.polling import PollManager, sync_channel_polling


async def _noop_sink(channel_type, event):
    return None


class _FakeBridge:
    """Pollable adapter whose poll returns nothing but signals it ran."""

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


class _FakeAdapters:
    def __init__(self, by_type):
        self._by_type = by_type

    def get(self, channel_type):
        return self._by_type.get(channel_type)


def test_poll_manager_start_stop_lifecycle():
    async def scenario():
        mgr = PollManager(sink=_noop_sink)
        bridge = _FakeBridge()
        await mgr.start("telegram", bridge)
        assert mgr.is_polling("telegram")
        await asyncio.wait_for(bridge.polled.wait(), 1.0)
        # Starting again while running is a no-op (no second task).
        await mgr.start("telegram", bridge)
        await mgr.stop("telegram")
        assert not mgr.is_polling("telegram")
        assert bridge.calls >= 1

    asyncio.run(scenario())


def test_poll_manager_ignores_non_pollable_adapter():
    async def scenario():
        mgr = PollManager(sink=_noop_sink)
        await mgr.start("slack", object())  # no poll() method
        assert not mgr.is_polling("slack")

    asyncio.run(scenario())


def test_sync_channel_polling_decision(monkeypatch):
    def _public_url(value):
        monkeypatch.setattr(
            "cowork.channels.polling.get_app_settings",
            lambda: type("S", (), {"public_base_url": value})(),
        )

    async def scenario():
        mgr = PollManager(sink=_noop_sink)
        adapters = _FakeAdapters({"telegram": _FakeBridge()})

        # No public URL + pollable adapter → poll.
        _public_url("")
        await sync_channel_polling(mgr, adapters, "telegram")
        assert mgr.is_polling("telegram")

        # A public URL exists → webhook owns ingress, stop polling.
        _public_url("https://hooks.example.com")
        await sync_channel_polling(mgr, adapters, "telegram")
        assert not mgr.is_polling("telegram")

        # No live adapter → stays stopped.
        _public_url("")
        await sync_channel_polling(mgr, _FakeAdapters({}), "telegram")
        assert not mgr.is_polling("telegram")

        await mgr.stop_all()

    asyncio.run(scenario())


def test_telegram_poll_parses_and_advances_offset(monkeypatch):
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

    monkeypatch.setattr(tg.httpx, "AsyncClient", _FakeClient)
    bridge = tg.TelegramBridge({"bot_token": "t", "secret_token": "s"})

    events, offset = asyncio.run(bridge.poll(offset=None))
    assert offset == 11
    assert len(events) == 1
    assert events[0].message.content == "hi"
    # First cycle clears any stale webhook so getUpdates is permitted.
    assert calls["deleteWebhook"] == 1
