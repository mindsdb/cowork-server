"""The instance→hub channel report (ENG-1003).

A hosted instance is stopped after 48h of no dashboard access, which takes its
connected channels offline (and deletes its DNS record, so the webhook URL the
provider holds stops resolving). Inbound webhook traffic never reaches the hub's
registry, so the instance has to say so itself.

The properties worth pinning:

* **Desktop and local dev never phone home.** The whole feature is gated on hub
  environment being present, so this is additive for every other platform.
* **A failed report can never fail the process.** It is telemetry-shaped: worst
  case the exemption expires and the instance hibernates as it does today.
* **Telegram counts as connected.** It doesn't need to be reachable, but it does
  need the process alive, so excluding it would leave Telegram users with the
  same silent death the ticket is about.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from cowork import hub_pin

_ENDPOINT_ENV = "COWORK_HUB_API_URL"
_KEY_ENV = "ANTON_MINDS_API_KEY"


@pytest.fixture()
def hosted(monkeypatch):
    monkeypatch.setenv(_ENDPOINT_ENV, "https://api.mindshub.ai")
    monkeypatch.setenv(_KEY_ENV, "mdb_instance.key")


class TestPlatformGate:
    def test_no_hub_env_means_no_endpoint(self, monkeypatch):
        monkeypatch.delenv(_ENDPOINT_ENV, raising=False)
        assert hub_pin.hub_endpoint() is None

    def test_endpoint_is_built_from_the_hub_base(self, hosted):
        assert hub_pin.hub_endpoint() == "https://api.mindshub.ai/instance/channel-state"

    def test_trailing_slash_does_not_double_up(self, monkeypatch):
        monkeypatch.setenv(_ENDPOINT_ENV, "https://api.mindshub.ai/")
        assert hub_pin.hub_endpoint() == "https://api.mindshub.ai/instance/channel-state"

    def test_start_is_a_noop_without_hub_env(self, monkeypatch):
        monkeypatch.delenv(_ENDPOINT_ENV, raising=False)
        monkeypatch.delenv(_KEY_ENV, raising=False)

        async def _run():
            hub_pin.start()
            try:
                assert hub_pin._task is None
            finally:
                await hub_pin.stop()

        asyncio.run(_run())

    def test_start_is_a_noop_without_a_key(self, monkeypatch):
        # Half-configured is not hosted: reporting without a credential would
        # just generate 401s on every tick.
        monkeypatch.setenv(_ENDPOINT_ENV, "https://api.mindshub.ai")
        monkeypatch.delenv(_KEY_ENV, raising=False)

        async def _run():
            hub_pin.start()
            try:
                assert hub_pin._task is None
            finally:
                await hub_pin.stop()

        asyncio.run(_run())

    def test_report_does_nothing_off_the_hub(self, monkeypatch):
        monkeypatch.delenv(_ENDPOINT_ENV, raising=False)
        called: list = []
        monkeypatch.setattr(hub_pin, "_post", lambda *a: called.append(a))

        assert asyncio.run(hub_pin.report_once()) is None
        assert called == []


class TestReport:
    def _capture(self, monkeypatch, *, connected: bool):
        sent: list = []
        monkeypatch.setattr(hub_pin, "channels_connected", lambda: connected)
        monkeypatch.setattr(
            hub_pin, "_post", lambda endpoint, key, active: sent.append((endpoint, key, active))
        )
        return sent

    def test_reports_true_when_a_channel_is_configured(self, hosted, monkeypatch):
        sent = self._capture(monkeypatch, connected=True)

        assert asyncio.run(hub_pin.report_once()) is True
        assert sent == [
            ("https://api.mindshub.ai/instance/channel-state", "mdb_instance.key", True)
        ]

    def test_reports_false_when_none_are(self, hosted, monkeypatch):
        # The clear direction matters as much: it's what makes the instance
        # eligible for hibernation again after the last channel goes.
        sent = self._capture(monkeypatch, connected=False)

        assert asyncio.run(hub_pin.report_once()) is False
        assert sent[0][2] is False

    def test_a_hub_failure_is_swallowed(self, hosted, monkeypatch):
        monkeypatch.setattr(hub_pin, "channels_connected", lambda: True)

        def _boom(*_args):
            raise OSError("hub unreachable")

        monkeypatch.setattr(hub_pin, "_post", _boom)

        # No exception, and the caller can tell it didn't land.
        assert asyncio.run(hub_pin.report_once()) is None

    def test_a_broken_channel_read_is_swallowed(self, hosted, monkeypatch):
        def _boom():
            raise RuntimeError("db gone")

        monkeypatch.setattr(hub_pin, "channels_connected", _boom)
        monkeypatch.setattr(hub_pin, "_post", lambda *a: None)

        assert asyncio.run(hub_pin.report_once()) is None

    def test_payload_is_a_real_boolean(self, hosted, monkeypatch):
        """The hub rejects a non-boolean channel_active, so the wire shape is
        part of the contract, not a formatting detail."""
        captured: dict = {}

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def read(self):
                return b""

        def _fake_urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data.decode())
            captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
            return _FakeResponse()

        monkeypatch.setattr(hub_pin, "channels_connected", lambda: True)
        monkeypatch.setattr(hub_pin.urllib.request, "urlopen", _fake_urlopen)

        asyncio.run(hub_pin.report_once())

        assert captured["body"] == {"agent": "cowork", "channel_active": True}
        assert captured["headers"]["authorization"] == "Bearer mdb_instance.key"


class TestChannelsConnected:
    def test_true_when_any_channel_is_configured(self, monkeypatch):
        from cowork.services import channels as channels_service

        class _Item:
            def __init__(self, configured):
                self.configured = configured

        class _Status:
            channels = [_Item(False), _Item(True)]

        monkeypatch.setattr(
            channels_service.ChannelConfigService, "status", lambda self: _Status()
        )
        assert hub_pin.channels_connected() is True

    def test_false_when_none_are_configured(self, monkeypatch):
        from cowork.services import channels as channels_service

        class _Status:
            channels = []

        monkeypatch.setattr(
            channels_service.ChannelConfigService, "status", lambda self: _Status()
        )
        assert hub_pin.channels_connected() is False

    def test_telegram_counts(self, monkeypatch):
        """Telegram ingests by outbound long-poll, so it never needed the edge —
        but it does need this process running, so it must pin the instance."""
        from cowork.services import channels as channels_service

        class _Item:
            channel_type = "telegram"
            configured = True

        class _Status:
            channels = [_Item()]

        monkeypatch.setattr(
            channels_service.ChannelConfigService, "status", lambda self: _Status()
        )
        assert hub_pin.channels_connected() is True


class TestNudge:
    def test_nudge_is_safe_when_the_loop_is_not_running(self):
        # Desktop calls this on every channel change; it must be inert.
        hub_pin._nudge = None
        hub_pin.nudge()

    def test_nudge_wakes_the_loop_early(self, hosted, monkeypatch):
        reports: list = []
        monkeypatch.setattr(hub_pin, "REPORT_INTERVAL_SECONDS", 30)

        async def _fake_report():
            reports.append(True)
            return True

        monkeypatch.setattr(hub_pin, "report_once", _fake_report)

        async def _run():
            hub_pin.start()
            try:
                await asyncio.sleep(0)  # let the loop send its boot report
                assert len(reports) == 1
                hub_pin.nudge()
                # Without the nudge this would wait out the 30s interval.
                await asyncio.wait_for(_until(lambda: len(reports) >= 2), timeout=2)
            finally:
                await hub_pin.stop()

        async def _until(predicate):
            while not predicate():
                await asyncio.sleep(0.01)

        asyncio.run(_run())
