"""The self-hosted router probe runs after bootstrap without delaying readiness."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from cowork import server
from cowork.services import router_probe


@pytest.fixture
def boot(monkeypatch):
    # Keep the real lifespan and its isolated DB recovery/shutdown; skip the
    # migrations and unrelated background work, which have their own tests.
    app = server.create_app()
    order = []
    monkeypatch.setattr(server, "run_dev_setup", lambda: order.append("bootstrap"))
    monkeypatch.setattr(
        server, "consume_pending_credential_clear", lambda *args: order.append("credentials")
    )
    monkeypatch.setattr(server, "_warm_agent_runtime", lambda: None)
    monkeypatch.setattr(server, "_warm_model_map_on_boot", AsyncMock(return_value=False))
    monkeypatch.setattr(server, "start_scheduler", lambda: None)
    monkeypatch.setattr(server, "_start_channels", AsyncMock())
    monkeypatch.setattr(server, "_run_artifact_owner_backfill", AsyncMock())
    monkeypatch.setattr("cowork.channels.ingress.start_reconciler", lambda *args: None)

    monkeypatch.setattr(router_probe, "run_router_startup_probe", AsyncMock())

    settings = server.get_app_settings()

    def configure(*, kind="self-hosted", tenancy="local", backend="inprocess", channel=""):
        monkeypatch.setattr(
            server,
            "get_app_settings",
            lambda: settings.model_copy(
                update={
                    "deployment_kind": kind,
                    "tenancy_mode": tenancy,
                    "install_channel_override": channel,
                }
            ),
        )
        monkeypatch.setenv("COWORK_TURN_BACKEND", backend)
        return app, order, router_probe

    return configure


async def test_probe_starts_after_bootstrap_without_blocking_and_is_drained_at_shutdown(
    boot, monkeypatch
):
    app, order, probe_module = boot()
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def waiting_probe():
        order.append("probe")
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            # A cancellation alone is insufficient: shutdown must await cleanup.
            await asyncio.sleep(0)
            stopped.set()

    async def stop_ingress():
        assert stopped.is_set()

    monkeypatch.setattr(app.state.channel_ingress, "stop_all", stop_ingress)
    probe_module.run_router_startup_probe.side_effect = waiting_probe
    async with asyncio.timeout(2):
        async with server.lifespan(app):
            task = app.state.router_startup_probe
            await started.wait()
            assert order == ["bootstrap", "credentials", "probe"]
            assert not task.done()
            assert not stopped.is_set()
        assert task.cancelled()
        assert stopped.is_set()
    probe_module.run_router_startup_probe.assert_awaited_once_with()


@pytest.mark.parametrize(
    "kind,tenancy,backend,channel",
    [
        (None, "local", "inprocess", ""),
        (None, "local", "inprocess", "desktop"),
        (None, "local", "inprocess", "hosted"),
        ("self-hosted", "org", "inprocess", ""),
        ("self-hosted", "local", "remote", ""),
    ],
    ids=["default", "desktop", "hosted-local", "org", "remote"],
)
async def test_probe_is_absent_outside_local_self_hosted(boot, kind, tenancy, backend, channel):
    app, _, probe_module = boot(kind=kind, tenancy=tenancy, backend=backend, channel=channel)
    async with server.lifespan(app):
        await asyncio.sleep(0)
        assert getattr(app.state, "router_startup_probe", None) is None
    probe_module.run_router_startup_probe.assert_not_called()
