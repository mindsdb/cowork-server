"""The lifespan's background work: the self-hosted router probe runs after
bootstrap without delaying readiness, the loop watchdog runs while the
server serves, and the shared root is resolved before the server serves."""

import asyncio
import os
import signal
import threading
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from cowork import server
from cowork.services import router_probe
from cowork.streaming import registry


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

    original = signal.getsignal(signal.SIGTERM)
    yield configure
    # A remote-mode lifespan closes the process-wide registry at shutdown and
    # may install its SIGTERM handler; later tests need neither.
    signal.signal(signal.SIGTERM, original)
    registry.reset()


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


async def test_the_lifespan_runs_the_watchdog_thread_and_stops_it(boot):
    """The watchdog is the lifespan's: started once startup is done, and
    stopped when the server shuts down."""
    from cowork.common import loop_watchdog

    app, _order, _probe = boot()

    def watchdogs() -> set[threading.Thread]:
        return {thread for thread in threading.enumerate() if thread.name == "cowork-loop-watchdog"}

    # An earlier test's watchdog may still be ending.
    earlier = watchdogs()
    async with server.lifespan(app):
        [running] = watchdogs() - earlier
        assert running.daemon and running.is_alive()
    await asyncio.to_thread(running.join, loop_watchdog.STALL_AFTER_S * 2)
    assert not running.is_alive()


async def test_the_lifespan_resolves_the_shared_root_on_a_file_store_thread_before_serving(
    boot, monkeypatch
):
    """Startup resolves the shared root on the file store's threads before
    the server serves, so a database unit's partition check
    (projects.resolved_shared_root) reads the kept answer instead of
    resolving the root on the file store while it holds a pooled connection.
    The session fixture already fills the cache, so this test empties it."""
    from cowork.common.settings.app_settings import get_app_settings
    from cowork.db.units import on_efs_thread
    from cowork.services import projects

    app, _order, _probe = boot()
    monkeypatch.setattr(projects, "_RESOLVED_SHARED_ROOTS", {})
    real_resolve = projects.resolve_shared_root
    on_file_store_thread = []

    def recording_resolve():
        on_file_store_thread.append(on_efs_thread())
        return real_resolve()

    monkeypatch.setattr(projects, "resolve_shared_root", recording_resolve)
    async with server.lifespan(app):
        assert on_file_store_thread == [True]
        configured = get_app_settings().storage.shared_root
        resolved = Path(os.path.realpath(configured))
        assert projects._RESOLVED_SHARED_ROOTS == {configured: resolved}
