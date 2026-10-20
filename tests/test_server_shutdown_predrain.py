"""Ending remote turns before Uvicorn's shutdown drain.

Uvicorn closes its listeners on SIGTERM, waits at most
--timeout-graceful-shutdown for open responses, cuts the rest mid-stream, and
only then runs the lifespan's shutdown. With remote turns the server takes the
first SIGTERM itself and ends its turns while their answer streams are still
open, then hands the signal to Uvicorn. Desktop and in-process deployments
keep Uvicorn's handling unchanged. tests/test_turn_shutdown_process.py runs the
whole sequence in a real process.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import threading
from types import SimpleNamespace

import fakeredis.aioredis
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork import scheduler, server
from cowork.streaming import cancellation
from cowork.streaming.registry import (
    RunHandle, RunIdentity, ServerShuttingDown, TurnLifecycle, registry,
)
from cowork.turnqueue.redis_client import cancel_flag_key

SERVER_LOGGER = "cowork.common.logger"


@pytest.fixture(autouse=True)
def _restore_sigterm_and_registry(monkeypatch):
    original = signal.getsignal(signal.SIGTERM)
    # The migration tests' alembic env calls fileConfig, which disables every
    # logger that existed before it for the rest of the session.
    monkeypatch.setattr(logging.getLogger(SERVER_LOGGER), "disabled", False)
    yield
    signal.signal(signal.SIGTERM, original)
    registry.reset()


@pytest.fixture
def controller_flags(monkeypatch):
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(cancellation, "get_redis", lambda: client)
    return client


async def _until(condition) -> None:
    async with asyncio.timeout(2):
        while not condition():
            await asyncio.sleep(0)


def _server_records(caplog, level: int) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == SERVER_LOGGER and r.levelno == level]


async def test_the_first_sigterm_ends_turns_before_uvicorn_sees_it(monkeypatch):
    handed_on: list[int] = []
    release = asyncio.Event()
    drain: list[str] = []

    def uvicorn_handle_exit(signum, frame):
        handed_on.append(signum)

    async def end_remote_turns():
        drain.append("started")
        await release.wait()
        drain.append("finished")

    signal.signal(signal.SIGTERM, uvicorn_handle_exit)
    monkeypatch.setattr(server, "_end_remote_turns", end_remote_turns)
    server._end_remote_turns_on_sigterm()
    assert signal.getsignal(signal.SIGTERM) is not uvicorn_handle_exit

    signal.raise_signal(signal.SIGTERM)
    await _until(lambda: drain)
    assert handed_on == [], "Uvicorn saw SIGTERM while the turns were still ending"

    # A second SIGTERM while they end goes straight to Uvicorn.
    signal.raise_signal(signal.SIGTERM)
    assert handed_on == [signal.SIGTERM]

    release.set()
    await _until(lambda: len(handed_on) == 2)
    assert drain == ["started", "finished"]


async def test_a_failed_pre_drain_still_hands_sigterm_on(monkeypatch, caplog):
    handed_on: list[int] = []

    async def end_remote_turns():
        raise RuntimeError("redis is down")

    signal.signal(signal.SIGTERM, lambda signum, frame: handed_on.append(signum))
    monkeypatch.setattr(server, "_end_remote_turns", end_remote_turns)
    server._end_remote_turns_on_sigterm()

    with caplog.at_level(logging.ERROR, logger=SERVER_LOGGER):
        signal.raise_signal(signal.SIGTERM)
        await _until(lambda: handed_on)

    assert handed_on == [signal.SIGTERM]
    assert _server_records(caplog, logging.ERROR) == ["Ending turns before shutdown failed"]


@pytest.mark.parametrize("previous", [signal.SIG_IGN, signal.SIG_DFL], ids=["ignored", "default"])
async def test_nothing_is_installed_without_a_handler_to_hand_on_to(previous, caplog):
    signal.signal(signal.SIGTERM, previous)

    with caplog.at_level(logging.WARNING, logger=SERVER_LOGGER):
        server._end_remote_turns_on_sigterm()

    assert signal.getsignal(signal.SIGTERM) == previous
    assert _server_records(caplog, logging.WARNING) == [
        f"SIGTERM handler is {previous!r}, not one to hand the signal on to; turns in "
        "progress are ended only after the server's request drain"
    ]


async def test_nothing_is_installed_off_the_main_thread(caplog):
    def uvicorn_handle_exit(signum, frame):
        pass

    signal.signal(signal.SIGTERM, uvicorn_handle_exit)
    with caplog.at_level(logging.WARNING, logger=SERVER_LOGGER):
        thread = threading.Thread(target=server._end_remote_turns_on_sigterm)
        thread.start()
        thread.join()

    assert signal.getsignal(signal.SIGTERM) is uvicorn_handle_exit
    assert _server_records(caplog, logging.WARNING) == [
        "Server lifespan runs off the main thread, where no signal handler can be set; "
        "turns in progress are ended only after the server's request drain"
    ]


@pytest.fixture
def quiet_lifespan(monkeypatch):
    """server.lifespan with the startup and teardown work that has nothing to
    do with turns stubbed, as tests/turn_shutdown_process_app.py does."""
    import cowork.coding.service as coding_service

    async def _noop(*args, **kwargs):
        return None

    async def _no_model_map():
        return False

    async def _start_channels(app):
        app.state.channel_ingress = SimpleNamespace(stop_all=_noop)
        app.state.channel_adapters = SimpleNamespace(shutdown=_noop)

    monkeypatch.setattr(server, "run_dev_setup", lambda: None)
    monkeypatch.setattr(server, "consume_pending_credential_clear", lambda *args: None)
    monkeypatch.setattr(server, "_warm_agent_runtime", lambda: None)
    monkeypatch.setattr(server, "_warm_model_map_on_boot", _no_model_map)
    monkeypatch.setattr(server, "start_scheduler", lambda: None)
    monkeypatch.setattr(server, "_start_channels", _start_channels)
    monkeypatch.setattr(coding_service, "get_coding_service", lambda: SimpleNamespace(close_all=lambda: None))


@pytest.mark.parametrize("backend", ["inprocess", "remote"])
async def test_only_remote_turns_take_sigterm_from_uvicorn(monkeypatch, quiet_lifespan, backend):
    """Desktop and in-process deployments keep Uvicorn's own SIGTERM handling;
    only a server whose turns run remotely puts the pre-drain in front of it."""
    def uvicorn_handle_exit(signum, frame):
        pass

    monkeypatch.setenv("COWORK_TURN_BACKEND", backend)
    signal.signal(signal.SIGTERM, uvicorn_handle_exit)

    async with server.lifespan(FastAPI()):
        during = signal.getsignal(signal.SIGTERM)

    if backend == "inprocess":
        assert during is uvicorn_handle_exit
    else:
        assert during is not uvicorn_handle_exit


async def test_ending_remote_turns_refuses_new_ones_and_lets_their_scheduled_runs_finish(
    monkeypatch, controller_flags, caplog,
):
    lifecycle = TurnLifecycle(identity=RunIdentity(remote=True))

    async def relay():
        await asyncio.Event().wait()

    turn = asyncio.create_task(relay())
    await asyncio.sleep(0)
    registry._by_cid["c"] = RunHandle(
        conversation_id="c", turn_id=0, buffer=SimpleNamespace(is_closed=False), task=turn,
        lifecycle=lifecycle,
    )
    run_saw: list[bool] = []

    async def scheduled_run():
        # A scheduled run drains its turn's stream, then records the outcome.
        await asyncio.gather(turn, return_exceptions=True)
        await asyncio.sleep(0.05)
        run_saw.append(lifecycle.shutting_down)

    run = asyncio.create_task(scheduled_run())
    scheduler._scheduled_runs.add(run)
    run.add_done_callback(scheduler._scheduled_runs.discard)
    poll_loop = asyncio.create_task(asyncio.Event().wait())
    monkeypatch.setattr(scheduler, "_scheduler_task", poll_loop)

    with caplog.at_level(logging.WARNING, logger=SERVER_LOGGER):
        await server._end_remote_turns()
        # The lifespan's teardown runs it again; nothing is left to do.
        await server._end_remote_turns()

    assert poll_loop.cancelled()
    assert turn.cancelled()
    assert run.done() and run_saw == [True]
    assert await controller_flags.exists(cancel_flag_key(lifecycle.identity.correlation_id))
    assert _server_records(caplog, logging.WARNING) == ["Cancelled 1 in-flight turn(s) for shutdown"]
    with pytest.raises(ServerShuttingDown):
        await registry.start(
            conversation_id="d", turn_id=0, open_buffer=None, produce=None,
        )


def test_a_turn_refused_for_shutdown_is_answered_as_a_busy_server():
    """The refusal a server too busy to take a turn gives: the web UI waits
    Retry-After, and the retry reaches another replica."""
    app = server.create_app()

    async def refused():
        raise ServerShuttingDown()

    app.add_api_route("/__shutting_down", refused)
    response = TestClient(app).get("/__shutting_down")

    assert response.status_code == 503
    assert response.json() == {"detail": "Cowork is busy. Try again in about 1 second.", "code": "server_busy"}
    assert response.headers["Retry-After"] == "1"
