"""The loop watchdog names a stall's culprit and a full limiter before the
kubelet restarts the pod.

Each test runs the real watchdog thread with its real timings, so a stall
takes STALL_AFTER_S to report and a full limiter SATURATED_AFTER_S.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from types import SimpleNamespace

import anyio
import httpx

from cowork.common import loop_watchdog, request_logging
from cowork.common.loop_watchdog import LoopWatchdog
from cowork.common.request_logging import RequestLoggingApp, _InFlight
from cowork.db.units import loop_limiters

STALL_SECONDS = 3.0


def _records(caplog, event: str) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == "cowork.common.loop_watchdog" and getattr(record, "event", None) == event
    ]


async def test_a_blocking_call_in_an_async_route_logs_one_stall_naming_it(caplog):
    caplog.set_level(logging.WARNING, logger="cowork.common.loop_watchdog")
    app = RequestLoggingApp()

    @app.get("/stalls")
    async def stalls_the_loop():
        time.sleep(STALL_SECONDS)
        return {}

    watchdog = LoopWatchdog({})
    watchdog.start()
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/stalls")).status_code == 200
        # The next ping, answered, is what reports the recovery.
        await asyncio.sleep(loop_watchdog.PING_EVERY_S * 2)
    finally:
        watchdog.stop()

    stalled = _records(caplog, "event_loop_stalled")
    recovered = _records(caplog, "event_loop_recovered")
    assert len(stalled) == 1, [r.getMessage() for r in caplog.records]
    assert "in stalls_the_loop" in stalled[0].stack
    assert "GET /stalls" in stalled[0].in_flight
    assert stalled[0].levelno == logging.ERROR
    assert len(recovered) == 1
    assert recovered[0].duration_ms >= STALL_SECONDS * 1000


async def test_a_free_loop_logs_nothing(caplog):
    caplog.set_level(logging.WARNING, logger="cowork.common.loop_watchdog")
    watchdog = LoopWatchdog(loop_limiters())
    watchdog.start()
    try:
        await asyncio.sleep(loop_watchdog.STALL_AFTER_S + loop_watchdog.PING_EVERY_S)
    finally:
        watchdog.stop()
    assert [r for r in caplog.records if r.name == "cowork.common.loop_watchdog"] == []


async def test_holding_the_default_thread_limiter_full_logs_one_saturation_naming_it(caplog):
    caplog.set_level(logging.WARNING, logger="cowork.common.loop_watchdog")
    default = anyio.to_thread.current_default_thread_limiter()
    default.total_tokens = 2
    release = threading.Event()
    held = [asyncio.create_task(anyio.to_thread.run_sync(release.wait, 30)) for _ in range(2)]
    watchdog = LoopWatchdog(loop_limiters())
    started = time.monotonic()
    watchdog.start()
    try:
        # Held until the report, with room for a slow runner's thread starts
        # and checks: the limiter is free once released, so a late report
        # would never come.
        deadline = started + loop_watchdog.SATURATED_AFTER_S + 3
        while not _records(caplog, "limiter_saturated") and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        # Held past the criterion's 6 s, and for at least three checks after
        # the report: a watchdog that reported a full limiter again would log
        # a second line every LIMITER_CHECK_EVERY_S.
        hold_until = started + loop_watchdog.SATURATED_AFTER_S + 1.5
        await asyncio.sleep(max(3 * loop_watchdog.LIMITER_CHECK_EVERY_S, hold_until - time.monotonic()))
    finally:
        release.set()
        await asyncio.gather(*held)
        watchdog.stop()

    saturated = _records(caplog, "limiter_saturated")
    assert [record.limiter for record in saturated] == ["anyio_default"]
    assert saturated[0].duration_ms >= loop_watchdog.SATURATED_AFTER_S * 1000


async def test_stopping_with_a_ping_waiting_logs_no_stall_and_does_not_hold_the_loop(caplog):
    caplog.set_level(logging.WARNING, logger="cowork.common.loop_watchdog")
    watchdog = LoopWatchdog({})
    watchdog.start()
    await asyncio.sleep(0.2)
    # Busy for less than a stall, so the next ping is still waiting when the
    # lifespan's shutdown stops the watchdog on the loop.
    time.sleep(1.5)
    stopping = time.monotonic()
    watchdog.stop()
    stop_took = time.monotonic() - stopping
    await asyncio.to_thread(watchdog._thread.join, loop_watchdog.STALL_AFTER_S * 2)

    assert _records(caplog, "event_loop_stalled") == []
    assert stop_took < 0.5
    assert not watchdog._thread.is_alive()


def _routed(*, template: str) -> dict:
    """An ASGI scope routing has matched to a route with ``template``."""
    return {"route": SimpleNamespace(path=template)}


def test_the_stall_line_names_the_newest_request_behind_long_streams(monkeypatch):
    now = time.monotonic()
    streams = {
        n: _InFlight(method="POST", started=now - 120 + n, scope=_routed(template="/api/v1/responses/"))
        for n in range(25)
    }
    newest = _InFlight(
        method="GET", started=now - 3, scope=_routed(template="/api/v1/artifacts/workspace/{project_ref}/{artifact_id}/review"),
    )
    monkeypatch.setattr(request_logging, "_in_flight", {**streams, 99: newest})

    summary = request_logging.in_flight_summary().split("; ")

    assert summary[-1].startswith("GET /api/v1/artifacts/workspace/{project_ref}/{artifact_id}/review (3.")
    assert summary[0].startswith("POST /api/v1/responses/ (120.")
    assert summary[10] == "6 more"
    assert len(summary) == 21


async def test_a_request_in_flight_is_named_by_its_route_template_never_its_path():
    """The stall line goes to the logs, so it carries no id or path segment a
    request's own path holds. A request routing has not matched yet has no
    template to give."""
    app = RequestLoggingApp()
    summaries = []

    @app.get("/projects/{project_id}/files/{path:path}")
    async def a_file(project_id: str, path: str):
        summaries.append(request_logging.in_flight_summary())
        return {}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/projects/7f3c1a52-private-id/files/reports/q3-salaries.csv")

    assert response.status_code == 200
    [summary] = summaries
    assert summary.startswith("GET /projects/{project_id}/files/{path:path} (")
    assert "7f3c1a52" not in summary and "salaries" not in summary
    assert _InFlight(method="GET", started=time.monotonic(), scope={}).route == "<unrouted>"


async def test_the_watchdog_watches_every_limiter_a_request_can_queue_on():
    assert set(loop_limiters()) == {"anyio_default", "run_db", "run_efs", "run_efs_unit"}


async def test_holding_the_file_store_slots_full_logs_a_saturation_naming_run_efs(caplog, monkeypatch):
    import cowork.db.session as db_session
    from cowork.db.scoped import LOCAL_SCOPE
    from test_run_efs import holding

    caplog.set_level(logging.WARNING, logger="cowork.common.loop_watchdog")
    monkeypatch.setattr(db_session.settings.storage, "efs_slots", 2)
    watchdog = LoopWatchdog(loop_limiters())
    watchdog.start()
    try:
        async with holding(2, LOCAL_SCOPE):
            deadline = time.monotonic() + loop_watchdog.SATURATED_AFTER_S + 3
            while not _records(caplog, "limiter_saturated") and time.monotonic() < deadline:
                await asyncio.sleep(0.1)
    finally:
        watchdog.stop()

    assert [record.limiter for record in _records(caplog, "limiter_saturated")] == ["run_efs"]
