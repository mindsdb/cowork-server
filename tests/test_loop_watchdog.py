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

import anyio
import httpx

from cowork.common import loop_watchdog
from cowork.common.loop_watchdog import LoopWatchdog
from cowork.common.request_logging import RequestLoggingApp
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
    watchdog.start()
    try:
        await asyncio.sleep(loop_watchdog.SATURATED_AFTER_S + 1)
    finally:
        release.set()
        await asyncio.gather(*held)
        watchdog.stop()

    saturated = _records(caplog, "limiter_saturated")
    assert [record.limiter for record in saturated] == ["anyio_default"]
    assert saturated[0].duration_ms >= loop_watchdog.SATURATED_AFTER_S * 1000
