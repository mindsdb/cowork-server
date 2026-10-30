"""The boot's timings: what an operator compares against the liveness delay.

Liveness counts from container start and restarts a boot that runs past it,
so the boot's length counts from process start, and a slow step names itself
while it runs and as it ends rather than only in the line a killed boot never
reaches.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time

import psutil
import pytest
from fastapi import FastAPI

from cowork import server
from cowork.common.structured_logging import make_formatter


def _events(caplog, name: str) -> list[logging.LogRecord]:
    """The captured records whose structured ``event`` is ``name``."""
    return [record for record in caplog.records if getattr(record, "event", None) == name]


def test_a_step_over_ten_seconds_names_itself_as_it_ends(caplog):
    caplog.set_level(logging.WARNING)
    clock = server._StartupClock()
    clock.begin("dev_setup")
    clock.begin("published_page_backfill")
    # The step started 11 s ago.
    clock._mark -= 11
    clock.finished()

    # One line: the fast step logs nothing.
    [slow] = _events(caplog, "startup_step_slow")
    assert slow.levelno == logging.WARNING
    assert slow.getMessage() == "Startup step slow"
    assert slow.step == "published_page_backfill"
    assert slow.duration_ms >= 11_000
    line = json.loads(make_formatter().format(slow))
    assert (line["step"], line["duration_ms"]) == ("published_page_backfill", slow.duration_ms)


def _wait_for_event(caplog, name: str, *, timeout: float = 2.0) -> list[logging.LogRecord]:
    deadline = time.monotonic() + timeout
    while not (found := _events(caplog, name)) and time.monotonic() < deadline:
        time.sleep(0.01)
    return found


def test_a_step_still_running_past_the_threshold_names_itself_before_it_ends(caplog, monkeypatch):
    """The line a boot that liveness kills mid-step still prints: the step
    that was running, logged from a timer thread while the step goes on."""
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(server, "_SLOW_STEP_S", 0.05)
    clock = server._StartupClock()
    clock.begin("published_page_backfill")

    running = _wait_for_event(caplog, "startup_step_running")
    assert _events(caplog, "startup_step_slow") == [], "the step had not ended"
    clock.finished()

    [record] = running
    assert record.levelno == logging.WARNING
    assert record.getMessage() == "Startup step running"
    assert record.step == "published_page_backfill"
    assert record.duration_ms >= 50
    assert record.thread != threading.get_ident()
    line = json.loads(make_formatter().format(record))
    assert (line["step"], line["duration_ms"]) == ("published_page_backfill", record.duration_ms)
    # The step that ended past the threshold still logs as it ends.
    [slow] = _events(caplog, "startup_step_slow")
    assert slow.step == "published_page_backfill"


def test_a_fast_step_logs_nothing_and_its_timer_is_cancelled(caplog, monkeypatch):
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(server, "_SLOW_STEP_S", 0.2)
    clock = server._StartupClock()
    clock.begin("dev_setup")
    first = clock._timer
    clock.begin("credential_clear")
    second = clock._timer
    clock.finished()

    for timer in (first, second):
        assert timer.daemon
        # A cancelled timer's thread ends at once; one left armed would run
        # until the threshold and then log.
        timer.join(timeout=1)
        assert not timer.is_alive()
    assert _events(caplog, "startup_step_running") == []
    assert _events(caplog, "startup_step_slow") == []
    [finished] = _events(caplog, "startup_finished")
    assert [step.split("=")[0] for step in finished.steps.split(", ")] == ["dev_setup", "credential_clear"]


async def test_a_step_that_raises_is_not_logged_as_still_running(caplog, monkeypatch):
    """A step that raises never reaches the next step, so the clock never
    ends it. The end of the task that ran the lifespan stops its timer."""
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(server, "_SLOW_STEP_S", 0.2)

    def fail() -> None:
        raise RuntimeError("dev setup failed")

    monkeypatch.setattr(server, "run_dev_setup", fail)

    async def boot() -> None:
        async with server.lifespan(FastAPI()):
            pass

    with pytest.raises(RuntimeError, match="dev setup failed"):
        await asyncio.create_task(boot())
    # Past the threshold, when a timer left armed would have logged.
    await asyncio.sleep(0.5)

    assert _events(caplog, "startup_step_running") == []


def test_the_boot_length_counts_from_process_start(caplog):
    caplog.set_level(logging.WARNING)
    process_age = time.time() - psutil.Process().create_time()

    server._StartupClock().finished()

    [finished] = _events(caplog, "startup_finished")
    assert finished.duration_ms >= int(process_age * 1000)
