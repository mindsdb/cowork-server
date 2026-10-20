"""Stopping the scheduler for a shutdown.

With remote turns the server stops the poll loop before it ends the turns in
flight, so no run starts behind that, and then waits for the runs the loop
already started, so each one records its turn's outcome and finishes its run
row before the process exits.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from cowork import scheduler


@pytest.fixture
async def one_due_schedule(monkeypatch):
    """A poll loop that finds one schedule due on its first poll and none
    after, and a run for it that holds until ``release`` is set."""
    schedule_id = uuid4()
    polls: list[int] = []
    release = asyncio.Event()
    finished: list[object] = []

    async def poll(fn, *, scope):
        polls.append(len(polls))
        return [schedule_id] if len(polls) == 1 else []

    async def execute_schedule(due_id, is_manual=False, conversation_id=None):
        await release.wait()
        finished.append(due_id)

    monkeypatch.setattr(scheduler, "_POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(scheduler, "run_db", poll)
    monkeypatch.setattr(scheduler, "execute_schedule", execute_schedule)
    yield polls, release, finished, schedule_id
    release.set()
    task = scheduler._scheduler_task
    if task is not None:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    scheduler._scheduler_task = None


async def _until(condition) -> None:
    async with asyncio.timeout(2):
        while not condition():
            await asyncio.sleep(0)


async def test_stopping_twice_is_safe_and_the_runs_it_started_are_awaited(one_due_schedule):
    polls, release, finished, schedule_id = one_due_schedule
    scheduler.start_scheduler()
    loop_task = scheduler._scheduler_task
    await _until(lambda: scheduler._scheduled_runs)

    await scheduler.stop_scheduler(timeout_seconds=1)
    await scheduler.stop_scheduler(timeout_seconds=1)

    assert loop_task.done() and scheduler._scheduler_task is None
    polled = len(polls)
    await asyncio.sleep(0.05)
    assert len(polls) == polled, "the loop polled after it was stopped"

    # The run holds, so a short wait reports it still going.
    assert await scheduler.wait_for_scheduled_runs(timeout_seconds=0.05) == 1
    release.set()
    assert await scheduler.wait_for_scheduled_runs(timeout_seconds=1) == 0
    assert finished == [schedule_id]
    assert not scheduler._scheduled_runs


async def test_stopping_a_scheduler_that_never_started_does_nothing():
    assert scheduler._scheduler_task is None
    await scheduler.stop_scheduler(timeout_seconds=1)
    assert await scheduler.wait_for_scheduled_runs(timeout_seconds=1) == 0
