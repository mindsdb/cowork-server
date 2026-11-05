"""A run started from POST /schedules/{id}/run-now holds no pooled connection
while it answers.

The run goes through the real ResponsesHandler and AntonHarness, set up as in
test_turn_pool_connections, with only anton's ChatSession replaced.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import test_turn_pool_connections as turn_pool
from _fakes import PausedModel

from cowork.harnesses.anton_harness import harness as harness_mod
from cowork.schemas.schedules import RunStatus

# The same in-process turn setup the pool tests use, autouse here too.
_turn_runs_in_this_process = turn_pool._turn_runs_in_this_process


async def test_a_run_started_from_run_now_holds_no_connection_while_it_answers(monkeypatch):
    """The route answers 202 and runs the schedule as a background task, and
    FastAPI closes the request's session only once that task ends. So the
    route gives its connection back before the run starts: none is checked
    out while the run's model answers, and the run completes."""
    pool = turn_pool._app_engine().pool
    model = PausedModel()
    monkeypatch.setattr(harness_mod, "build_chat_session", model.build)

    with turn_pool._a_daily_schedule(due=datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc)) as schedule_id:
        async with turn_pool._client() as client:
            posted = asyncio.create_task(client.post(f"/api/v1/schedules/{schedule_id}/run-now"))
            await asyncio.wait_for(model.answering.wait(), timeout=10)
            held_while_answering = pool.checkedout()
            model.release.set()
            answered = await asyncio.wait_for(posted, timeout=10)
        finished = turn_pool._finished_run(schedule_id)

    assert answered.status_code == 202, answered.text
    assert held_while_answering == 0
    assert finished.status == RunStatus.success
    assert str(finished.conversation_id) == answered.json()["conversation_id"]
