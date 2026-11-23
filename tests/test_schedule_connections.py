"""A scheduled task can leave out connections it doesn't need (ENG-3672)."""
from __future__ import annotations

import httpx

from cowork.harnesses.anton_harness.harness import _left_out_connections_guidance
from cowork.models.schedule import Schedule
from tests import test_schedule_model_fallback as fallback

# The model-fallback tests' fixtures: a daily task due now, and a turn handler
# that records each request.
_fresh_scheduler = fallback._fresh_scheduler
schedule_id = fallback.schedule_id
turns = fallback.turns
COMPLETED, _run_due_slot, _scoped = fallback.COMPLETED, fallback._run_due_slot, fallback._scoped

GMAIL = {"engine": "gmail", "name": "Work mail"}


def _client() -> httpx.AsyncClient:
    from cowork.server import create_app

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test")


def _set(schedule_id, disabled):
    session, _ = _scoped()
    try:
        schedule = session.get(Schedule, schedule_id)
        schedule.disabled_connections = disabled
        session.add(schedule)
        session.commit()
    finally:
        session.close()


async def test_a_run_leaves_out_the_connections_its_task_does(schedule_id, turns):
    _set(schedule_id, [GMAIL])
    turns.endings.append(COMPLETED)

    await _run_due_slot(schedule_id)

    (request,) = turns.requests
    assert [c.model_dump() for c in request.disabled_connections] == [GMAIL]


async def test_a_task_with_none_left_out_runs_with_every_connection(schedule_id, turns):
    turns.endings.append(COMPLETED)

    await _run_due_slot(schedule_id)

    (request,) = turns.requests
    assert not request.disabled_connections


async def test_the_api_saves_keeps_and_clears_a_tasks_left_out_connections(schedule_id):
    url = f"/api/v1/schedules/{schedule_id}"
    async with _client() as client:
        saved = (await client.patch(url, json={"disabledConnections": [GMAIL]})).json()
        assert saved["disabledConnections"] == [GMAIL]
        # An edit that doesn't name them keeps them.
        kept = (await client.patch(url, json={"title": "Renamed"})).json()
        assert kept["disabledConnections"] == [GMAIL]
        cleared = (await client.patch(url, json={"disabledConnections": []})).json()
        assert not cleared["disabledConnections"]


def test_a_scheduled_run_is_told_which_connections_its_task_leaves_out():
    guidance = _left_out_connections_guidance([GMAIL, {"engine": "hubspot", "name": "CRM"}])
    assert "Work mail (gmail), CRM (hubspot)" in guidance
    assert "say which connection it needs" in guidance


async def test_a_task_can_leave_out_connections_from_the_start():
    from datetime import datetime, timedelta, timezone

    body = {
        "title": "Weekly digest", "prompt": "Summarize", "cadence": "weekly",
        "nextRunAt": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
        "disabledConnections": [GMAIL],
    }
    async with _client() as client:
        created = await client.post("/api/v1/schedules/", json=body)
        assert created.status_code in (200, 201), created.text
        assert created.json()["disabledConnections"] == [GMAIL]
        await client.delete(f"/api/v1/schedules/{created.json()['id']}")
