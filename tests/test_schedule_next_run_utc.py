"""A start time sent with a UTC offset keeps its instant. SQLite drops the
offset when it stores a datetime, so the request schemas convert to UTC
first; otherwise 09:00-07:00 would run at 09:00 UTC."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import test_turn_pool_connections as turn_pool

from cowork.db.scoped import SYSTEM_SCOPE
from cowork.db.units import unit_session
from cowork.services.schedules import ScheduleService

_turn_runs_in_this_process = turn_pool._turn_runs_in_this_process


async def test_an_offset_start_is_stored_as_the_same_instant():
    created_id = None
    try:
        async with turn_pool._client() as client:
            created = await client.post("/api/v1/schedules/", json={
                "title": "Offset", "prompt": "x", "cadence": "daily",
                "timezone": "America/Los_Angeles", "nextRunAt": "2030-01-08T09:00:00-08:00",
            })
            assert created.status_code in (200, 201), created.text
            created_id = created.json()["id"]
            read = (await client.get(f"/api/v1/schedules/{created_id}")).json()
            assert datetime.fromisoformat(read["nextRunAt"]) == datetime(2030, 1, 8, 17, 0, tzinfo=timezone.utc)

            edited = await client.patch(f"/api/v1/schedules/{created_id}", json={"nextRunAt": "2030-01-09T09:00:00+05:30"})
            assert datetime.fromisoformat(edited.json()["nextRunAt"]) == datetime(2030, 1, 9, 3, 30, tzinfo=timezone.utc)
    finally:
        if created_id:
            with unit_session(scope=SYSTEM_SCOPE) as session:
                ScheduleService(session).delete_schedule(UUID(created_id))
