"""Editing a scheduled task's project from the page: a new project sticks,
and "No project" clears it back to the default instead of being dropped."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import test_turn_pool_connections as turn_pool

from cowork.db.scoped import SYSTEM_SCOPE
from cowork.db.units import unit_session
from cowork.services.projects import GENERAL_PROJECT_ID, ProjectService

_turn_runs_in_this_process = turn_pool._turn_runs_in_this_process


async def test_an_edit_moves_a_task_to_another_project_and_back_to_none():
    with unit_session(scope=SYSTEM_SCOPE) as session:
        project_id = ProjectService(session).create_project("Reports").id
    due = datetime.now(timezone.utc) + timedelta(days=1)
    try:
        with turn_pool._a_daily_schedule(due=due) as schedule_id:
            url = f"/api/v1/schedules/{schedule_id}"
            async with turn_pool._client() as client:
                moved = await client.patch(url, json={"projectId": str(project_id)})
                assert moved.status_code == 200, moved.text
                assert moved.json()["projectId"] == str(project_id)

                untouched = await client.patch(url, json={"title": "Renamed"})
                assert untouched.json()["projectId"] == str(project_id)

                cleared = await client.patch(url, json={"projectId": None})
                assert cleared.status_code == 200, cleared.text
                assert cleared.json()["projectId"] == str(GENERAL_PROJECT_ID)
                assert (await client.get(url)).json()["projectId"] == str(GENERAL_PROJECT_ID)
    finally:
        with unit_session(scope=SYSTEM_SCOPE) as session:
            ProjectService(session).delete_project(project_id)
