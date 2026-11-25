"""A scheduled task can leave out connections it doesn't need (ENG-3672)."""
from __future__ import annotations

import importlib
import os
from datetime import datetime, timedelta, timezone

import httpx
import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from cowork.harnesses.anton_harness.harness import _left_out_connections_guidance
from cowork.models.schedule import Schedule
from tests import test_schedule_model_fallback as fallback
from tests.integration.prereq import missing_prerequisite

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


@pytest.mark.parametrize("dialect", ["sqlite", pytest.param("postgresql", marks=pytest.mark.postgres_service)])
def test_downgrade_pauses_only_tasks_with_connector_exclusions(dialect, monkeypatch):
    uri = "sqlite://"
    if dialect == "postgresql":
        uri = os.environ.get("COWORK_TEST_POSTGRES_URI")
        if not uri:
            missing_prerequisite("set COWORK_TEST_POSTGRES_URI to a disposable database",
                                 required_by="COWORK_REQUIRE_SERVICES")
    migration = importlib.import_module("cowork.db.alembic.versions.d4e8a1c6b3f2_schedule_disabled_connections")
    engine = sa.create_engine(uri)
    schedules = sa.Table(
        "schedules", sa.MetaData(),
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("disabled_connections", sa.JSON),
        sa.Column("enabled", sa.Boolean),
        sa.Column("paused_at", sa.DateTime(timezone=True)),
        prefixes=["TEMPORARY"],
    )
    prior_pause = datetime(2026, 11, 25, tzinfo=timezone.utc)
    try:
        with engine.begin() as connection:
            schedules.create(connection)
            connection.execute(schedules.insert(), [
                {"id": "excluded", "disabled_connections": [GMAIL], "enabled": True, "paused_at": None},
                {"id": "paused", "disabled_connections": [GMAIL], "enabled": False, "paused_at": prior_pause},
                {"id": "empty", "disabled_connections": [], "enabled": True, "paused_at": None},
                {"id": "json_null", "disabled_connections": None, "enabled": True, "paused_at": None},
            ])
            connection.execute(schedules.insert().values(id="sql_null", disabled_connections=sa.null(), enabled=True))
            monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(connection)))
            migration.downgrade()

            assert "disabled_connections" not in {c["name"] for c in sa.inspect(connection).get_columns("schedules")}
            rows = {
                row.id: (row.enabled, row.paused_at)
                for row in connection.execute(sa.select(schedules.c.id, schedules.c.enabled, schedules.c.paused_at))
            }
            assert rows["excluded"][0] is False
            assert rows["excluded"][1] is not None
            assert rows["paused"] == (False, prior_pause if dialect == "postgresql" else prior_pause.replace(tzinfo=None))
            assert rows["empty"] == rows["json_null"] == rows["sql_null"] == (True, None)
    finally:
        engine.dispose()
