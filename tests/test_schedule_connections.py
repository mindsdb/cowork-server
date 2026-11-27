"""A scheduled task can leave out connections it doesn't need (ENG-3672)."""
from __future__ import annotations

import importlib
import json
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from fastapi import HTTPException, Request

from cowork.harnesses.anton_harness.harness import _left_out_connections_guidance
from cowork.api.v1.endpoints import schedules as schedules_api
from cowork.db.scoped import TenantScope
from cowork.harnesses.anton_harness.schedule_connector_reporting import build_blocked_connector_tool
from cowork.harnesses.anton_harness.schedule_tools import _list_unit
from cowork.models.schedule import Schedule
from cowork.schemas.connectors import DisabledConnection
from cowork.turnqueue.producer import _datasource_connection_refs
from tests import test_schedule_model_fallback as fallback
from tests.integration.prereq import missing_prerequisite

# The model-fallback tests' fixtures: a daily task due now, and a turn handler
# that records each request.
_fresh_scheduler = fallback._fresh_scheduler
schedule_id = fallback.schedule_id
turns = fallback.turns
COMPLETED, _run_due_slot, _scoped = fallback.COMPLETED, fallback._run_due_slot, fallback._scoped

GMAIL = {"engine": "gmail", "name": "Work mail"}


@pytest.fixture(autouse=True)
def visible_local_connections(monkeypatch):
    monkeypatch.setattr(schedules_api, "turn_vault_for_scope", lambda _: SimpleNamespace(
        list_connections=lambda: [GMAIL],
    ))


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


async def test_only_a_connector_excluded_from_this_run_can_be_reported():
    tool = build_blocked_connector_tool([GMAIL])
    accepted = await tool.handler(None, GMAIL)
    assert json.loads(accepted) == {"blocked_connector": "Work mail (gmail)"}
    assert await tool.handler(None, {"engine": "gmail", "name": "Personal mail"}) == (
        "Choose a connector that this task excludes."
    )


def test_chat_lists_the_task_exclusions(schedule_id):
    _set(schedule_id, [GMAIL])
    session, scoped = _scoped()
    try:
        (summary,) = [item for item in _list_unit(scoped) if item["schedule_id"] == str(schedule_id)]
        assert summary["disabled_connections"] == [GMAIL]
    finally:
        session.close()


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
        unknown = await client.patch(url, json={"disabledConnections": [{"engine": "gmail", "name": "Typo"}]})
        assert unknown.status_code == 422
        cleared = (await client.patch(url, json={"disabledConnections": []})).json()
        assert not cleared["disabledConnections"]


async def test_a_task_stays_editable_after_a_connection_it_leaves_out_is_removed(schedule_id, monkeypatch):
    url = f"/api/v1/schedules/{schedule_id}"
    async with _client() as client:
        saved = (await client.patch(url, json={"disabledConnections": [GMAIL]})).json()
        monkeypatch.setattr(schedules_api, "turn_vault_for_scope", lambda _: SimpleNamespace(
            list_connections=lambda: [],
        ))
        # The editor resends the stored list with every save.
        edit = await client.patch(url, json={"title": "Renamed", "disabledConnections": saved["disabledConnections"]})
        assert edit.status_code == 200
        # Still left out, should the connection come back.
        assert edit.json()["disabledConnections"] == [GMAIL]
        other = {"engine": "gmail", "name": "Never saved"}
        added = await client.patch(url, json={"disabledConnections": [GMAIL, other]})
        assert added.status_code == 422


async def test_a_stored_hosted_exclusion_survives_its_datasource_being_deleted(monkeypatch):
    async def none(*_):
        return []

    async def catalogue(*_):
        return {"items": []}

    monkeypatch.setattr(schedules_api.auth_proxy, "proxy_catalogue", catalogue)
    monkeypatch.setattr(schedules_api.auth_proxy, "proxy_datasource_list", none)
    monkeypatch.setattr(schedules_api.auth_proxy, "proxy_custom_connection_list", none)
    stored = {"engine": "postgres", "name": "Primary", "datasource_id": 12}
    scope, request = TenantScope(org_mode=True), Request({"type": "http"})
    assert await schedules_api._resolve_exclusions(
        [DisabledConnection(**stored)], scope, request, stored=[stored],
    ) == [stored]
    with pytest.raises(HTTPException, match="could not be found"):
        await schedules_api._resolve_exclusions(
            [DisabledConnection(engine="postgres", name="Primary", datasource_id=13)], scope, request, stored=[stored],
        )


async def test_a_hosted_datasource_exclusion_binds_the_visible_id(monkeypatch):
    datasource = {"id": 12, "connector_id": "postgres", "name": "Primary"}

    async def catalogue(*_):
        return {"items": []}

    async def datasources(*_):
        return [datasource]

    async def custom(*_):
        return []

    monkeypatch.setattr(schedules_api.auth_proxy, "proxy_catalogue", catalogue)
    monkeypatch.setattr(schedules_api.auth_proxy, "proxy_datasource_list", datasources)
    monkeypatch.setattr(schedules_api.auth_proxy, "proxy_custom_connection_list", custom)
    scope = TenantScope(org_mode=True)
    request = Request({"type": "http"})
    assert await schedules_api._resolve_exclusions(
        [DisabledConnection(engine="postgres", name="Primary")], scope, request,
    ) == [{"engine": "postgres", "name": "Primary", "datasource_id": 12}]
    datasource["name"] = "Renamed"
    assert await schedules_api._resolve_exclusions(
        [DisabledConnection(engine="postgres", name="Primary", datasource_id=12)], scope, request,
    ) == [{"engine": "postgres", "name": "Renamed", "datasource_id": 12}]
    with pytest.raises(HTTPException, match="could not be found"):
        await schedules_api._resolve_exclusions(
            [DisabledConnection(engine="postgres", name="Renamed", datasource_id=13)], scope, request,
        )


def test_hosted_grants_keep_the_renamed_datasource_excluded_and_allow_its_peer():
    connections = [
        {"id": 12, "connector_id": "postgres", "name": "Renamed", "status": "verified", "credential_version": 1},
        {"id": 13, "connector_id": "postgres", "name": "Other", "status": "verified", "credential_version": 1},
    ]
    assert _datasource_connection_refs(
        connections, [{"engine": "postgres", "name": "Primary", "datasource_id": 12}], scheduled=True,
    ) == [{"connection_id": 13, "credential_version": 1}]
    assert _datasource_connection_refs(connections, [{"engine": "postgres", "name": "Primary"}], scheduled=True) == []


def test_a_datasource_deleted_and_added_again_under_its_name_stays_left_out():
    """The task leaves out Primary, id 12. Its owner deletes it and adds a
    new Primary, which auth lists as id 13. The run still gets no grant for
    it, and the engine's other datasources stay usable."""
    connections = [
        {"id": 13, "connector_id": "postgres", "name": "Primary", "status": "verified", "credential_version": 1},
        {"id": 14, "connector_id": "postgres", "name": "Other", "status": "verified", "credential_version": 1},
    ]
    assert _datasource_connection_refs(
        connections, [{"engine": "postgres", "name": "Primary", "datasource_id": 12}], scheduled=True,
    ) == [{"connection_id": 14, "credential_version": 1}]


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
