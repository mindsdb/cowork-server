"""The scheduled-task revisions, upgraded over existing data and taken back,
on SQLite and, when COWORK_TEST_POSTGRES_URI points at a disposable database,
on Postgres."""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest
import sqlalchemy as sa
from alembic import command
from sqlalchemy import create_engine, text
from sqlmodel import SQLModel

import cowork.models.schedule  # noqa: F401
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.migrations import _alembic_config
from cowork.models.schedule import ScheduleOccurrence, ScheduleRun

POSTGRES_URI = os.environ.get("COWORK_TEST_POSTGRES_URI", "")
BEFORE_OCCURRENCES = "3e4b5f7586d3"
OCCURRENCES = "bb263d8b3b42"


@pytest.fixture(params=["sqlite", "postgres"])
def database(request, tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    if request.param == "postgres":
        if not POSTGRES_URI:
            pytest.skip("set COWORK_TEST_POSTGRES_URI to a disposable database")
        uri = POSTGRES_URI
        engine = create_engine(uri)
        _wipe(engine)
    else:
        uri = f"sqlite:///{tmp_path / 'migrate.db'}"
        engine = create_engine(uri)
    yield engine, uri
    if request.param == "postgres":
        _wipe(engine)
    engine.dispose()


def _wipe(engine) -> None:
    SQLModel.metadata.drop_all(engine)
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS alembic_version"))


def _migrate(engine, uri: str, revision: str, *, down: bool = False) -> None:
    config = _alembic_config(uri)
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        (command.downgrade if down else command.upgrade)(config, revision)


def _columns(engine, table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(engine).get_columns(table)}


_meta = sa.MetaData()
# Only the columns these tests write and read, typed as the models type them,
# so the rows bind the same way on both engines whatever revision is current.
_projects = sa.Table("projects", _meta, sa.Column("id", sa.Uuid), sa.Column("name", sa.String),
                     sa.Column("path", sa.String))
_schedules = sa.Table(
    "schedules", _meta, sa.Column("id", sa.Uuid), sa.Column("title", sa.String),
    sa.Column("prompt", sa.String), sa.Column("cadence", sa.String),
    sa.Column("next_run_at", sa.DateTime(timezone=True)), sa.Column("enabled", sa.Boolean),
    sa.Column("model", sa.String), sa.Column("project_id", sa.Uuid), sa.Column("timezone", sa.String),
    sa.Column("missed_runs", sa.Integer),
)
_runs = sa.Table(
    "schedule_runs", _meta, sa.Column("id", sa.Uuid), sa.Column("schedule_id", sa.Uuid),
    sa.Column("started_at", sa.DateTime(timezone=True)), sa.Column("finished_at", sa.DateTime(timezone=True)),
    sa.Column("status", sa.String), sa.Column("is_manual", sa.Boolean),
)


def _seed_schedule_and_run(engine) -> tuple[uuid.UUID, uuid.UUID]:
    schedule_id, run_id, project_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    now = datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc)
    with engine.begin() as connection:
        connection.execute(_projects.insert().values(id=project_id, name="seed", path="/seed"))
        connection.execute(_schedules.insert().values(
            id=schedule_id, title="Daily report", prompt="Summarize", cadence="daily",
            next_run_at=now, enabled=True, model="default", project_id=project_id,
            timezone="UTC", missed_runs=0,
        ))
        connection.execute(_runs.insert().values(
            id=run_id, schedule_id=schedule_id, started_at=now, finished_at=now,
            status="success", is_manual=False,
        ))
    return schedule_id, run_id


def _run_row(engine, run_id: uuid.UUID) -> dict:
    with engine.connect() as connection:
        reflected = sa.Table("schedule_runs", sa.MetaData(), autoload_with=connection)
        rows = connection.execute(sa.select(reflected)).mappings().all()
    [row] = [r for r in rows if str(r["id"]).replace("-", "") == run_id.hex]
    return dict(row)


def test_occurrences_upgrade_keeps_existing_runs_and_matches_the_models(database):
    engine, uri = database
    _migrate(engine, uri, BEFORE_OCCURRENCES)
    schedule_id, run_id = _seed_schedule_and_run(engine)

    _migrate(engine, uri, OCCURRENCES)

    assert _columns(engine, "schedule_occurrences") == set(ScheduleOccurrence.__table__.columns.keys())
    assert set(ScheduleRun.__table__.columns.keys()) <= _columns(engine, "schedule_runs")
    row = _run_row(engine, run_id)
    assert row["status"] == "success"
    assert (row["occurrence_id"], row["attempt"], row["outcome"], row["execution_started_at"]) == (
        None, None, None, None,
    )

    _migrate(engine, uri, BEFORE_OCCURRENCES, down=True)

    assert "schedule_occurrences" not in sa.inspect(engine).get_table_names()
    assert {"occurrence_id", "outcome", "execution_started_at"}.isdisjoint(_columns(engine, "schedule_runs"))
