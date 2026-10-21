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

    # version_id arrives with the audit-trail revision.
    assert _columns(engine, "schedule_occurrences") == set(ScheduleOccurrence.__table__.columns.keys()) - {
        "version_id",
    }
    assert set(ScheduleRun.__table__.columns.keys()) <= _columns(engine, "schedule_runs")
    row = _run_row(engine, run_id)
    assert row["status"] == "success"
    assert (row["occurrence_id"], row["attempt"], row["outcome"], row["execution_started_at"]) == (
        None, None, None, None,
    )

    _migrate(engine, uri, BEFORE_OCCURRENCES, down=True)

    assert "schedule_occurrences" not in sa.inspect(engine).get_table_names()
    assert {"occurrence_id", "outcome", "execution_started_at"}.isdisjoint(_columns(engine, "schedule_runs"))


PAUSE_AND_MISSED = "15812ea140b3"


def _add_schedule(connection, project_id, *, cadence: str, enabled: bool, missed_runs: int = 0) -> uuid.UUID:
    schedule_id = uuid.uuid4()
    connection.execute(_schedules.insert().values(
        id=schedule_id, title=cadence, prompt="Summarize", cadence=cadence,
        next_run_at=datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc), enabled=enabled, model="default",
        project_id=project_id, timezone="UTC", missed_runs=missed_runs,
    ))
    return schedule_id


def test_pause_upgrade_sorts_existing_disabled_schedules_into_their_states(database):
    from cowork.models.schedule import Schedule, ScheduleSkippedPeriod

    engine, uri = database
    _migrate(engine, uri, OCCURRENCES)
    project_id = uuid.uuid4()
    with engine.begin() as connection:
        connection.execute(_projects.insert().values(id=project_id, name="seed", path="/seed"))
        active = _add_schedule(connection, project_id, cadence="daily", enabled=True)
        paused = _add_schedule(connection, project_id, cadence="daily", enabled=False)
        ran = _add_schedule(connection, project_id, cadence="once", enabled=False)
        missed = _add_schedule(connection, project_id, cadence="once", enabled=False, missed_runs=1)
        paused_once = _add_schedule(connection, project_id, cadence="once", enabled=False)
        now = datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc)
        connection.execute(_runs.insert().values(
            id=uuid.uuid4(), schedule_id=ran, started_at=now, finished_at=now, status="failed", is_manual=False,
        ))

    _migrate(engine, uri, PAUSE_AND_MISSED)

    assert {"paused_at", "missed_at", "auto_paused_at", "consecutive_failures"} <= _columns(engine, "schedules")
    assert _columns(engine, "schedule_skipped_periods") == set(ScheduleSkippedPeriod.__table__.columns.keys())
    assert set(Schedule.__table__.columns.keys()) <= _columns(engine, "schedules")
    with engine.connect() as connection:
        reflected = sa.Table("schedules", sa.MetaData(), autoload_with=connection)
        rows = {
            str(r["id"]).replace("-", ""): r
            for r in connection.execute(sa.select(reflected)).mappings().all()
        }

    def state(schedule_id):
        row = rows[schedule_id.hex]
        return (row["paused_at"] is not None, row["missed_at"] is not None)

    assert state(active) == (False, False)
    assert state(paused) == (True, False)
    assert state(ran) == (False, False)
    assert state(missed) == (False, True)
    assert state(paused_once) == (True, False)

    _migrate(engine, uri, OCCURRENCES, down=True)

    assert "schedule_skipped_periods" not in sa.inspect(engine).get_table_names()
    assert {"paused_at", "missed_at"}.isdisjoint(_columns(engine, "schedules"))


AUDIT_TRAIL = "4552aea8bb5f"


def test_audit_upgrade_gives_every_schedule_its_first_version_and_keeps_history(database):
    from cowork.models.schedule import ScheduleEvent, ScheduleVersion

    engine, uri = database
    _migrate(engine, uri, PAUSE_AND_MISSED)
    schedule_id, run_id = _seed_schedule_and_run(engine)

    _migrate(engine, uri, AUDIT_TRAIL)

    assert _columns(engine, "schedule_versions") == set(ScheduleVersion.__table__.columns.keys())
    assert _columns(engine, "schedule_events") == set(ScheduleEvent.__table__.columns.keys())
    assert "version_id" in _columns(engine, "schedule_occurrences")
    assert "ix_schedules_enabled_next_run_at" in {ix["name"] for ix in sa.inspect(engine).get_indexes("schedules")}
    with engine.connect() as connection:
        versions = sa.Table("schedule_versions", sa.MetaData(), autoload_with=connection)
        rows = connection.execute(sa.select(versions)).mappings().all()
    [version] = [r for r in rows if str(r["schedule_id"]).replace("-", "") == schedule_id.hex]
    assert (version["number"], version["prompt"], version["model"], version["cadence"]) == (
        1, "Summarize", "default", "daily",
    )
    assert _run_row(engine, run_id)["status"] == "success"

    # Running it again on an up-to-date database adds no second version.
    _migrate(engine, uri, PAUSE_AND_MISSED, down=True)
    _migrate(engine, uri, AUDIT_TRAIL)
    with engine.connect() as connection:
        versions = sa.Table("schedule_versions", sa.MetaData(), autoload_with=connection)
        assert connection.execute(sa.select(sa.func.count()).select_from(versions)).scalar() == 1

    _migrate(engine, uri, PAUSE_AND_MISSED, down=True)
    assert {"schedule_versions", "schedule_events"}.isdisjoint(sa.inspect(engine).get_table_names())
