"""Desktop-upgrade proof for the tenancy columns migration (a3f9c2e8b1d4).

Builds a real SQLite database at the PREVIOUS migration head, seeds rows in
every touched table (simulating an existing desktop install), then exercises
upgrade / downgrade / re-upgrade. A PostgreSQL variant runs when
COWORK_TEST_POSTGRES_URI points at a disposable database (no Postgres in CI
yet — the variant is the contract for when it lands).
"""
from __future__ import annotations

import os
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.models.conversation import Conversation
from cowork.models.pin import Pin
from cowork.models.project import Project
from cowork.models.schedule import Schedule
from cowork.models.setting import Setting

PREVIOUS_HEAD = "f7d2b9e4a1c6"

# (table, column) pairs the migration must add — keep in sync with _COLUMNS.
EXPECTED = [
    ("projects", "org_id"), ("projects", "created_by"),
    ("conversations", "org_id"), ("conversations", "created_by"),
    ("messages", "created_by"),
    ("files", "org_id"), ("files", "created_by"),
    ("schedules", "org_id"), ("schedules", "created_by"),
    ("channel_installations", "org_id"),
    ("channel_bindings", "org_id"), ("channel_bindings", "created_by"),
    ("pins", "user_id"), ("pins", "org_id"),
    ("settings", "scope"), ("settings", "user_id"), ("settings", "org_id"),
]

INDEXED = [
    ("projects", "ix_projects_org_id"),
    ("conversations", "ix_conversations_org_id"),
    ("files", "ix_files_org_id"),
    ("schedules", "ix_schedules_org_id"),
    ("channel_installations", "ix_channel_installations_org_id"),
    ("channel_bindings", "ix_channel_bindings_org_id"),
    ("pins", "ix_pins_user_id_org_id"),
]


def _seed(engine) -> dict[str, str]:
    """Insert one desktop-style row per touched table; return seeded ids."""
    project_id = uuid4().hex
    conversation_id = uuid4().hex
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO projects (id, name, path, is_active) "
            f"VALUES ('{project_id}', 'seeded', '/tmp/seeded', 1)"
        ))
        conn.execute(text(
            "INSERT INTO conversations (id, topic, project_id) "
            f"VALUES ('{conversation_id}', 'old chat', '{project_id}')"
        ))
        conn.execute(text(
            "INSERT INTO messages (id, conversation_id, role, content) "
            f"VALUES ('{uuid4().hex}', '{conversation_id}', 'user', '{{}}')"
        ))
        conn.execute(text(
            "INSERT INTO files (id, filename, content_type, size, purpose, path) "
            f"VALUES ('{uuid4().hex}', 'a.txt', 'text/plain', 1, 'attachment:x', '/tmp/a.txt')"
        ))
        conn.execute(text(
            "INSERT INTO schedules (id, title, prompt, cadence, timezone, next_run_at, "
            "enabled, project_id, model, missed_runs) "
            f"VALUES ('{uuid4().hex}', 'daily report', 'do it', 'daily', 'UTC', "
            f"'2026-01-01 00:00:00', 1, '{project_id}', 'sonnet', 0)"
        ))
        conn.execute(text(
            "INSERT INTO pins (id, item_type, item_id) "
            f"VALUES ('{uuid4().hex}', 'conversation', 'c-1')"
        ))
        conn.execute(text(
            "INSERT INTO settings (id, key, value) "
            f"VALUES ('{uuid4().hex}', 'seeded_key', 'seeded_value')"
        ))
        conn.execute(text(
            "INSERT INTO channel_installations (id, channel_type, display_name, enabled, status) "
            f"VALUES ('{uuid4().hex}', 'telegram', 'Telegram', 0, 'disconnected')"
        ))
        conn.execute(text(
            "INSERT INTO channel_bindings (id, channel_type, external_group_id, "
            "external_thread_key, trigger_rule) "
            f"VALUES ('{uuid4().hex}', 'telegram', 'g-1', '__default__', 'always')"
        ))
    return {"project_id": project_id, "conversation_id": conversation_id}


@pytest.fixture()
def desktop_db(tmp_path, monkeypatch):
    """A SQLite file at the previous head, seeded like a desktop install."""
    db_path = tmp_path / "cowork.db"
    monkeypatch.setenv("DATABASE_URI", f"sqlite:///{db_path}")
    get_app_settings.cache_clear()

    cfg = Config("alembic.ini")
    command.upgrade(cfg, PREVIOUS_HEAD)
    _seed(create_engine(f"sqlite:///{db_path}"))

    yield cfg, db_path
    get_app_settings.cache_clear()


def test_upgrade_adds_nullable_columns_and_keeps_rows(desktop_db):
    cfg, db_path = desktop_db
    command.upgrade(cfg, "head")

    inspector = inspect(create_engine(f"sqlite:///{db_path}"))
    for table, column in EXPECTED:
        cols = {c["name"]: c for c in inspector.get_columns(table)}
        assert column in cols, f"{table}.{column} missing after upgrade"
        assert cols[column]["nullable"], f"{table}.{column} must be nullable"

    # old rows survive with NULL ownership and untouched values
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        for table, column in EXPECTED:
            rows = conn.execute(text(f"SELECT {column} FROM {table}")).fetchall()
            assert rows, f"seed row lost from {table}"
            assert all(value is None for (value,) in rows)
        name, path = conn.execute(
            text("SELECT name, path FROM projects WHERE path = '/tmp/seeded'")
        ).one()
        assert (name, path) == ("seeded", "/tmp/seeded")
        assert conn.execute(
            text("SELECT value FROM settings WHERE key = 'seeded_key'")
        ).scalar_one() == "seeded_value"


def test_indexes_created(desktop_db):
    cfg, db_path = desktop_db
    command.upgrade(cfg, "head")
    inspector = inspect(create_engine(f"sqlite:///{db_path}"))
    for table, index in INDEXED:
        assert index in {i["name"] for i in inspector.get_indexes(table)}, index


def test_upgraded_db_works_with_current_models(desktop_db):
    cfg, db_path = desktop_db
    command.upgrade(cfg, "head")

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        # reads: old rows load through the new model definitions (filter by
        # seeded values — the init migration auto-seeds the GENERAL project)
        project = session.exec(select(Project).where(Project.path == "/tmp/seeded")).one()
        assert project.org_id is None and project.created_by is None
        assert session.exec(select(Conversation).where(Conversation.topic == "old chat")).one().org_id is None
        assert session.exec(select(Schedule).where(Schedule.title == "daily report")).one().created_by is None
        pin = session.exec(select(Pin).where(Pin.item_id == "c-1")).one()
        assert pin.user_id is None and pin.org_id is None
        setting = session.exec(select(Setting).where(Setting.key == "seeded_key")).one()
        assert setting.scope is None and setting.org_id is None

        # writes: new columns are usable
        session.add(Project(name="cloud", path="/tmp/cloud", org_id="o-1", created_by="u-1"))
        session.commit()
        assert session.exec(select(Project).where(Project.name == "cloud")).one().org_id == "o-1"


def test_downgrade_restores_previous_schema_and_constraints(desktop_db):
    cfg, db_path = desktop_db
    command.upgrade(cfg, "head")
    command.downgrade(cfg, PREVIOUS_HEAD)

    engine = create_engine(f"sqlite:///{db_path}")
    inspector = inspect(engine)
    for table, column in EXPECTED:
        assert column not in {c["name"] for c in inspector.get_columns(table)}
    for table, index in INDEXED:
        assert index not in {i["name"] for i in inspector.get_indexes(table)}

    # the SQLite batch rebuild must preserve constraints: uniques still enforce
    with engine.begin() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM pins")).scalar_one() == 1
        with pytest.raises(IntegrityError):
            conn.execute(text(
                f"INSERT INTO pins (id, item_type, item_id) VALUES ('{uuid4().hex}', 'conversation', 'c-1')"
            ))
    with engine.begin() as conn:
        with pytest.raises(IntegrityError):
            conn.execute(text(
                f"INSERT INTO settings (id, key, value) VALUES ('{uuid4().hex}', 'seeded_key', 'dup')"
            ))


def test_upgrade_again_after_downgrade(desktop_db):
    cfg, db_path = desktop_db
    command.upgrade(cfg, "head")
    command.downgrade(cfg, PREVIOUS_HEAD)
    command.upgrade(cfg, "head")

    inspector = inspect(create_engine(f"sqlite:///{db_path}"))
    for table, column in EXPECTED:
        assert column in {c["name"] for c in inspector.get_columns(table)}
    for table, index in INDEXED:
        assert index in {i["name"] for i in inspector.get_indexes(table)}


def test_unexpected_schema_fails_loudly(desktop_db):
    # Deterministic migration: a column that already exists is an unexpected
    # schema state and must fail, not be silently tolerated.
    cfg, db_path = desktop_db
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE projects ADD COLUMN org_id VARCHAR(36)"))
    with pytest.raises(OperationalError):
        command.upgrade(cfg, "head")


def test_migrated_schema_matches_fresh_models(desktop_db, tmp_path):
    # Drift guard: a DB upgraded through migrations must expose the same
    # columns (names AND types) as one created fresh from the current models.
    cfg, db_path = desktop_db
    command.upgrade(cfg, "head")

    import cowork.models.channel  # noqa: F401
    import cowork.models.message_event  # noqa: F401
    import cowork.models.skill  # noqa: F401
    import cowork.models.task_object  # noqa: F401
    from sqlmodel import SQLModel

    fresh_engine = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    SQLModel.metadata.create_all(fresh_engine)

    migrated = inspect(create_engine(f"sqlite:///{db_path}"))
    fresh = inspect(fresh_engine)
    new_cols = {(t, c) for t, c in EXPECTED}
    for table in sorted({t for t, _ in EXPECTED}):
        migrated_cols = {c["name"]: str(c["type"]) for c in migrated.get_columns(table)}
        fresh_cols = {c["name"]: str(c["type"]) for c in fresh.get_columns(table)}
        assert set(migrated_cols) == set(fresh_cols), (
            f"{table}: columns differ {set(migrated_cols) ^ set(fresh_cols)}"
        )
        for name in migrated_cols:
            if (table, name) in new_cols:
                assert migrated_cols[name] == fresh_cols[name], (
                    f"{table}.{name}: migrated type {migrated_cols[name]} != model type {fresh_cols[name]}"
                )


POSTGRES_URI = os.environ.get("COWORK_TEST_POSTGRES_URI", "")


@pytest.mark.skipif(not POSTGRES_URI, reason="set COWORK_TEST_POSTGRES_URI to a disposable database")
def test_postgres_upgrade_downgrade_cycle(monkeypatch):
    # Same contract on Postgres: full upgrade, columns + indexes, downgrade,
    # re-upgrade. Runs once CI (or a developer) provides a disposable DB.
    monkeypatch.setenv("DATABASE_URI", POSTGRES_URI)
    get_app_settings.cache_clear()
    cfg = Config("alembic.ini")
    try:
        command.upgrade(cfg, "head")
        inspector = inspect(create_engine(POSTGRES_URI))
        for table, column in EXPECTED:
            cols = {c["name"]: c for c in inspector.get_columns(table)}
            assert column in cols and cols[column]["nullable"]
        for table, index in INDEXED:
            assert index in {i["name"] for i in inspector.get_indexes(table)}
        command.downgrade(cfg, PREVIOUS_HEAD)
        inspector = inspect(create_engine(POSTGRES_URI))
        for table, column in EXPECTED:
            assert column not in {c["name"] for c in inspector.get_columns(table)}
        command.upgrade(cfg, "head")
    finally:
        command.downgrade(cfg, "base")
        get_app_settings.cache_clear()
