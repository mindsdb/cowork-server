import sqlite3

import pytest
from sqlalchemy import create_engine, text
from sqlmodel import SQLModel

from alembic import command
from alembic.script import ScriptDirectory

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.migrations import (
    DatabaseSchemaAheadError,
    _alembic_config,
    run_schema_migrations,
)

# Import models so SQLModel.metadata can create a pre-Alembic legacy schema.
import cowork.models.conversation  # noqa: F401
import cowork.models.file  # noqa: F401
import cowork.models.message  # noqa: F401
import cowork.models.message_event  # noqa: F401
import cowork.models.pin  # noqa: F401
import cowork.models.project  # noqa: F401
import cowork.models.schedule  # noqa: F401
import cowork.models.setting  # noqa: F401
import cowork.models.skill  # noqa: F401


def _sqlite_uri(path) -> str:
    return f"sqlite:///{path}"


def _message_columns(path) -> set[str]:
    with sqlite3.connect(path) as connection:
        return {row[1] for row in connection.execute("pragma table_info(messages)")}


def _alembic_version(path) -> str:
    with sqlite3.connect(path) as connection:
        return connection.execute("select version_num from alembic_version").fetchone()[0]


def _set_alembic_version(path, revision: str) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("update alembic_version set version_num = ?", (revision,))
        connection.commit()


def _has_table(path, table_name: str) -> bool:
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "select name from sqlite_master where type='table' and name=?", (table_name,)
        ).fetchone()
        return row is not None


def _downgrade_to(engine, uri: str, revision: str) -> None:
    config = _alembic_config(uri)
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.downgrade(config, revision)


def expected_head() -> str:
    # Resolve the head from the script directory so new migrations don't
    # require updating a hardcoded revision here.
    return ScriptDirectory.from_config(_alembic_config("sqlite://")).get_current_head()


def test_schema_migrations_create_new_database(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()

    db_path = tmp_path / "new.db"
    uri = _sqlite_uri(db_path)
    engine = create_engine(uri)

    run_schema_migrations(engine, uri)

    assert "harness" in _message_columns(db_path)
    assert _alembic_version(db_path) == expected_head()


def test_schema_migrations_rerun_on_up_to_date_database_is_noop(tmp_path, monkeypatch):
    # A database already at head must upgrade cleanly a second time (no raise).
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()

    db_path = tmp_path / "current.db"
    uri = _sqlite_uri(db_path)
    engine = create_engine(uri)

    run_schema_migrations(engine, uri)
    run_schema_migrations(engine, uri)  # should not raise

    assert _alembic_version(db_path) == expected_head()


def test_schema_migrations_rejects_database_from_newer_build(tmp_path, monkeypatch):
    # Simulate ENG-324: a newer app stamped the DB at a revision this build
    # doesn't ship. The guard must raise a legible error instead of letting
    # Alembic fail deep inside upgrade with "Can't locate revision".
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()

    db_path = tmp_path / "ahead.db"
    uri = _sqlite_uri(db_path)
    engine = create_engine(uri)

    run_schema_migrations(engine, uri)
    _set_alembic_version(db_path, "ffffffffffff_from_the_future")

    with pytest.raises(DatabaseSchemaAheadError) as excinfo:
        run_schema_migrations(engine, uri)

    assert "ffffffffffff_from_the_future" in str(excinfo.value)
    # The DB is left untouched — still stamped at the future revision.
    assert _alembic_version(db_path) == "ffffffffffff_from_the_future"


def test_schema_migrations_upgrade_pre_alembic_database(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()

    db_path = tmp_path / "legacy.db"
    uri = _sqlite_uri(db_path)
    engine = create_engine(uri)
    SQLModel.metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE messages DROP COLUMN harness"))
        # A real pre-Alembic database predates the channel tables too; drop them
        # so the upgrade path recreates them via the channels migration.
        for table in (
            "task_objects",
            "channel_events",
            "channel_sessions",
            "channel_bindings",
            "channel_installations",
        ):
            connection.execute(text(f"DROP TABLE IF EXISTS {table}"))

    run_schema_migrations(engine, uri)

    assert "harness" in _message_columns(db_path)
    assert _alembic_version(db_path) == expected_head()


def test_task_objects_downgrade_drops_table(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()

    db_path = tmp_path / "downgrade.db"
    uri = _sqlite_uri(db_path)
    engine = create_engine(uri)
    run_schema_migrations(engine, uri)
    assert _has_table(db_path, "task_objects")

    _downgrade_to(engine, uri, "c4e7a1b9d2f0")

    assert not _has_table(db_path, "task_objects")
    assert _alembic_version(db_path) == "c4e7a1b9d2f0"


def test_task_objects_downgrade_guards_missing_table(tmp_path, monkeypatch):
    # Mirrors the upgrade guard: downgrade() must not crash if task_objects
    # was already removed out-of-band before the alembic_version pointer is
    # walked back past this revision.
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()

    db_path = tmp_path / "downgrade_missing.db"
    uri = _sqlite_uri(db_path)
    engine = create_engine(uri)
    run_schema_migrations(engine, uri)

    with engine.begin() as connection:
        connection.execute(text("DROP TABLE task_objects"))

    _downgrade_to(engine, uri, "c4e7a1b9d2f0")  # must not raise

    assert _alembic_version(db_path) == "c4e7a1b9d2f0"
