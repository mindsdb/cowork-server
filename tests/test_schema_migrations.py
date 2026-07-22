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
        # It also predates the tenancy columns (a3f9c2e8b1d4); drop them (and
        # their indexes — SQLite can't drop an indexed column) so that
        # migration recreates them. uq_pins_item_user (d2e8f1a4c7b9) rides on
        # pins.user_id and must go for the same reason.
        for index in (
            "ix_projects_org_id",
            "ix_conversations_org_id",
            "ix_files_org_id",
            "ix_schedules_org_id",
            "ix_pins_user_id_org_id",
            "uq_pins_item_user",
        ):
            connection.execute(text(f"DROP INDEX IF EXISTS {index}"))
        for table, column in (
            ("projects", "org_id"), ("projects", "created_by"),
            ("conversations", "org_id"), ("conversations", "created_by"),
            ("messages", "created_by"),
            ("files", "org_id"), ("files", "created_by"),
            ("schedules", "org_id"), ("schedules", "created_by"),
            ("pins", "user_id"), ("pins", "org_id"),
            ("settings", "scope"), ("settings", "user_id"), ("settings", "org_id"),
        ):
            connection.execute(text(f"ALTER TABLE {table} DROP COLUMN {column}"))

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


# ── ENG-338: attachment purpose re-keying (f7d2b9e4a1c6) ─────────────────

ATTACH_REKEY_REV = "f7d2b9e4a1c6"
SID = "d6ad2000-915b-4915-baf4-369e2db05f17"
ORPHAN_SID = "e7be3111-026c-5026-cbf5-47af3ec16f28"


def _upgrade_to(engine, uri: str, revision: str) -> None:
    config = _alembic_config(uri)
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, revision)


def _insert_file(path, file_id: str, purpose: str) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO files (id, filename, content_type, size, purpose, path,"
            " created_at, modified_at) VALUES (?, 'f.csv', 'text/csv', 1, ?, '',"
            " CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            (file_id, purpose),
        )
        connection.commit()


def _purposes(path) -> dict[str, str]:
    with sqlite3.connect(path) as connection:
        return dict(connection.execute("SELECT id, purpose FROM files"))


def _seed_conversation(path, sid: str, project_name: str) -> None:
    """Project + conversation the downgrade join can resolve. Uuid columns
    store 32-char hex (see the init migration's GENERAL_PROJECT_ID.hex)."""
    project_hex = "aa" * 16
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO projects (id, name, path, is_active) VALUES (?, ?, '', 0)",
            (project_hex, project_name),
        )
        connection.execute(
            "INSERT INTO conversations (id, topic, project_id) VALUES (?, 't', ?)",
            (sid.replace("-", ""), project_hex),
        )
        connection.commit()


def test_attachment_rekey_upgrade_rewrites_old_format_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()

    db_path = tmp_path / "rekey.db"
    uri = _sqlite_uri(db_path)
    engine = create_engine(uri)
    _upgrade_to(engine, uri, "e8b3c5d7a9f1")  # the revision before the rekey

    _insert_file(db_path, "01" * 16, f"attachment:My Project:{SID}")
    _insert_file(db_path, "02" * 16, f"attachment:odd:name:with:colons:{SID}")
    _insert_file(db_path, "03" * 16, f"attachment:{SID}")  # already new-format
    _insert_file(db_path, "04" * 16, "assistants")  # non-attachment

    _upgrade_to(engine, uri, "head")

    purposes = _purposes(db_path)
    assert purposes["01" * 16] == f"attachment:{SID}"
    assert purposes["02" * 16] == f"attachment:{SID}"
    assert purposes["03" * 16] == f"attachment:{SID}"
    assert purposes["04" * 16] == "assistants"
    # The migration also creates the purpose index the boot-time rekey walks.
    with sqlite3.connect(db_path) as connection:
        indexes = {row[1] for row in connection.execute("pragma index_list(files)")}
    assert "ix_files_purpose" in indexes


def test_attachment_rekey_downgrade_restores_names_best_effort(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()

    db_path = tmp_path / "rekey-down.db"
    uri = _sqlite_uri(db_path)
    engine = create_engine(uri)
    run_schema_migrations(engine, uri)

    _seed_conversation(db_path, SID, "Campaign-Q3")
    _insert_file(db_path, "01" * 16, f"attachment:{SID}")
    # No conversation row for this one — downgrade can't resolve a project.
    _insert_file(db_path, "02" * 16, f"attachment:{ORPHAN_SID}")

    _downgrade_to(engine, uri, "e8b3c5d7a9f1")

    purposes = _purposes(db_path)
    assert purposes["01" * 16] == f"attachment:Campaign-Q3:{SID}"
    assert purposes["02" * 16] == f"attachment:{ORPHAN_SID}"  # left as-is

    # Round-trip: re-upgrading rewrites the restored row back to new-format.
    _upgrade_to(engine, uri, "head")
    assert _purposes(db_path)["01" * 16] == f"attachment:{SID}"


def test_startup_rekeys_stray_rows_written_by_old_builds(tmp_path, monkeypatch):
    # A rolled-back skip-on-unknown-revision build can write old-format rows
    # AFTER f7d2b9e4a1c6 already ran; `alembic upgrade head` is then a no-op,
    # so run_schema_migrations must heal them itself on every boot.
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()

    db_path = tmp_path / "stray.db"
    uri = _sqlite_uri(db_path)
    engine = create_engine(uri)
    run_schema_migrations(engine, uri)
    assert _alembic_version(db_path) == expected_head()

    _insert_file(db_path, "05" * 16, f"attachment:Renamed Project:{SID}")

    run_schema_migrations(engine, uri)  # simulated next boot

    assert _purposes(db_path)["05" * 16] == f"attachment:{SID}"
