import sqlite3

import pytest
from alembic import command
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text
from sqlmodel import SQLModel

# Import models so SQLModel.metadata can create a pre-Alembic legacy schema.
import cowork.models.conversation  # noqa: F401
import cowork.models.file  # noqa: F401
import cowork.models.message  # noqa: F401
import cowork.models.message_event  # noqa: F401
import cowork.models.pin  # noqa: F401
import cowork.models.project  # noqa: F401
import cowork.models.schedule  # noqa: F401
import cowork.models.setting  # noqa: F401
import cowork.models.shared_resource  # noqa: F401
import cowork.models.skill  # noqa: F401
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.migrations import (
    DatabaseSchemaAheadError,
    _alembic_config,
    run_schema_migrations,
)


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


def _upgrade_to(engine, uri: str, revision: str) -> None:
    config = _alembic_config(uri)
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, revision)


def _insert_setting(connection, *, key, value, scope=None, org_id=None, user_id=None):
    connection.execute(
        text(
            "INSERT INTO settings (id, key, value, scope, org_id, user_id) VALUES "
            "(lower(hex(randomblob(16))), :key, :value, :scope, :org_id, :user_id)"
        ),
        {"key": key, "value": value, "scope": scope, "org_id": org_id, "user_id": user_id},
    )


def expected_head() -> str:
    # Resolve the head from the script directory so new migrations don't
    # require updating a hardcoded revision here.
    return ScriptDirectory.from_config(_alembic_config("sqlite://")).get_current_head()


def test_migration_graph_has_single_head():
    # Two migrations forking off one parent leave `upgrade head` unresolvable,
    # which breaks server startup and the deploy migrate step, not just tests.
    heads = ScriptDirectory.from_config(_alembic_config("sqlite://")).get_heads()
    assert len(heads) == 1, (
        f"migration graph has {len(heads)} heads ({', '.join(sorted(heads))}). "
        "Add a merge revision joining them: alembic merge -m '<why>' "
        + " ".join(sorted(heads))
    )


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


def test_code_control_migration_backfills_parent_projection_for_existing_records(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    db_path = tmp_path / "code-control-upgrade.db"
    uri = _sqlite_uri(db_path)
    engine = create_engine(uri)
    _upgrade_to(engine, uri, "a4c8e1f6b3d9")

    with engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE code_control_records (
                namespace_id VARCHAR(128) NOT NULL,
                collection VARCHAR(32) NOT NULL,
                document_id VARCHAR(160) NOT NULL,
                payload JSON NOT NULL,
                assigned_computer_id VARCHAR(128),
                lifecycle_status VARCHAR(32),
                revision INTEGER NOT NULL DEFAULT 1,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (namespace_id, collection, document_id)
            )
        """))
        connection.execute(
            text("""
                INSERT INTO code_control_records
                    (namespace_id, collection, document_id, payload)
                VALUES (:namespace_id, 'runs', 'run-one', :payload)
            """),
            {"namespace_id": "org", "payload": '{"id":"run-one","task_id":"task-one"}'},
        )

    _upgrade_to(engine, uri, "head")

    with engine.begin() as connection:
        parent_id = connection.execute(text("""
            SELECT parent_id FROM code_control_records
            WHERE namespace_id = 'org' AND collection = 'runs' AND document_id = 'run-one'
        """)).scalar_one()
    assert parent_id == "task-one"


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
            # The settings scope-split partial indexes (c8e1a4f7b2d9) filter on
            # settings.scope, so they must go before that column is dropped too.
            "uq_settings_key_global",
            "uq_settings_key_org",
            "uq_settings_key_user",
            # The single-flight partial index (a7e4c2f1b9d3) is created by its
            # migration, so drop the model-declared copy first.
            "uq_schedule_runs_one_active",
        ):
            connection.execute(text(f"DROP INDEX IF EXISTS {index}"))
        # The settings row-shape CHECK (c8e1a4f7b2d9) references scope/org_id/
        # user_id, so SQLite won't let those columns drop while it exists.
        # Rebuild the table without it (batch) before the DROP COLUMN loop.
        from alembic.migration import MigrationContext
        from alembic.operations import Operations
        _ops = Operations(MigrationContext.configure(connection))
        with _ops.batch_alter_table("settings") as _batch:
            _batch.drop_constraint("ck_settings_scope_shape", type_="check")
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


def test_shared_resource_audit_upgrade_and_downgrade(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()

    db_path = tmp_path / "shared-resources.db"
    uri = _sqlite_uri(db_path)
    engine = create_engine(uri)
    run_schema_migrations(engine, uri)

    assert _has_table(db_path, "shared_resource_attributions")
    assert _has_table(db_path, "shared_resource_mutations")

    _downgrade_to(engine, uri, "a4c8e1f6b3d9")
    assert not _has_table(db_path, "shared_resource_attributions")
    assert not _has_table(db_path, "shared_resource_mutations")

    _upgrade_to(engine, uri, "head")
    assert _has_table(db_path, "shared_resource_attributions")
    assert _has_table(db_path, "shared_resource_mutations")


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


def test_settings_scope_split_removes_global_key_uniqueness(tmp_path, monkeypatch):
    # The REAL migration path (init creates UNIQUE(key), c8e1a4f7b2d9 drops it),
    # driven through alembic — NOT create_all, which never had the constraint.
    import sqlalchemy
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    uri = _sqlite_uri(tmp_path / "split.db")
    engine = create_engine(uri)
    _upgrade_to(engine, uri, "head")

    # one key across all three scopes + two orgs + the SAME user in both orgs
    with engine.begin() as c:
        _insert_setting(c, key="k", value="g")
        _insert_setting(c, key="k", value="oA", scope="org", org_id="A")
        _insert_setting(c, key="k", value="oB", scope="org", org_id="B")
        _insert_setting(c, key="k", value="uA", scope="user", org_id="A", user_id="u")
        _insert_setting(c, key="k", value="uB", scope="user", org_id="B", user_id="u")

    # but a duplicate within one scope is still rejected
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        with engine.begin() as c:
            _insert_setting(c, key="k", value="dup", scope="org", org_id="A")


def test_settings_scope_split_downgrade_preflights_duplicates(tmp_path, monkeypatch):
    # A downgrade that can't restore UNIQUE(key) must abort BEFORE touching the
    # schema, leaving the DB intact at head.
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    db = tmp_path / "dg.db"
    uri = _sqlite_uri(db)
    engine = create_engine(uri)
    _upgrade_to(engine, uri, "head")
    with engine.begin() as c:
        _insert_setting(c, key="k", value="oA", scope="org", org_id="A")
        _insert_setting(c, key="k", value="oB", scope="org", org_id="B")

    with pytest.raises(Exception, match="cannot downgrade"):
        _downgrade_to(engine, uri, "b3d7f1a9c5e2")

    # schema untouched: still at head with the partial indexes present
    assert _alembic_version(db) == expected_head()
    with sqlite3.connect(db) as conn:
        names = {
            r[0] for r in conn.execute(
                "select name from sqlite_master where type='index' and tbl_name='settings'"
            )
        }
    assert {"uq_settings_key_global", "uq_settings_key_org", "uq_settings_key_user"} <= names


def _insert_channel_installation(connection, *, channel_type, org_id=None):
    connection.execute(
        text(
            "INSERT INTO channel_installations (id, channel_type, display_name, enabled, status, org_id) "
            "VALUES (lower(hex(randomblob(16))), :channel_type, :channel_type, 0, 'disconnected', :org_id)"
        ),
        {"channel_type": channel_type, "org_id": org_id},
    )


def test_channel_installations_per_org_allows_one_per_org(tmp_path, monkeypatch):
    import sqlalchemy

    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    db = tmp_path / "channels.db"
    uri = _sqlite_uri(db)
    engine = create_engine(uri)
    _upgrade_to(engine, uri, "head")

    with engine.begin() as c:
        _insert_channel_installation(c, channel_type="slack", org_id="A")
        _insert_channel_installation(c, channel_type="slack", org_id="B")

    # but a duplicate within one org (or within local mode) is still rejected
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        with engine.begin() as c:
            _insert_channel_installation(c, channel_type="slack", org_id="A")
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        with engine.begin() as c:
            _insert_channel_installation(c, channel_type="telegram")
            _insert_channel_installation(c, channel_type="telegram")


def test_channel_installations_per_org_downgrade_preflights_duplicates(tmp_path, monkeypatch):
    # Same shape as the settings scope split: a downgrade that can't restore
    # the single global UniqueConstraint(channel_type) must abort BEFORE
    # touching the schema, leaving the DB intact at head.
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    db = tmp_path / "channels-dg.db"
    uri = _sqlite_uri(db)
    engine = create_engine(uri)
    _upgrade_to(engine, uri, "head")
    with engine.begin() as c:
        _insert_channel_installation(c, channel_type="slack", org_id="A")
        _insert_channel_installation(c, channel_type="slack", org_id="B")

    with pytest.raises(Exception, match="cannot downgrade"):
        _downgrade_to(engine, uri, "c3f8a2b6d1e4")

    assert _alembic_version(db) == expected_head()
    with sqlite3.connect(db) as conn:
        names = {
            r[0] for r in conn.execute(
                "select name from sqlite_master where type='index' and tbl_name='channel_installations'"
            )
        }
    assert {"uq_channel_installations_type_global", "uq_channel_installations_type_org"} <= names


def test_migrated_schema_enforces_one_attribution_row_per_resource(
    tmp_path, monkeypatch
):
    """The unique key the whole first-writer race design rests on.

    Two replicas can both pass the pre-check at READ COMMITTED, so the loser is
    supposed to lose on this constraint and adopt the winner's row. Without it
    a resource ends up with two creator rows and the authorization read becomes
    nondeterministic. The suite builds its schema from the models, so only a
    real migration run proves the constraint actually ships.
    """
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()

    db_path = tmp_path / "attribution.db"
    uri = _sqlite_uri(db_path)
    run_schema_migrations(create_engine(uri), uri)

    with sqlite3.connect(db_path) as connection:
        indexes = connection.execute(
            "select name, \"unique\" from pragma_index_list('shared_resource_attributions')"
        ).fetchall()
        unique_columns = {
            tuple(
                row[2]
                for row in connection.execute(
                    f"select * from pragma_index_info('{name}')"
                ).fetchall()
            )
            for name, is_unique in indexes
            if is_unique
        }

    assert ("org_id", "resource_kind", "resource_key") in unique_columns, (
        "shared_resource_attributions lost its unique key; concurrent creates "
        f"can now record two creators. Unique indexes found: {unique_columns}"
    )
