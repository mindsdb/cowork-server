import json
import sqlite3

from sqlalchemy import create_engine, inspect, text
from sqlmodel import SQLModel, Session

from alembic.script import ScriptDirectory

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.migrations import _alembic_config, run_schema_migrations
from cowork.services.artifact_versions import ArtifactVersionService

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


def _table_columns(path, table: str) -> set[str]:
    with sqlite3.connect(path) as connection:
        return {row[1] for row in connection.execute(f"pragma table_info({table})")}


def _alembic_version(path) -> str:
    with sqlite3.connect(path) as connection:
        return connection.execute("select version_num from alembic_version").fetchone()[0]


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


def test_schema_migrations_handle_partial_artifact_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()

    db_path = tmp_path / "partial-artifacts.db"
    uri = _sqlite_uri(db_path)
    engine = create_engine(uri)
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE artifacts (
                    id CHAR(32) NOT NULL PRIMARY KEY,
                    slug VARCHAR(255) NOT NULL,
                    title VARCHAR(255) NOT NULL
                )
                """
            )
        )

    run_schema_migrations(engine, uri)

    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    assert {
        "artifact_versions",
        "artifact_version_files",
        "artifact_drafts",
        "artifact_deployments",
        "artifact_comments",
        "artifact_activity_events",
        "project_collaborators",
        "project_invitations",
        "project_notification_hooks",
        "notification_deliveries",
    }.issubset(tables)
    assert {"path", "current_version_id", "last_known_good_version_id"}.issubset(
        _table_columns(db_path, "artifacts")
    )
    assert {"branch_name", "forked_from_version_id", "pre_snapshot_version_id", "snapshot_role", "store_path"}.issubset(
        _table_columns(db_path, "artifact_versions")
    )
    assert {"proposed_patch", "notification_state", "review_verdict"}.issubset(
        _table_columns(db_path, "artifact_comments")
    )
    assert {"token_hash", "expires_at", "invited_by_email"}.issubset(
        _table_columns(db_path, "project_invitations")
    )
    folder = tmp_path / "project" / ".anton" / "artifacts" / "partial"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(
        json.dumps({"slug": "partial", "name": "Partial Artifact", "type": "document"}),
        encoding="utf-8",
    )
    (folder / "report.md").write_text("# Migrated\n", encoding="utf-8")
    with Session(engine) as session:
        version = ArtifactVersionService(session, tmp_path / "store").snapshot_artifact(folder)
        assert version.file_count == 1
    assert _alembic_version(db_path) == expected_head()


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
        for table in ("channel_events", "channel_sessions", "channel_bindings", "channel_installations"):
            connection.execute(text(f"DROP TABLE IF EXISTS {table}"))

    run_schema_migrations(engine, uri)

    assert "harness" in _message_columns(db_path)
    assert _alembic_version(db_path) == expected_head()
