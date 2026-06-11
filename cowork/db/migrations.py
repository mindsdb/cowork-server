"""Schema migration helpers."""

import logging
from pathlib import Path

import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.util.exc import CommandError
from sqlalchemy.engine import Engine


logger = logging.getLogger(__name__)

INITIAL_REVISION = "93375a6617f4"
INITIAL_SCHEMA_TABLES = {
    "conversations",
    "files",
    "message_events",
    "messages",
    "pins",
    "projects",
    "schedule_runs",
    "schedules",
    "settings",
    "skills",
}


def _script_location() -> Path:
    return Path(__file__).resolve().parent / "alembic"


def _alembic_config(db_uri: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_script_location()))
    config.set_main_option("sqlalchemy.url", db_uri)
    return config


def _stamp_existing_schema(config: Config, connection: sa.Connection) -> None:
    inspector = sa.inspect(connection)
    table_names = set(inspector.get_table_names())
    if not table_names or "alembic_version" in table_names:
        return

    if INITIAL_SCHEMA_TABLES.issubset(table_names):
        command.stamp(config, INITIAL_REVISION)


def _recover_unknown_revision(config: Config, connection: sa.Connection) -> None:
    """If the DB is stamped with a revision this codebase doesn't know about,
    re-stamp to head so the server can start.  This happens when a user
    downgrades cowork-server (or switches branches) after a migration ran
    against their local SQLite database."""
    inspector = sa.inspect(connection)
    if "alembic_version" not in inspector.get_table_names():
        return

    result = connection.execute(sa.text("SELECT version_num FROM alembic_version"))
    row = result.first()
    if row is None:
        return

    current_rev = row[0]
    script = ScriptDirectory.from_config(config)
    try:
        script.get_revision(current_rev)
    except Exception:
        # Revision not in this codebase — re-stamp to head.
        logger.warning(
            "Database stamped at unknown revision %s — re-stamping to head. "
            "This is normal after a cowork-server downgrade.",
            current_rev,
        )
        connection.execute(sa.text("DELETE FROM alembic_version"))
        command.stamp(config, "head")


def run_schema_migrations(engine: Engine, db_uri: str) -> None:
    """Run schema migrations, baselining pre-Alembic local databases."""
    config = _alembic_config(db_uri)
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        _stamp_existing_schema(config, connection)
        _recover_unknown_revision(config, connection)
        command.upgrade(config, "head")
