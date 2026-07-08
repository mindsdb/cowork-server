"""Schema migration helpers."""

import logging
from pathlib import Path

import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class DatabaseSchemaAheadError(RuntimeError):
    """The database is stamped at a migration this build does not know about.

    This happens when a newer version of the app advanced the schema and an
    older build is then opened against the same database (ENG-324). Alembic
    would otherwise fail deep inside ``command.upgrade`` with an opaque
    "Can't locate revision identified by '<rev>'" that gives the user nothing
    to act on. We detect the condition up front and raise this instead, with a
    message that names the offending revision(s) and tells the user what to do.
    """


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


def _assert_db_not_ahead(config: Config, connection: sa.Connection) -> None:
    """Fail fast (and legibly) if the DB is newer than this build's migrations.

    Compares the revision(s) recorded in the database's ``alembic_version``
    table against every revision this build ships. If the database references a
    revision we don't have, upgrading is impossible — the schema was written by
    a newer app — so we raise :class:`DatabaseSchemaAheadError` before Alembic
    hits its own opaque failure.

    A fresh database (no ``alembic_version`` row) reports no current heads, so
    this is a no-op there and on any in-sync or behind database.
    """
    script = ScriptDirectory.from_config(config)
    known_revisions = {script_rev.revision for script_rev in script.walk_revisions()}

    context = MigrationContext.configure(connection)
    current_heads = set(context.get_current_heads())

    unknown = current_heads - known_revisions
    if unknown:
        revs = ", ".join(sorted(unknown))
        logger.error(
            "Database is ahead of this build: unknown migration(s) %s. "
            "Known head(s): %s.",
            revs,
            ", ".join(script.get_heads()) or "(none)",
        )
        raise DatabaseSchemaAheadError(
            f"This database was created by a newer version of the app "
            f"(unknown migration(s): {revs}). Update to the latest version, or "
            f"reset this build's database, to continue."
        )


def run_schema_migrations(engine: Engine, db_uri: str) -> None:
    """Run schema migrations, baselining pre-Alembic local databases."""
    config = _alembic_config(db_uri)
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        _stamp_existing_schema(config, connection)
        _assert_db_not_ahead(config, connection)
        command.upgrade(config, "head")
