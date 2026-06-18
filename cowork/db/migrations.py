"""Schema migration helpers."""

import logging
import os
from pathlib import Path

import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from alembic.util.exc import CommandError
from sqlalchemy.engine import Engine, make_url


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


def _try_upgrade(config: Config, connection: sa.Connection) -> None:
    """Run ``alembic upgrade head``, recovering from orphaned revisions.

    An orphaned revision occurs when the DB's ``alembic_version`` points to a
    migration that no longer exists in the installed package — typically after
    switching branches, a server downgrade, or a migration-history rewrite.

    Recovery: clear ``alembic_version`` and stamp at ``head``, trusting that
    the on-disk schema is compatible (the DB was at-or-ahead of head when the
    stale revision was written).
    """
    try:
        command.upgrade(config, "head")
    except CommandError as exc:
        if "Can't locate revision" not in str(exc):
            raise
        logger.warning(
            "Migration revision mismatch — the database references a revision "
            "that no longer exists in the installed migration tree: %s. "
            "Clearing alembic_version and re-stamping at head to recover.",
            exc,
        )
        connection.execute(sa.text("DELETE FROM alembic_version"))
        command.stamp(config, "head")


def run_schema_migrations(engine: Engine, db_uri: str) -> None:
    """Run schema migrations, baselining pre-Alembic local databases.

    If the migration fails because the DB references a revision that no longer
    exists (branch switch, version mismatch), attempts an in-place recovery.
    As a last resort for SQLite databases, deletes the DB file and retries
    from scratch so the server can start.
    """
    config = _alembic_config(db_uri)
    try:
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            _stamp_existing_schema(config, connection)
            _try_upgrade(config, connection)
    except Exception:
        # Last resort (SQLite only): delete the DB and rebuild from scratch.
        # This loses conversation history but unblocks the user.
        parsed = make_url(db_uri)
        db_path = parsed.database
        if not (
            parsed.drivername.startswith("sqlite")
            and db_path
            and db_path != ":memory:"
            and os.path.isfile(db_path)
        ):
            raise
        logger.error(
            "Migration recovery failed. Deleting %s and rebuilding the "
            "database from scratch so the server can start.",
            db_path,
        )
        engine.dispose()
        os.remove(db_path)
        # Evict the disposed engine from the cache so subsequent
        # get_engine() calls create a fresh connection to the new DB.
        from cowork.db.session import _engines

        _engines.pop(db_uri, None)

        from cowork.db.session import get_engine

        fresh_engine = get_engine(db_uri)
        fresh_config = _alembic_config(db_uri)
        with fresh_engine.begin() as connection:
            fresh_config.attributes["connection"] = connection
            command.upgrade(fresh_config, "head")
