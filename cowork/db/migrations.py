"""Schema migration helpers.

Local-first migration policy
----------------------------
Because users may downgrade cowork-server (or switch branches) at any time,
every Alembic migration MUST be **additive-only**:

  - ADD COLUMN, ADD TABLE, ADD INDEX  ✅
  - DROP / RENAME column or table     ❌

This guarantees that a database touched by a *newer* version is still a valid
superset of what an *older* version expects — SQLite silently ignores columns
it doesn't know about, and SQLModel selects explicit columns.

When the database is stamped at a revision this codebase doesn't recognise
(i.e. it was upgraded by a newer release), we **skip** migrations entirely
rather than re-stamping.  The superset schema is safe to run against.
"""

import logging
from pathlib import Path

import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
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


def _is_future_revision(config: Config, connection: sa.Connection) -> bool:
    """Return True if the DB is stamped at a revision this codebase doesn't know.

    This happens when a user downgrades cowork-server after a newer version
    already ran its migrations.  Because we enforce additive-only migrations,
    the schema is a safe superset — we just skip upgrading.
    """
    inspector = sa.inspect(connection)
    if "alembic_version" not in inspector.get_table_names():
        return False

    row = connection.execute(
        sa.text("SELECT version_num FROM alembic_version")
    ).first()
    if row is None:
        return False

    current_rev = row[0]
    script = ScriptDirectory.from_config(config)
    try:
        script.get_revision(current_rev)
    except Exception:
        logger.warning(
            "Database stamped at unknown revision %s — skipping migrations. "
            "This is expected after a cowork-server downgrade.",
            current_rev,
        )
        return True
    return False


def run_schema_migrations(engine: Engine, db_uri: str) -> None:
    """Run schema migrations, baselining pre-Alembic local databases."""
    config = _alembic_config(db_uri)
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        _stamp_existing_schema(config, connection)
        if _is_future_revision(config, connection):
            return
        command.upgrade(config, "head")
