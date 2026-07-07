"""Schema migration helpers."""

from pathlib import Path

import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import Engine

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


def run_schema_migrations(engine: Engine, db_uri: str) -> None:
    """Run schema migrations, baselining pre-Alembic local databases."""
    config = _alembic_config(db_uri)
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        _stamp_existing_schema(config, connection)
        command.upgrade(config, "head")
