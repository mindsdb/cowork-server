from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine, pool
from sqlalchemy.engine import make_url
from sqlmodel import SQLModel

# Import models so SQLModel.metadata is fully populated for autogenerate.
import cowork.models.conversation  # noqa: F401
import cowork.models.file  # noqa: F401
import cowork.models.message  # noqa: F401
import cowork.models.message_event  # noqa: F401
import cowork.models.pin  # noqa: F401
import cowork.models.project  # noqa: F401
import cowork.models.schedule  # noqa: F401
import cowork.models.setting  # noqa: F401
import cowork.models.skill  # noqa: F401
from cowork.common.settings.app_settings import get_app_settings


def _ensure_sqlite_dir(url: str) -> None:
    parsed = make_url(url)
    if not parsed.drivername.startswith("sqlite"):
        return
    db_path = parsed.database
    if db_path and db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata


def _database_url() -> str:
    database_uri = get_app_settings().database.uri
    return database_uri or config.get_main_option("sqlalchemy.url", database_uri)


def run_migrations_offline() -> None:
    url = _database_url()
    _ensure_sqlite_dir(url)
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()
        return

    url = _database_url()
    _ensure_sqlite_dir(url)
    connectable = create_engine(url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
