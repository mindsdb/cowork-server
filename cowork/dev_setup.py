"""Storage bootstrap helpers.

Applies schema migrations and required base rows. This is safe to run at
startup for the local SQLite deployment and remains exposed as a CLI helper
for development/test environments.
"""

from pathlib import Path

from sqlalchemy.engine import make_url
from sqlmodel import Session as SQLSession

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.migrations import run_schema_migrations
from cowork.db.session import get_engine
from cowork.models.project import Project
from cowork.services.projects import GENERAL_PROJECT, GENERAL_PROJECT_ID


def run_dev_setup() -> None:
    """Create local schema, seed required base rows, and run migrations."""
    settings = get_app_settings()
    db_uri = settings.database.uri

    parsed = make_url(db_uri)
    if (
        parsed.drivername.startswith("sqlite")
        and parsed.database
        and parsed.database != ":memory:"
    ):
        Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)

    engine = get_engine(db_uri)
    run_schema_migrations(engine, db_uri)
    # Re-fetch the engine — run_schema_migrations may have disposed and
    # rebuilt it (e.g. after deleting a corrupt SQLite DB).
    engine = get_engine(db_uri)

    with SQLSession(engine) as session:
        if session.get(Project, GENERAL_PROJECT_ID) is None:
            project_root = Path(settings.project.root_dir)
            general_path = project_root / GENERAL_PROJECT
            general_path.mkdir(parents=True, exist_ok=True)
            session.add(
                Project(
                    id=GENERAL_PROJECT_ID,
                    name=GENERAL_PROJECT,
                    path=str(general_path),
                    is_active=True,
                )
            )
            session.commit()

    # Migrate .env settings to DB (one-time, idempotent).
    from cowork.migrations import migrate_env_to_db

    with SQLSession(engine) as session:
        migrate_env_to_db(session)
