"""Dev-only bootstrap helpers.

This module intentionally keeps migration-like convenience logic out of
application startup. Use it explicitly in local development and tests.
"""

from pathlib import Path

from sqlalchemy.engine import make_url
from sqlmodel import SQLModel
from sqlmodel import Session as SQLSession

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.project import Project
from cowork.services.projects import GENERAL_PROJECT, GENERAL_PROJECT_ID

# Import models so SQLModel.metadata is fully populated before create_all.
import cowork.models.conversation  # noqa: F401
import cowork.models.file  # noqa: F401
import cowork.models.message  # noqa: F401
import cowork.models.message_event  # noqa: F401
import cowork.models.pin  # noqa: F401
import cowork.models.project  # noqa: F401
import cowork.models.schedule  # noqa: F401
import cowork.models.setting  # noqa: F401
import cowork.models.skill  # noqa: F401


def run_dev_setup() -> None:
    """Create local schema and seed required base rows for development."""
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
    SQLModel.metadata.create_all(engine)

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
