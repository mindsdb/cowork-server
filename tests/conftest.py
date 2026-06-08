"""Test bootstrap for the channel tests.

Isolates the app onto a throwaway SQLite DB + master key BEFORE any cowork
module is imported (settings/engine are read at import time), then builds the
schema directly and seeds the ``general`` project the runtime depends on.
"""
import importlib
import os
import pkgutil
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="cowork-chan-test-"))
# Force isolation (assignment, not setdefault — never touch a real DB).
os.environ["DATABASE_URI"] = f"sqlite:///{TMP / 'test.db'}"
os.environ["MASTER_KEY_PATH"] = str(TMP / "master.key")
os.environ["COWORK_PUBLIC_BASE_URL"] = "https://hooks.example.com"
os.environ["COWORK_CONVERSATION_LINK_TEMPLATE"] = "https://app.example.com/c/{conversation_id}"
os.environ["COWORK_PROJECTS_DIR"] = str(TMP / "projects")
os.environ["ENV"] = "test"

import pytest
from sqlmodel import Session, SQLModel


@pytest.fixture(scope="session", autouse=True)
def db_schema():
    # Import every model module so the mapper + metadata are complete.
    import cowork.models as models_pkg
    for _, name, _ in pkgutil.iter_modules(models_pkg.__path__):
        importlib.import_module(f"cowork.models.{name}")

    from cowork.common.settings.app_settings import get_app_settings
    from cowork.db.session import get_engine
    from cowork.models.project import Project
    from cowork.services.projects import GENERAL_PROJECT, GENERAL_PROJECT_ID

    engine = get_engine(get_app_settings().database.uri)
    SQLModel.metadata.create_all(engine)

    # Under the projects root so the artifacts scanner treats it as registered.
    general_dir = TMP / "projects" / "general"
    general_dir.mkdir(parents=True, exist_ok=True)
    with Session(engine) as session:
        if session.get(Project, GENERAL_PROJECT_ID) is None:
            session.add(Project(id=GENERAL_PROJECT_ID, name=GENERAL_PROJECT, path=str(general_dir)))
            session.commit()
    yield
