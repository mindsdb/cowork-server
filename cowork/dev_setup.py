"""Storage bootstrap helpers.

Applies schema migrations and required base rows. This is safe to run at
startup for the local SQLite deployment and remains exposed as a CLI helper
for development/test environments.
"""

import shutil
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

    # Migrate harness-local memory into ~/.cowork/memory, then wire runtime symlinks.
    import cowork.harnesses  # noqa: F401 — registers memory adapters

    from cowork.harnesses.memory.migration import migrate_harness_memory_to_shared
    from cowork.harnesses.memory.runtime import ensure_all_layouts

    with SQLSession(engine) as session:
        migrate_harness_memory_to_shared(session)

    ensure_all_layouts()

    # Migrate DB-backed skills to agentskills.io files (one-time, idempotent).
    from cowork.migrations import migrate_skills_to_files, seed_builtin_skills

    with SQLSession(engine) as session:
        migrate_skills_to_files(session)
        # Seed packaged builtin skills (versioned, idempotent).
        seed_builtin_skills(session)

    _link_hermes_skills_dir()


def _link_hermes_skills_dir() -> None:
    """Symlink Hermes's skills dir to cowork's canonical skills folder"""
    from cowork.harnesses.hermes_harness.settings import HermesHarnessSettings

    target = Path(get_app_settings().skill.root_dir)
    target.mkdir(parents=True, exist_ok=True)

    link = Path(HermesHarnessSettings().root_dir) / "skills"
    link.parent.mkdir(parents=True, exist_ok=True)

    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        shutil.rmtree(link) if link.is_dir() else link.unlink()

    link.symlink_to(target, target_is_directory=True)
