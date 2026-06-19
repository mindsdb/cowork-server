"""Storage bootstrap helpers.

Applies schema migrations and required base rows. This is safe to run at
startup for the local SQLite deployment and remains exposed as a CLI helper
for development/test environments.
"""

import os
import subprocess
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


# Sibling repos a developer may have checked out locally, mapped to their
# default location relative to this repo root. Only anton is a source
# dependency today; add more here if other repos become path-overridable.
_LOCAL_SIBLINGS = {
    "anton-agent": ("ANTON_LOCAL_DIR", "anton"),
}


def link_local_siblings() -> None:
    """Overlay local sibling-repo checkouts as editable installs for dev.

    cowork-server pins anton-agent to a git branch in ``[tool.uv.sources]``,
    so a developer's local ``../anton`` feature branch is otherwise ignored.
    When the checkout is present we editable-install it on top of the synced
    environment; the dev launcher then runs the server with ``UV_NO_SYNC=1``
    so the overlay survives — a plain ``uv run`` re-syncs and reverts it.

    Auto-detects each sibling next to the repo root; the location can be
    overridden per package (e.g. ``ANTON_LOCAL_DIR``). Best-effort: a failure
    leaves the pinned dependency in place rather than blocking dev startup.
    """
    repo_root = Path(__file__).resolve().parents[1]
    for package, (env_var, default_dir) in _LOCAL_SIBLINGS.items():
        location = Path(os.environ.get(env_var) or repo_root.parent / default_dir)
        if not (location / "pyproject.toml").is_file():
            print(f"[dev-link] no local {package} at {location} — using the pinned dependency")
            continue
        print(f"[dev-link] linking {package} (editable) from {location}")
        result = subprocess.run(["uv", "pip", "install", "-e", str(location)], cwd=str(repo_root))
        if result.returncode != 0:
            print(f"[dev-link] WARNING: could not link {package}; keeping the pinned dependency")
