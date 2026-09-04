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


def _migrate_env_to_db_if_local(session) -> None:
    """One-time `.env` -> DB migration. Desktop-only, for the same reason as
    the skills migration below: migrate_env_to_db reads cowork_home()/".env"
    (cowork/migrations.py's _ENV_PATH) and upserts what it finds through an
    unscoped SettingService, i.e. into global rows no org owns. In org mode
    COWORK_HOME is /mnt/cowork-shared (deployment values.yaml), so that path
    is the ROOT of the shared tree every organization's agent writes into: a
    dropped .env there would seed this deployment's provider keys and
    endpoints at the next boot, for every tenant.
    """
    if get_app_settings().tenancy_mode == "org":
        return
    from cowork.migrations import migrate_env_to_db
    migrate_env_to_db(session)


def _distribute_skill_links() -> None:
    """Boot-time symlink fan-out of the skill store into project dirs.

    Desktop-only: in org mode this reads the unkeyed root and scans every
    project dir. Cloud turns read skills straight off the shared mount instead
    (no payload); the pod's org gets seeded lazily, see
    `_stage_remote_workspace_files` and `SkillService.ensure_builtin_skills`.
    """
    if get_app_settings().tenancy_mode == "org":
        return
    from cowork.services.skill_links import reconcile_all
    from cowork.services.skills import SkillService

    reconcile_all(SkillService().list_skills())


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
            # Dir only on desktop: org mode never adopts this NULL-org row.
            if settings.tenancy_mode != "org":
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
    from cowork.migrations import backfill_minds_url

    with SQLSession(engine) as session:
        _migrate_env_to_db_if_local(session)
        # Rewrite the legacy MindsHub host (mdb.ai -> api.mindshub.ai) for
        # users who configured MindsHub before the default flipped. Idempotent;
        # runs every boot (not gated by the env-migration sentinel, since
        # affected users already passed it).
        backfill_minds_url(session)

    # Migrate harness-local memory into ~/.cowork/memory, then wire runtime
    # symlinks. Desktop-only: both write the unkeyed root; org-mode memory is
    # org-first under the shared root and created on demand.
    import cowork.harnesses  # noqa: F401 — registers memory adapters

    # Needs the registry above, so it cannot sit with the migrations further up.
    # Not desktop-only: a stored harness the install cannot build breaks every
    # settings read in either mode.
    from cowork.migrations import reset_unbuildable_harness

    with SQLSession(engine) as session:
        reset_unbuildable_harness(session)

    if settings.tenancy_mode != "org":
        from cowork.harnesses.memory.migration import migrate_harness_memory_to_shared
        from cowork.harnesses.memory.runtime import ensure_all_layouts

        with SQLSession(engine) as session:
            migrate_harness_memory_to_shared(session)

        ensure_all_layouts()

    # Skill migration + builtin seeding write the unkeyed root via an unscoped
    # SkillService, so both are desktop-only. An org store is per-org and there
    # is no org-creation hook at boot, so org deployments seed lazily on first
    # read instead — see `SkillService.ensure_builtin_skills`.
    if get_app_settings().tenancy_mode != "org":
        from cowork.migrations import migrate_skills_to_files, seed_builtin_skills

        with SQLSession(engine) as session:
            migrate_skills_to_files(session)
            seed_builtin_skills(session)
            _distribute_skill_links()

