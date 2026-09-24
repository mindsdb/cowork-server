"""One-time owner backfill for project-root artifacts (ENG-2961).

Artifacts written to a project-level root between ENG-2056 and ENG-2961
(staging since 2026-09-14, prod since 2026-09-23) have no owner row. This pass
records one for each of them, once per environment, at cowork-server startup:

1. ``metadata.json`` ``provenance[0].conversation``, when that conversation
   exists in the organization, belongs to the artifact's project and has a
   creator. Provenance is agent-writable; trusting it for this one pass is a
   deliberate decision (no rewritten provenance is known to exist on Cloud),
   and nothing reads it as an owner source afterwards.
2. Otherwise exactly one ``task_objects`` row for ``(project, slug)`` whose
   conversation has a creator.
3. Otherwise the artifact stays ``unknown`` and is listed in the summary.

Legacy per-conversation roots are not touched: their owner is still derived
from the path. The result is only logged. To run the pass again, delete the
``_artifact_owner_backfill_v1`` setting row and restart cowork-server.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import sqlalchemy as sa
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.models.project import Project
from cowork.models.setting import Setting
from cowork.models.shared_resource import SharedResourceAttribution
from cowork.models.task_object import TaskObject
from cowork.services.artifact_ownership import (
    artifact_resource_key,
    conversation_creator,
    provenance_origin,
    record_artifact_owner,
)
from cowork.services.artifact_roots import (
    _is_real_directory,
    _storage_components_are_safe,
    project_artifacts_base,
)
from cowork.services.settings import SettingService
from cowork.services.shared_resources import (
    ARTIFACT,
    _database_lock_engine,
    advisory_lock_key,
)
from cowork.services.task_objects import KIND_ARTIFACT

logger = logging.getLogger(__name__)

SENTINEL_KEY = "_artifact_owner_backfill_v1"
_LOCK_KEY = advisory_lock_key(b"cowork\0artifact_owner_backfill_v1")


@dataclass
class BackfillSummary:
    recorded: int = 0
    unknown: list[tuple[str, str]] = field(default_factory=list)
    skipped_projects: int = 0


@contextmanager
def _advisory_try_lock(engine):
    """One non-blocking cross-replica lock; SQLite (tests, desktop) has none."""
    if engine.dialect.name != "postgresql":
        yield True
        return
    connection = _database_lock_engine(engine).connect()
    acquired = False
    try:
        acquired = bool(
            connection.execute(
                sa.text("SELECT pg_try_advisory_lock(:key)"), {"key": _LOCK_KEY}
            ).scalar()
        )
        yield acquired
    finally:
        try:
            if acquired:
                connection.execute(
                    sa.text("SELECT pg_advisory_unlock(:key)"), {"key": _LOCK_KEY}
                )
        finally:
            connection.close()


def _sentinel_present(engine) -> bool:
    with Session(engine) as raw:
        # No scope: the settings service then reads the global (NULL-scope) row.
        return SettingService(raw)._fetch_row(SENTINEL_KEY) is not None


def _write_sentinel(engine) -> None:
    with Session(engine) as raw:
        raw.add(Setting(key=SENTINEL_KEY, value="1"))
        try:
            raw.commit()
        except sa.exc.IntegrityError:
            raw.rollback()  # another replica finished first


def _is_candidate(folder: Path) -> bool:
    """A real artifact folder: not a link, not the locks dir, has metadata."""
    if folder.name.startswith(".") or not _is_real_directory(folder):
        return False
    try:
        (folder / "metadata.json").lstat()
    except OSError:
        return False
    return True


def _owned_slugs(session, project_id, slugs: list[str]) -> set[str]:
    """Slugs of ``project_id`` that already have an attribution row. One query."""
    keys = {artifact_resource_key(project_id, slug): slug for slug in slugs}
    rows = session.exec(
        session.select(SharedResourceAttribution).where(
            SharedResourceAttribution.resource_kind == ARTIFACT,
            SharedResourceAttribution.resource_key.in_(list(keys)),
        )
    ).all()
    return {keys[row.resource_key] for row in rows}


def _owner_from_task_objects(session, project_id, slug: str) -> str | None:
    rows = session.exec(
        session.select(TaskObject).where(
            TaskObject.project_id == project_id,
            TaskObject.kind == KIND_ARTIFACT,
            TaskObject.ref == slug,
        )
    ).all()
    if len(rows) != 1:
        return None
    return conversation_creator(session, rows[0].conversation_id)


def _backfill_project(engine, project_id, project_path: str, org_id: str, summary: BackfillSummary) -> None:
    project_dir = Path(project_path)
    # The same "no link in the writable chain" rule root discovery applies.
    if not _storage_components_are_safe(project_dir, may_be_absent=False):
        return
    base = project_artifacts_base(project_path)
    try:
        folders = [folder for folder in sorted(base.iterdir()) if _is_candidate(folder)]
    except OSError:
        # One unreadable project must not abort the pass for every other org.
        logger.warning("artifact_owner_backfill cannot list project=%s", project_id, exc_info=True)
        return
    if not folders:
        return
    # One raw session per project, wrapped with that project's org scope: a
    # raw session can only ever carry a single tenant scope, and SYSTEM_SCOPE
    # (local) would write rows with org_id NULL that no org-scoped request
    # could see.
    with Session(engine) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=str(org_id)))
        owned = _owned_slugs(session, project_id, [folder.name for folder in folders])
        # Several artifacts of one project usually name the same few
        # conversations; a miss is cached too, since the identity map does not.
        creators: dict = {}
        for folder in folders:
            slug = folder.name
            if slug in owned:
                continue
            try:
                owner = None
                conversation_id = provenance_origin(folder)
                if conversation_id is not None:
                    if conversation_id not in creators:
                        creators[conversation_id] = conversation_creator(
                            session, conversation_id, project_id=project_id
                        )
                    owner = creators[conversation_id]
                owner = owner or _owner_from_task_objects(session, project_id, slug)
                if owner and record_artifact_owner(
                    session, project_id, slug, owner, action="backfill"
                ):
                    summary.recorded += 1
                else:
                    summary.unknown.append((str(project_id), slug))
            except (sa.exc.OperationalError, sa.exc.InterfaceError):
                # A connection-level error (a blip mid-walk) must abort the
                # whole pass rather than mark the rest of this project's
                # artifacts `unknown` forever: the sentinel is written even if
                # the DB is back by the time the pass finishes, so a partial
                # walk would never be retried. Statement errors (DataError,
                # ProgrammingError, IntegrityError) are about one artifact and
                # stay per-artifact below; aborting on them would retry, and
                # fail, the same pass on every start.
                session.rollback()
                logger.error(
                    "artifact_owner_backfill aborted project=%s slug=%s",
                    project_id, slug, exc_info=True,
                )
                raise
            except Exception:
                session.rollback()
                logger.warning(
                    "artifact_owner_backfill failed project=%s slug=%s",
                    project_id, slug, exc_info=True,
                )
                summary.unknown.append((str(project_id), slug))


def run_artifact_owner_backfill(
    engine=None, *, try_lock=None, project_ids=None
) -> BackfillSummary | None:
    """Run the pass once per environment. None when skipped.

    Raises only when the pass cannot run at all (for example the database is
    unreachable); the sentinel is then not written and the next start retries.
    ``project_ids`` limits the pass to those projects; it exists for tests,
    whose database is shared across modules. Production passes nothing.

    The advisory-lock connection (NullPool, one connection) is held for the
    whole walk, which can take minutes on a large EFS tree; other replicas
    skip instead of waiting.
    """
    if get_app_settings().tenancy_mode != "org":
        return None
    engine = engine or get_engine(get_app_settings().database.uri)
    lock = try_lock or _advisory_try_lock
    if _sentinel_present(engine):
        return None
    with lock(engine) as acquired:
        if not acquired or _sentinel_present(engine):
            return None
        with Session(engine) as raw:
            query = select(Project.id, Project.path, Project.org_id)
            if project_ids is not None:
                query = query.where(Project.id.in_(list(project_ids)))
            projects = raw.exec(query).all()
        summary = BackfillSummary()
        for project_id, project_path, org_id in projects:
            if not org_id:
                summary.skipped_projects += 1
                continue
            _backfill_project(engine, project_id, project_path, org_id, summary)
        logger.warning(
            "artifact_owner_backfill recorded=%d unknown=%d skipped_projects=%d",
            summary.recorded, len(summary.unknown), summary.skipped_projects,
        )
        for project_id, slug in summary.unknown:
            logger.warning("artifact_owner_backfill unknown project=%s slug=%s", project_id, slug)
        _write_sentinel(engine)
    return summary
