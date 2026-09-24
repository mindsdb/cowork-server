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

import hashlib
import logging
import stat
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import sqlalchemy as sa
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.models.setting import Setting
from cowork.models.shared_resource import SharedResourceAttribution
from cowork.models.task_object import TaskObject
from cowork.services.artifact_ownership import (
    ARTIFACT,
    artifact_resource_key,
    provenance_origin,
    record_artifact_owner,
)

logger = logging.getLogger(__name__)

SENTINEL_KEY = "_artifact_owner_backfill_v1"
_ARTIFACTS_SUBPATH = (".anton", "artifacts")
_LOCK_KEY = int.from_bytes(
    hashlib.blake2b(b"cowork\0artifact_owner_backfill_v1", digest_size=8).digest(),
    byteorder="big",
    signed=True,
)


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
    from cowork.services.shared_resources import _database_lock_engine

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
        return raw.exec(
            select(Setting).where(Setting.key == SENTINEL_KEY, Setting.scope.is_(None))
        ).first() is not None


def _write_sentinel(engine) -> None:
    with Session(engine) as raw:
        raw.add(Setting(key=SENTINEL_KEY, value="1"))
        try:
            raw.commit()
        except sa.exc.IntegrityError:
            raw.rollback()  # another replica finished first


def _is_real_directory(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode)
    except OSError:
        return False


def _is_candidate(folder: Path) -> bool:
    """A real artifact folder: not a link, not the locks dir, has metadata."""
    if folder.name.startswith(".") or not _is_real_directory(folder):
        return False
    try:
        (folder / "metadata.json").lstat()
    except OSError:
        return False
    return True


def _has_owner_row(session, project_id, slug: str) -> bool:
    return session.exec(
        session.select(SharedResourceAttribution).where(
            SharedResourceAttribution.resource_kind == ARTIFACT,
            SharedResourceAttribution.resource_key == artifact_resource_key(project_id, slug),
        )
    ).first() is not None


def _owner_from_provenance(session, project_id, folder: Path) -> str | None:
    conversation_id = provenance_origin(folder)
    if conversation_id is None:
        return None
    conversation = session.get(Conversation, conversation_id)
    if conversation is None or str(conversation.project_id) != str(project_id):
        return None
    return conversation.created_by or None


def _owner_from_task_objects(session, project_id, slug: str) -> str | None:
    rows = session.exec(
        session.select(TaskObject).where(
            TaskObject.project_id == project_id,
            # task_objects.KIND_ARTIFACT; importing it would cycle through
            # artifact_ownership -> task_objects.
            TaskObject.kind == "artifact",
            TaskObject.ref == slug,
        )
    ).all()
    if len(rows) != 1:
        return None
    conversation = session.get(Conversation, rows[0].conversation_id)
    if conversation is None:
        return None
    return conversation.created_by or None


def _backfill_project(engine, project_id, project_path: str, org_id: str, summary: BackfillSummary) -> None:
    project_dir = Path(project_path)
    base = project_dir.joinpath(*_ARTIFACTS_SUBPATH)
    if not _is_real_directory(project_dir / _ARTIFACTS_SUBPATH[0]) or not _is_real_directory(base):
        return
    # One raw session per organization: a raw session can only ever be
    # wrapped with a single tenant scope, and SYSTEM_SCOPE (local) would write
    # rows with org_id NULL that no org-scoped request could see.
    try:
        folders = sorted(base.iterdir())
    except OSError:
        # One unreadable project must not abort the pass for every other org.
        logger.warning("artifact_owner_backfill cannot list project=%s", project_id, exc_info=True)
        return
    with Session(engine) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=str(org_id)))
        for folder in folders:
            if not _is_candidate(folder):
                continue
            slug = folder.name
            try:
                if _has_owner_row(session, project_id, slug):
                    continue
                owner = _owner_from_provenance(session, project_id, folder) or (
                    _owner_from_task_objects(session, project_id, slug)
                )
                if owner and record_artifact_owner(
                    session, project_id, slug, owner, action="backfill"
                ):
                    summary.recorded += 1
                else:
                    summary.unknown.append((str(project_id), slug))
            except (sa.exc.OperationalError, sa.exc.DBAPIError):
                # A transient DB error (a connection blip mid-walk) must abort
                # the whole pass rather than mark the rest of this project's
                # artifacts `unknown` forever: the sentinel is written even if
                # the DB is back by the time the pass finishes, so a partial
                # walk would never be retried.
                session.rollback()
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
