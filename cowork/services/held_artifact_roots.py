"""The held_artifact_roots rows (cowork/models/held_artifact_root.py): which
artifact roots have held an artifact folder.

Org tenancy only: there `COWORK_HOME` is the shared file store, where every
open and stat is a round trip to the file server. The artifacts list opens
only the roots these rows name (artifact_roots.with_held_roots), so a project
or a conversation workspace that never held an artifact costs it no file
store call. It trusts the rows once their backfill has finished, and until
then finds its roots on the file store as before.

A row is written wherever the server sees folders in a root: a turn end that
created or listed folders in its project's shared root
(task_objects.record_new_artifacts), the self-heal's full listing of that
root (artifact_autopublish._record_listed_locations), a task's move into a
project (TaskObjectService._relocate_artifacts), and the one-time backfill
below, which walks every project's shared root and conversation workspaces.
Nothing writes artifacts into a conversation workspace any more
(artifact_roots.conversation_artifacts_base), so only the backfill writes a
legacy root's row. Rows are never removed, except with their project or
conversation: a root emptied later only costs the list one open.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Collection
from dataclasses import dataclass
from typing import NamedTuple
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from cowork.common.logger import log_context
from cowork.common.paths import dir_scandir
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.models.held_artifact_root import HeldArtifactRoot
from cowork.models.project import Project
from cowork.models.setting import Setting
from cowork.services.artifact_identity import _strictly_holds_no_id, opened_artifact_root
from cowork.services.artifact_locations import keeps_locations
from cowork.services.artifact_owner_backfill import advisory_try_lock, sentinel_present, write_sentinel
from cowork.services.artifacts import ProjectArtifacts
from cowork.services.shared_resources import advisory_lock_key

logger = logging.getLogger(__name__)

#: Written once a backfill pass has read every project; the list trusts the
#: rows from then on. The revision that made the table deletes it on a
#: downgrade (7e636946b5f8_held_artifact_roots.py), which names it as a literal.
SENTINEL_KEY = "_held_artifact_roots_backfill_v1"
_LOCK_KEY = advisory_lock_key(b"cowork\0held_artifact_roots_backfill_v1")


class HeldRoot(NamedTuple):
    """One root a row names: a project's shared root when `conversation_id`
    is None, else that conversation's legacy root."""

    project_id: UUID
    conversation_id: UUID | None


def hold_root(session: ScopedSession, *, project_id, conversation_id=None) -> None:
    """Record that a root of `project_id` holds an artifact folder: its
    shared root, or `conversation_id`'s legacy root. A root already recorded
    is left as it is, and so is one whose project or conversation another
    writer deleted meanwhile."""
    if not keeps_locations(session.scope):
        return
    project_id = UUID(str(project_id))
    conversation_id = UUID(str(conversation_id)) if conversation_id is not None else None
    query = session.select(HeldArtifactRoot).where(HeldArtifactRoot.project_id == project_id)
    if conversation_id is None:
        query = query.where(HeldArtifactRoot.conversation_id.is_(None))
    else:
        query = query.where(HeldArtifactRoot.conversation_id == conversation_id)
    if session.exec(query).first() is not None:
        return
    session.add(HeldArtifactRoot(project_id=project_id, conversation_id=conversation_id))
    try:
        session.commit()
    except IntegrityError:
        # Another writer recorded the root first, or its project or
        # conversation is gone.
        session.rollback()


def _backfill_finished(session: ScopedSession) -> bool:
    """Whether the backfill wrote its sentinel, the global settings row
    SENTINEL_KEY."""
    query = session.select(Setting).where(Setting.key == SENTINEL_KEY, Setting.scope.is_(None))
    return session.exec(query).first() is not None


def held_roots(session: ScopedSession, *, project_ids: Collection[UUID]) -> frozenset[HeldRoot] | None:
    """The roots of `project_ids` the rows name, or None when the caller has
    to find its roots on the file store: this scope keeps no rows, or their
    backfill has not finished."""
    if not keeps_locations(session.scope) or not _backfill_finished(session):
        return None
    ids = list(project_ids)
    if not ids:
        return frozenset()
    rows = session.exec(session.select(HeldArtifactRoot).where(HeldArtifactRoot.project_id.in_(ids))).all()
    return frozenset(HeldRoot(row.project_id, row.conversation_id) for row in rows)


def holds_a_folder(source: ProjectArtifacts) -> bool:
    """Whether `source`'s root holds an entry the artifacts list would open:
    a folder, not a link, whose name does not start with a dot
    (artifacts.list_artifacts). A root that is missing, a link, refused by
    its anchor, or one the server may not read holds none, since the list
    could show nothing from it. Any other error raises, so a root the walk
    could not read is never taken for an empty one."""
    try:
        with opened_artifact_root(source) as root:
            with dir_scandir(root) as entries:
                for entry in entries:
                    if entry.name.startswith("."):
                        continue
                    try:
                        if not entry.is_symlink() and entry.is_dir(follow_symlinks=False):
                            return True
                    except OSError as exc:
                        if not _strictly_holds_no_id(exc=exc):
                            raise
    except ValueError:
        return False
    except OSError as exc:
        if _strictly_holds_no_id(exc=exc):
            return False
        raise
    return False


@dataclass
class HeldRootBackfillSummary:
    #: Roots found holding a folder, and given a row (or already holding one).
    held: int = 0
    #: Projects whose roots could not all be read; the sentinel waits for them.
    unread: int = 0


class _Stopped(Exception):
    """``stop`` was set during the walk."""


def _check_stop(stop: threading.Event | None) -> None:
    if stop is not None and stop.is_set():
        raise _Stopped


def _held_in(project, *, stop: threading.Event | None) -> list[HeldRoot]:
    """The roots of `project` that hold a folder: its shared root and every
    member's conversation workspaces. Raises OSError when a root cannot be
    read."""
    from cowork.services.artifact_roots import project_roots

    found: list[HeldRoot] = []
    for root in project_roots(project, strict=True):
        _check_stop(stop)
        if root.legacy and root.conversation_id is None:
            # A workspace whose name is no conversation id has no owner, so
            # no list ever reads it.
            continue
        if holds_a_folder(root.source):
            found.append(HeldRoot(project.id, root.conversation_id if root.legacy else None))
    return found


def _record(engine, *, org_id: str, roots: list[HeldRoot]) -> None:
    with Session(engine) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=org_id))
        for root in roots:
            hold_root(session, project_id=root.project_id, conversation_id=root.conversation_id)


def run_held_root_backfill(
    engine=None, *, try_lock=None, project_ids=None, stop: threading.Event | None = None
) -> HeldRootBackfillSummary | None:
    """Walk every org project's roots once per environment, record each root
    that holds a folder, then write the sentinel. None when skipped or
    stopped.

    One replica runs it at a time (an advisory lock). A stop, a restart or a
    project it could not read leaves the sentinel unwritten, so the next
    start walks every project again; recording a root again changes nothing.
    Raises when a database write fails for any reason but a row another
    writer recorded first. ``project_ids`` limits the pass to those projects;
    it exists for tests, whose database is shared across modules. Production
    passes nothing.
    """
    from cowork.services.artifact_roots import ProjectRow, left_out_for_another_partition
    from cowork.services.projects import resolved_shared_root

    if get_app_settings().tenancy_mode != "org":
        return None
    engine = engine or get_engine(get_app_settings().database.uri)
    lock = try_lock or advisory_try_lock
    if sentinel_present(engine, SENTINEL_KEY):
        return None
    with lock(engine, _LOCK_KEY) as acquired:
        if not acquired or sentinel_present(engine, SENTINEL_KEY):
            return None
        with Session(engine) as raw:
            query = select(Project.id, Project.name, Project.path, Project.org_id).order_by(Project.id)
            if project_ids is not None:
                query = query.where(Project.id.in_(list(project_ids)))
            projects = raw.exec(query).all()
        summary = HeldRootBackfillSummary()
        # Read once for every project the pass walks, not once per project.
        shared_root = resolved_shared_root()
        try:
            for project_id, name, path, org_id in projects:
                _check_stop(stop)
                if not org_id:
                    continue
                project = ProjectRow(id=UUID(str(project_id)), name=name, path=path, external=False)
                # In org tenancy an organization's partition of the shared
                # root is its id (projects.partition_of).
                if left_out_for_another_partition(project, partition=str(org_id), shared_root=shared_root):
                    continue
                try:
                    roots = _held_in(project, stop=stop)
                except OSError as exc:
                    summary.unread += 1
                    logger.warning(
                        "held_artifact_roots backfill could not read a project; the next start walks again",
                        extra={
                            **log_context(project_id=project.id),
                            "event": "held_artifact_roots_backfill_unread",
                            "error_type": type(exc).__name__,
                        },
                    )
                    continue
                _record(engine, org_id=str(org_id), roots=roots)
                summary.held += len(roots)
        except _Stopped:
            logger.warning(
                "held_artifact_roots backfill stopped before it finished; the next start walks again",
                extra={"event": "held_artifact_roots_backfill_stopped"},
            )
            return None
        for outcome, count in (("held", summary.held), ("unread", summary.unread)):
            logger.warning(
                "held_artifact_roots backfill finished its walk",
                extra={"event": "held_artifact_roots_backfill_summary", "outcome": outcome, "count": count},
            )
        if not summary.unread:
            write_sentinel(engine, SENTINEL_KEY)
    return summary
