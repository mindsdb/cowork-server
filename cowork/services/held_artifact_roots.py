"""The held_artifact_roots rows (cowork/models/held_artifact_root.py): which
artifact roots have held an artifact folder.

Org tenancy only: there `COWORK_HOME` is the shared file store, where every
open and stat is a round trip to the file server. The artifacts list opens
only the roots these rows name (artifact_roots.with_held_roots), so a project
or a conversation workspace that never held an artifact costs it no file
store call. It trusts the rows once their backfill has walked every project,
and until then finds its roots on the file store as before. It still finds
there the roots of each project the backfill could not read, which its
sentinel lists (HeldRootSentinel).

A row is written wherever the server sees or writes folders in a root: a
turn end that created or listed folders in its project's shared root
(task_objects.record_new_artifacts), the self-heal's full listing of that
root (artifact_autopublish._record_listed_locations), which the heal sweep
also runs for every project with a recent message (artifact_heal_sweep), a
task's move into a project (TaskObjectService._relocate_artifacts), the
project files PUT that writes into an artifact folder of a project's shared
root or of the caller's own conversation workspace
(project_files._write_project_file), a comparison side continued into a
project, which brings its whole workspace along, `.anton` included, and
holds that workspace's legacy root when it holds a folder
(ComparisonService._hold_carried_workspace), and the one-time backfill
below, which walks every project's shared root and conversation workspaces.
A pod's artifacts land in its project's shared root, not in its workspace
(artifact_roots.CONVERSATIONS_DIRNAME), so after the backfill the PUT and
the comparison carry are the writers that record a legacy root. Rows are
never removed, except with their project or conversation: a root emptied
later only costs the list one open.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
from uuid import UUID

from pydantic import BaseModel, ValidationError
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from cowork.common.logger import log_context
from cowork.common.paths import dir_scandir
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.models.held_artifact_root import HeldArtifactRoot
from cowork.models.project import Project
from cowork.models.setting import Setting
from cowork.services.artifact_identity import _strictly_holds_no_id, opened_artifact_root
from cowork.services.artifact_locations import keeps_locations
from cowork.services.artifact_owner_backfill import (
    BackfillProgress,
    BackfillStopped as _Stopped,
    advisory_try_lock,
    check_stop as _check_stop,
    delete_progress,
    read_progress,
    save_progress,
    write_sentinel,
)
from cowork.services.artifacts import ProjectArtifacts
from cowork.services.shared_resources import advisory_lock_key

if TYPE_CHECKING:
    from cowork.services.artifact_roots import ProjectRow

logger = logging.getLogger(__name__)

#: Written once a backfill pass has walked every project, holding the
#: projects it could not read (HeldRootSentinel); the list trusts the rows
#: for every other project from then on. The revision that made the table
#: deletes it on a downgrade (7e636946b5f8_held_artifact_roots.py), which
#: names it as a literal.
SENTINEL_KEY = "_held_artifact_roots_backfill_v2"
#: The first pass's progress (BackfillProgress), kept until it writes the
#: sentinel, so a stop or a restart goes on from the last project it
#: finished. The same downgrade deletes it.
PROGRESS_KEY = "_held_artifact_roots_backfill_v2_progress"
_LOCK_KEY = advisory_lock_key(b"cowork\0held_artifact_roots_backfill_v1")


class HeldRoot(NamedTuple):
    """One root a row names: a project's shared root when `conversation_id`
    is None, else that conversation's legacy root."""

    project_id: UUID
    conversation_id: UUID | None


def hold_root(session: ScopedSession, *, project_id: UUID, conversation_id: UUID | None = None) -> None:
    """Record that a root of `project_id` holds an artifact folder: its
    shared root, or `conversation_id`'s legacy root. A root already recorded
    is left as it is, and so is one whose project or conversation another
    writer deleted meanwhile."""
    if not keeps_locations(session.scope):
        return
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


def sentinel_written(session: ScopedSession, *, key: str = SENTINEL_KEY) -> bool:
    """Whether a backfill wrote its sentinel, the global settings row `key`:
    this module's by default."""
    query = session.select(Setting).where(Setting.key == key, Setting.scope.is_(None))
    return session.exec(query).first() is not None


def held_roots(session: ScopedSession, *, project_ids: Collection[UUID]) -> frozenset[HeldRoot] | None:
    """The roots of `project_ids` the rows name, or None when the caller has
    to find its roots on the file store: this scope keeps no rows, or their
    backfill has not finished."""
    if not keeps_locations(session.scope) or not sentinel_written(session):
        return None
    ids = list(project_ids)
    if not ids:
        return frozenset()
    from cowork.services.conversations import ConversationService

    # Every shared root's row, and a legacy root's only for a conversation
    # the caller owns, by the rule the plan's owner filter keeps
    # (ConversationService.owned_ids), so no caller reads another member's
    # legacy rows from here.
    owned = ConversationService(session).owned_ids_query().scalar_subquery()
    query = session.select(HeldArtifactRoot).where(
        HeldArtifactRoot.project_id.in_(ids),
        or_(HeldArtifactRoot.conversation_id.is_(None), HeldArtifactRoot.conversation_id.in_(owned)),
    )
    return frozenset(HeldRoot(row.project_id, row.conversation_id) for row in session.exec(query).all())


class HeldRootSentinel(BaseModel):
    """The sentinel's value, as JSON in the settings row SENTINEL_KEY."""

    #: Projects whose roots the pass could not read, so they have no rows
    #: from it: the list finds their roots on the file store, and each start
    #: walks them again and rewrites this list.
    unread: list[UUID] = []


def read_sentinel(session: ScopedSession) -> HeldRootSentinel | None:
    """The backfill's sentinel, or None before a pass has walked every
    project. A value that is not a HeldRootSentinel, as one written by hand,
    also reads as None: the list then finds its roots on the file store
    instead of failing, and the next pass rewrites the value."""
    query = session.select(Setting).where(Setting.key == SENTINEL_KEY, Setting.scope.is_(None))
    row = session.exec(query).first()
    if row is None:
        return None
    try:
        return HeldRootSentinel.model_validate_json(row.value)
    except ValidationError:
        logger.warning(
            "held_artifact_roots sentinel holds a value it cannot read; the list finds its roots on the file store",
            extra={"event": "held_artifact_roots_sentinel_unreadable"},
        )
        return None


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
    #: Projects whose roots could not all be read; the sentinel lists them.
    unread: int = 0


def _held_in(project: ProjectRow, *, stop: threading.Event | None) -> list[HeldRoot]:
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


def _walk_project(
    engine,
    *,
    project: ProjectRow,
    org_id: str | None,
    shared_root: Path,
    summary: HeldRootBackfillSummary,
    stop: threading.Event | None,
) -> bool:
    """Record the held roots of one project. False when its roots could not
    all be read, so it got no rows. `shared_root` is the resolved shared root
    (projects.resolved_shared_root), read once for the pass."""
    from cowork.services.artifact_roots import left_out_for_another_partition

    if not org_id:
        return True
    # In org tenancy an organization's partition of the shared root is its id
    # (projects.partition_of).
    if left_out_for_another_partition(project, partition=str(org_id), shared_root=shared_root):
        return True
    try:
        roots = _held_in(project, stop=stop)
    except OSError as exc:
        summary.unread += 1
        logger.warning(
            "held_artifact_roots backfill could not read a project; the list walks it, and the next start "
            "reads it again",
            extra={
                **log_context(project_id=project.id),
                "event": "held_artifact_roots_backfill_unread",
                "error_type": type(exc).__name__,
            },
        )
        return False
    _record(engine, org_id=str(org_id), roots=roots)
    summary.held += len(roots)
    return True


def _read_sentinel(engine) -> HeldRootSentinel | None:
    with Session(engine) as raw:
        return read_sentinel(ScopedSession(raw, SYSTEM_SCOPE))


def _skipped(reason: str) -> None:
    """Log why a pass walks nothing: `not_org`, `finished` (the sentinel
    lists no unread project) or `locked` (another replica holds the walk).
    WARNING, as the start line below and the summary lines: every hosted
    deployment runs at LOG_LEVEL=WARNING, where an INFO line never shows, so
    an operator could not tell a walk that never started from one that
    runs. One line per pass."""
    logger.warning(
        "held_artifact_roots backfill skipped its walk",
        extra={"event": "held_artifact_roots_backfill_skipped", "reason": reason},
    )


def run_held_root_backfill(
    engine=None, *, try_lock=None, project_ids=None, stop: threading.Event | None = None
) -> HeldRootBackfillSummary | None:
    """Walk every org project's roots once per environment, record each root
    that holds a folder, then write the sentinel with the projects it could
    not read. None when skipped or stopped.

    One replica runs it at a time (an advisory lock). The first pass walks
    the projects in id order and keeps the last one it finished, and each one
    it could not read, in the PROGRESS_KEY setting row, so a stop or a
    restart costs only the project it was in. Once it has walked the last
    project it deletes the progress row and writes the sentinel, listing the
    projects it could not read (HeldRootSentinel). A later start that finds
    that list walks only those projects and rewrites it, to an empty list
    once all of them read; one that finds it empty walks nothing. Recording a
    root again changes nothing. Raises when a database write fails for any
    reason but a row another writer recorded first. ``project_ids`` limits
    the pass to those projects; it exists for tests, whose database is shared
    across modules. Production passes nothing.
    """
    from cowork.services.artifact_roots import ProjectRow
    from cowork.services.projects import resolved_shared_root

    if get_app_settings().tenancy_mode != "org":
        _skipped("not_org")
        return None
    engine = engine or get_engine(get_app_settings().database.uri)
    lock = try_lock or advisory_try_lock
    sentinel = _read_sentinel(engine)
    if sentinel is not None and not sentinel.unread:
        _skipped("finished")
        return None
    with lock(engine, _LOCK_KEY) as acquired:
        if not acquired:
            _skipped("locked")
            return None
        sentinel = _read_sentinel(engine)
        if sentinel is not None and not sentinel.unread:
            _skipped("finished")
            return None
        query = select(Project.id, Project.name, Project.path, Project.org_id).order_by(Project.id)
        if sentinel is not None:
            # Once the sentinel exists, the pass walks only the projects it
            # lists, and keeps no progress: the list is short.
            reason = "unread"
            progress = BackfillProgress()
            query = query.where(Project.id.in_(sentinel.unread))
        else:
            reason = "first"
            progress = read_progress(engine, PROGRESS_KEY)
            if progress.after is not None:
                reason = "resumed"
                query = query.where(or_(Project.id > progress.after, Project.id.in_(progress.unread)))
        if project_ids is not None:
            query = query.where(Project.id.in_(list(project_ids)))
        with Session(engine) as raw:
            projects = raw.exec(query).all()
        progress = progress.within({UUID(str(row[0])) for row in projects})
        logger.warning(
            "held_artifact_roots backfill starts its walk",
            extra={"event": "held_artifact_roots_backfill_started", "reason": reason, "count": len(projects)},
        )
        summary = HeldRootBackfillSummary()
        # Read once for every project the pass walks, not once per project.
        shared_root = resolved_shared_root()
        try:
            for project_id, name, path, org_id in projects:
                _check_stop(stop)
                project = ProjectRow(id=UUID(str(project_id)), name=name, path=path, external=False)
                read = _walk_project(
                    engine, project=project, org_id=org_id, shared_root=shared_root, summary=summary, stop=stop
                )
                progress = progress.finished(project_id=project.id, read=read)
                if sentinel is None:
                    save_progress(engine, PROGRESS_KEY, progress=progress)
        except _Stopped:
            logger.warning(
                "held_artifact_roots backfill stopped before it finished; the next start resumes it",
                extra={"event": "held_artifact_roots_backfill_stopped"},
            )
            return None
        for outcome, count in (("held", summary.held), ("unread", summary.unread)):
            logger.warning(
                "held_artifact_roots backfill finished its walk",
                extra={"event": "held_artifact_roots_backfill_summary", "outcome": outcome, "count": count},
            )
        if sentinel is None:
            # Progress first, as the location backfill does: a crash between
            # the two leaves neither, and the next start walks every project
            # again, rather than a progress row that would skip projects once
            # the sentinel is deleted for a rerun.
            delete_progress(engine, PROGRESS_KEY)
        write_sentinel(engine, SENTINEL_KEY, value=HeldRootSentinel(unread=progress.unread).model_dump_json())
    return summary
