"""The held_artifact_roots rows (cowork/models/held_artifact_root.py): which
artifact roots have held an artifact folder.

Org tenancy only: there `COWORK_HOME` is the shared file store, where every
open and stat is a round trip to the file server. The artifacts list opens
only the roots these rows name (artifact_roots.with_held_roots), so a project
or a conversation workspace that never held an artifact costs it no file
store call. It trusts the rows once their backfill has walked every project,
and until then finds its roots on the file store as before. It still finds
there the roots of each project the backfill could not read, or that moved
or went away during its walk, which its sentinel lists (HeldRootSentinel).

A row is written wherever the server sees or writes folders in a root: a
turn end that created, listed or timed folders in its project's shared
root, or could not read a folder's time there
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
from collections.abc import Collection, Iterator
from contextlib import closing
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from cowork.common.logger import log_context
from cowork.common.paths import PinnedDir, dir_scandir
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.models.conversation import Conversation
from cowork.models.held_artifact_root import HeldArtifactRoot
from cowork.models.project import Project
from cowork.services.artifact_identity import _strictly_holds_no_id, is_listable_folder_name, opened_artifact_root
from cowork.services.artifact_locations import keeps_locations
from cowork.services.artifact_owner_backfill import (
    BackfillProgress,
    BackfillStopped,
    advisory_try_lock,
    check_stop,
    delete_progress,
    log_skipped,
    read_progress,
    read_sentinel_value,
    save_progress,
    walked_folder_is_gone,
    walked_path_is_current,
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


def lock_root_parents(session: ScopedSession, *, project_id: UUID, conversation_id: UUID | None) -> None:
    """Take FOR KEY SHARE, until the caller commits, on the rows a root's
    rows point at: `conversation_id`'s row for a legacy root, then
    `project_id`'s. A writer of a root's rows (hold_root, and every
    artifact_sort_times writer) takes them before it locks or adds a row,
    in the order every delete takes them: ProjectService.delete_project
    deletes the project's conversations before the project, and a
    conversation delete (ConversationService.delete_conversation) deletes
    the conversation alone, each with ON DELETE CASCADE into the root's
    rows. A writer then waits behind a running delete, or the delete waits
    behind the writer, and neither holds a row the other needs. An insert's
    foreign key checks take the same locks, but only after the row is
    written, and in either order. A parent that is gone locks nothing, and
    the write fails on its foreign key. SQLite ignores the lock."""
    if conversation_id is not None:
        session.exec(
            session.select(Conversation)
            .with_only_columns(Conversation.id)
            .where(Conversation.id == conversation_id)
            .with_for_update(read=True, key_share=True)
        ).all()
    session.exec(
        session.select(Project)
        .with_only_columns(Project.id)
        .where(Project.id == project_id)
        .with_for_update(read=True, key_share=True)
    ).all()


def hold_root(session: ScopedSession, *, project_id: UUID, conversation_id: UUID | None = None) -> None:
    """Record that a root of `project_id` holds an artifact folder: its
    shared root, or `conversation_id`'s legacy root. A root already recorded
    is left as it is, and so is one whose project or conversation another
    writer deleted meanwhile. A new row's parents are locked first
    (lock_root_parents), so the insert never deadlocks a delete."""
    if not keeps_locations(session.scope):
        return
    query = session.select(HeldArtifactRoot).where(HeldArtifactRoot.project_id == project_id)
    if conversation_id is None:
        query = query.where(HeldArtifactRoot.conversation_id.is_(None))
    else:
        query = query.where(HeldArtifactRoot.conversation_id == conversation_id)
    if session.exec(query).first() is not None:
        return
    lock_root_parents(session, project_id=project_id, conversation_id=conversation_id)
    session.add(HeldArtifactRoot(project_id=project_id, conversation_id=conversation_id))
    try:
        session.commit()
    except IntegrityError:
        # Another writer recorded the root first, or its project or
        # conversation is gone.
        session.rollback()


def held_root_rows(session: ScopedSession, *, project_ids: Collection[UUID]) -> frozenset[HeldRoot]:
    """The roots of `project_ids` the rows name. It reads no sentinel: its
    caller, artifact_roots.with_held_roots, trusts the rows only in a scope
    that keeps them and once read_sentinel answers, which it reads once per
    list."""
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

    #: Projects whose roots the pass could not read, or that moved or went
    #: away during their walk, so they have no rows from it: the list finds
    #: their roots on the file store, and each later pass walks them again
    #: and rewrites this list. The startup task runs one every
    #: ARTIFACT_BACKFILL_UNREAD_RETRY_SECONDS while this list is not empty
    #: (server._run_artifact_backfills).
    unread: list[UUID] = []


def read_sentinel(session: ScopedSession) -> HeldRootSentinel | None:
    """The backfill's sentinel, or None before a pass has walked every
    project. A value that is not a HeldRootSentinel, as one written by hand,
    also reads as None (read_sentinel_value): the list then finds its roots
    on the file store instead of failing, and the next pass rewrites the
    value."""
    return read_sentinel_value(
        session,
        key=SENTINEL_KEY,
        model=HeldRootSentinel,
        logger=logger,
        unreadable_event="held_artifact_roots_sentinel_unreadable",
        unreadable_message=(
            "held_artifact_roots sentinel holds a value it cannot read; the list finds its roots on the file store"
        ),
    )


def listable_folders(root: PinnedDir) -> Iterator[str]:
    """The name of each entry of `root`, an opened artifacts root, that the
    artifacts list would open: a folder, not a link, whose name it would open
    (artifact_identity.is_listable_folder_name). An entry whose type cannot
    be read is skipped when the error says the list could show nothing from
    it (_strictly_holds_no_id); any other error raises."""
    with dir_scandir(root) as entries:
        for entry in entries:
            if not is_listable_folder_name(entry.name):
                continue
            try:
                if not entry.is_symlink() and entry.is_dir(follow_symlinks=False):
                    yield entry.name
            except OSError as exc:
                if not _strictly_holds_no_id(exc=exc):
                    raise


def _listed_folder_names(source: ProjectArtifacts, *, limit: int | None) -> list[str]:
    """The first `limit` names listable_folders yields for `source`'s root,
    or every name when `limit` is None, read in one scandir that stops
    there. A root that is missing, a link, refused by its anchor, or one the
    server may not read holds none, since the list could show nothing from
    it. Any other error raises, so a root the server could not read is never
    taken for an empty one."""
    try:
        with opened_artifact_root(source) as root, closing(listable_folders(root)) as names:
            return list(islice(names, limit))
    except ValueError:
        return []
    except OSError as exc:
        if _strictly_holds_no_id(exc=exc):
            return []
        raise


def listable_folder_names(source: ProjectArtifacts) -> list[str]:
    """The name of each entry of `source`'s root the artifacts list would
    open (listable_folders), by the rules of _listed_folder_names."""
    return _listed_folder_names(source, limit=None)


def holds_a_folder(source: ProjectArtifacts) -> bool:
    """Whether `source`'s root holds an entry the artifacts list would open,
    by the rules of _listed_folder_names. The read stops at the first
    folder, so an entry after it is never read: one whose type read would
    fail, as on EIO, does not make the root unread."""
    return bool(_listed_folder_names(source, limit=1))


@dataclass
class HeldRootBackfillSummary:
    #: Roots found holding a folder, and given a row (or already holding one).
    held: int = 0
    #: Projects whose roots could not all be read, or that moved or went away
    #: during their walk; the sentinel lists them.
    unread: int = 0


def _held_in(project: ProjectRow, *, stop: threading.Event | None) -> list[HeldRoot]:
    """The roots of `project` that hold a folder: its shared root and every
    member's conversation workspaces. Raises OSError when a root cannot be
    read."""
    from cowork.services.artifact_roots import project_roots

    found: list[HeldRoot] = []
    for root in project_roots(project, strict=True):
        check_stop(stop)
        if root.legacy and root.conversation_id is None:
            # A workspace whose name is no conversation id has no owner, so
            # no list ever reads it.
            continue
        if holds_a_folder(root.source):
            found.append(HeldRoot(project.id, root.conversation_id if root.legacy else None))
    return found


def _record(engine, *, project: ProjectRow, org_id: str, roots: list[HeldRoot]) -> bool:
    """Hold `roots`, found by a walk of `project`'s path, and True; or hold
    none and False when the project's row no longer holds that path
    (walked_path_is_current), since the walk may then have missed its roots
    or read another project's."""
    with Session(engine) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=org_id))
        if not walked_path_is_current(session, project=project):
            return False
        for root in roots:
            hold_root(session, project_id=root.project_id, conversation_id=root.conversation_id)
    return True


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
    all be read, or when the project moved or went away during its walk
    (walked_folder_is_gone, _record), so it got no rows. `shared_root` is
    the resolved shared root (projects.resolved_shared_root), read once for
    the pass."""
    from cowork.services.artifact_roots import left_out_for_another_partition

    if not org_id:
        return True
    # In org tenancy an organization's partition of the shared root is its id
    # (projects.partition_of).
    if left_out_for_another_partition(project, partition=str(org_id), shared_root=shared_root):
        return True
    try:
        # A folder there before the walk and gone after it moved during the
        # walk, as a rename moves it before it commits. One already gone
        # before the walk is taken for a project with no folder, which holds
        # no root, so this pass, which walks every project, never lists such
        # a project as unread on every pass.
        there = not walked_folder_is_gone(project)
        roots = _held_in(project, stop=stop)
        gone = there and walked_folder_is_gone(project)
    except OSError as exc:
        summary.unread += 1
        logger.warning(
            "held_artifact_roots backfill could not read a project; the list walks it, and the next pass "
            "reads it again",
            extra={
                **log_context(project_id=project.id),
                "event": "held_artifact_roots_backfill_unread",
                "error_type": type(exc).__name__,
            },
        )
        return False
    if gone or not _record(engine, project=project, org_id=str(org_id), roots=roots):
        summary.unread += 1
        logger.warning(
            "held_artifact_roots backfill walked a project that moved or went away meanwhile; the list walks it, "
            "and the next pass reads it again",
            extra={**log_context(project_id=project.id), "event": "held_artifact_roots_backfill_moved"},
        )
        return False
    summary.held += len(roots)
    return True


def _read_sentinel(engine) -> HeldRootSentinel | None:
    with Session(engine) as raw:
        return read_sentinel(ScopedSession(raw, SYSTEM_SCOPE))


def run_held_root_backfill(
    engine=None, *, try_lock=None, project_ids=None, stop: threading.Event | None = None
) -> HeldRootBackfillSummary | None:
    """Walk every org project's roots once per environment, record each root
    that holds a folder, then write the sentinel with the projects it could
    not read, or that moved or went away during their walk. None when
    skipped or stopped.

    One replica runs it at a time (an advisory lock). The first pass walks
    the projects in id order and keeps the last one it finished, and each one
    it could not read, in the PROGRESS_KEY setting row, so a stop or a
    restart costs only the project it was in. Once it has walked the last
    project it deletes the progress row and writes the sentinel, listing the
    projects it could not read (HeldRootSentinel). A later pass that finds
    that list, at a start or the startup task's next run, walks only those
    projects and rewrites it, to an empty list once all of them read; one
    that finds it empty walks nothing. Recording a
    root again changes nothing. Raises when a database write fails for any
    reason but a row another writer recorded first. ``project_ids`` limits
    the pass to those projects; it exists for tests, whose database is shared
    across modules. Production passes nothing.
    """
    from cowork.services.artifact_roots import ProjectRow
    from cowork.services.projects import resolved_shared_root

    if get_app_settings().tenancy_mode != "org":
        log_skipped(logger, event_prefix="held_artifact_roots", reason="not_org")
        return None
    engine = engine or get_engine(get_app_settings().database.uri)
    lock = try_lock or advisory_try_lock
    sentinel = _read_sentinel(engine)
    if sentinel is not None and not sentinel.unread:
        log_skipped(logger, event_prefix="held_artifact_roots", reason="finished")
        return None
    with lock(engine, _LOCK_KEY) as acquired:
        if not acquired:
            log_skipped(logger, event_prefix="held_artifact_roots", reason="locked")
            return None
        sentinel = _read_sentinel(engine)
        if sentinel is not None and not sentinel.unread:
            log_skipped(logger, event_prefix="held_artifact_roots", reason="finished")
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
                check_stop(stop)
                project = ProjectRow(id=UUID(str(project_id)), name=name, path=path, external=False)
                read = _walk_project(
                    engine, project=project, org_id=org_id, shared_root=shared_root, summary=summary, stop=stop
                )
                progress = progress.finished(project_id=project.id, read=read)
                if sentinel is None:
                    save_progress(engine, PROGRESS_KEY, progress=progress)
        except BackfillStopped:
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
