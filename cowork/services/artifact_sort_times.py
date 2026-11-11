"""The artifact_sort_times rows (cowork/models/artifact_sort_time.py): the time
the artifacts list sorts each artifact folder by.

Org tenancy only. With the held-root rows (held_artifact_roots), they let the
list choose its newest folders in the database and open only those
(newest_folders), instead of reading every folder's `metadata.json` in the
roots it opens. newest_folders answers once the rows' backfill has written
its sentinel (read_sort_sentinel), and leaves out the roots of the projects
that sentinel lists (SortTimeSentinel.unread), whose every folder the list
reads; until then it answers None and the list reads every folder.

A row holds the newest modification time of the folder's `metadata.json`
the server has read: a writer that read a time no newer than the row's
leaves the row as it is (record_sort_times). The times come from the turn
end that created, changed or named the folder
(task_objects.record_new_artifacts, with the times turn_artifact_changes
read), from the self-heal's full listing of a shared root, the heal sweep's
included, which writes every time it read and drops the rows of folders it
found gone or holding no id (artifact_autopublish._record_listing,
reconcile_sort_times), from the project files PUT
(project_files._record_artifact_folder_write), from a comparison side's
carry (ComparisonService._hold_carried_workspace), and from the backfill
below. Only the self-heal's listing moves a row back, to the older time a
folder replaced by an older copy now holds, and only a row no writer
changed while it listed. A rewrite of `metadata.json` the server makes on
its own, such as an id widening (artifact_identity.ensure_full_id), puts
the file's time back, so an old artifact does not move up the list. A row
moves with a task's folders (move_sort_times) and goes with a folder the
artifact delete removes (forget_sort_time). The project files DELETE of an
artifact folder's `metadata.json`, in a shared root or the caller's own
legacy root, drops the folder's row too, then reads the folder's time
again and records it, so a PUT that wrote the file back meanwhile keeps its
row (project_files._forget_artifact_folder_delete).
"""
from __future__ import annotations

import logging
import stat
import threading
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import and_, or_, tuple_
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from cowork.common.logger import log_context
from cowork.common.paths import dir_child_stat
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.models.artifact_sort_time import ArtifactSortTime
from cowork.models.held_artifact_root import HeldArtifactRoot
from cowork.models.project import Project
from cowork.services.artifact_identity import (
    _opened_from_trusted_root,
    _strictly_holds_no_id,
    is_listable_folder_name,
    opened_artifact_root,
)
from cowork.services.artifact_locations import Location, keeps_locations
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
from cowork.services.held_artifact_roots import (
    HeldRoot,
    HeldRootSentinel,
    listable_folders,
    lock_root_parents,
    read_sentinel,
)
from cowork.services.shared_resources import advisory_lock_key

if TYPE_CHECKING:
    from cowork.services.artifact_roots import ProjectRow

logger = logging.getLogger(__name__)

#: Written once a backfill pass has walked every project with a held root,
#: holding the projects that have no rows from it (SortTimeSentinel); the
#: list trusts the rows for every other project from then on. The revision
#: that made the table deletes it on a downgrade
#: (33c9ba9927ad_artifact_sort_times.py), which names it as a literal.
SENTINEL_KEY = "_artifact_sort_times_backfill_v1"
#: The first pass's progress (BackfillProgress), kept until it writes the
#: sentinel, so a stop or a restart goes on from the last project it
#: finished. The same downgrade deletes it.
PROGRESS_KEY = "_artifact_sort_times_backfill_v1_progress"
_LOCK_KEY = advisory_lock_key(b"cowork\0artifact_sort_times_backfill_v1")


class SortTime(NamedTuple):
    """One row: a folder of a root (its project, and its conversation for a
    legacy root) and the time the list sorts it by."""

    project_id: UUID
    conversation_id: UUID | None
    slug: str
    sort_time: float


class NewestFolders(NamedTuple):
    """The newest rows of the roots a list may open, newest first, and
    whether they are all of the rows (none was cut at the query's limit)."""

    rows: tuple[SortTime, ...]
    complete: bool
    #: The projects whose every folder a list reads, and whose roots `rows`
    #: leaves out. newest_folders names those among the roots asked for
    #: that the backfill's sentinel lists (SortTimeSentinel.unread), which
    #: may have no rows. The artifacts list adds the projects the held-root
    #: sentinel lists (RootsPlan.walk), whose held rows may not name every
    #: root, and which it asks no rows for.
    unread: frozenset[UUID]
    #: When the limit cut a run of rows that share the time of the last of
    #: `rows`, the rest of that run, in the query's order (slug, then id):
    #: its rows less those of the folders `rows` holds. Empty when no run
    #: was cut, and when the run holds more than _TIED_ROWS rows
    #: (`ties_read` is then false).
    tied: tuple[SortTime, ...]
    #: Whether `rows` and `tied` hold every row at the time of the last of
    #: `rows`, so that every row they leave out is older than that. False
    #: only when the limit cut a run longer than _TIED_ROWS.
    ties_read: bool


class SortTimeRow(NamedTuple):
    """A row as a self-heal read it before its listing (sort_time_rows)."""

    id: UUID
    sort_time: float


class SortTimesRead(NamedTuple):
    """What sort_times_below read below one root. A tuple of two fields, so
    always true: test `times` or `unreadable`, never the value itself."""

    #: The time of each folder whose `metadata.json` it read.
    times: dict[str, float]
    #: The folders whose read failed with an error that leaves unknown what
    #: they hold, such as EIO: every folder asked for when the root itself
    #: failed that way. A caller never takes one for a folder with no time.
    unreadable: frozenset[str]


def _folder_time(root, name: str) -> float | None:
    """The modification time of `name`'s `metadata.json`, `name` a direct
    child folder of the pinned `root`, opened without following a link in its
    place (dir_child_stat). None when `metadata.json` is not a regular file:
    the list shows no card for such a folder."""
    found = dir_child_stat(root, name, "metadata.json")
    return found.st_mtime if stat.S_ISREG(found.st_mode) else None


def sort_times_below(*, base: Path, slugs: Iterable[str]) -> SortTimesRead:
    """The time of each of `slugs` that is an artifact folder directly below
    `base`, an artifacts root (a project's shared root or a conversation's
    legacy root) opened from the trusted root, among the names the list
    would open (artifact_identity.is_listable_folder_name).
    A name that is gone, a link, not a folder, holds no plain `metadata.json`,
    or one the server may not read gets no time, since the list could show no
    card for it (_strictly_holds_no_id); any other read error puts it in
    `unreadable`. Opens nothing when no name is left."""
    names = sorted({slug for slug in slugs if is_listable_folder_name(slug)})
    if not names:
        return SortTimesRead(times={}, unreadable=frozenset())
    times: dict[str, float] = {}
    unreadable: set[str] = set()
    try:
        with _opened_from_trusted_root(Path(base)) as root:
            for name in names:
                try:
                    found = _folder_time(root, name)
                except OSError as exc:
                    if not _strictly_holds_no_id(exc=exc):
                        unreadable.add(name)
                    continue
                if found is not None:
                    times[name] = found
    except ValueError:
        return SortTimesRead(times={}, unreadable=frozenset())
    except OSError as exc:
        if _strictly_holds_no_id(exc=exc):
            return SortTimesRead(times={}, unreadable=frozenset())
        return SortTimesRead(times={}, unreadable=frozenset(names))
    return SortTimesRead(times=times, unreadable=frozenset(unreadable))


def _in_root(query, *, project_id: UUID, conversation_id: UUID | None, slugs: Collection[str]):
    """`query`, a select of artifact_sort_times, narrowed to the rows of
    `slugs` in one root: `project_id`'s shared root when `conversation_id`
    is None, else that conversation's legacy root."""
    query = query.where(ArtifactSortTime.project_id == project_id, ArtifactSortTime.slug.in_(list(slugs)))
    if conversation_id is None:
        return query.where(ArtifactSortTime.conversation_id.is_(None))
    return query.where(ArtifactSortTime.conversation_id == conversation_id)


def _times_held(
    session: ScopedSession, *, project_id: UUID, conversation_id: UUID | None, slugs: Collection[str]
) -> dict[str, float]:
    """The time of each of `slugs` that has a row in one root, read without a
    lock: a writer learns from it which folders it may have to write, and
    locks only those (_write). Read from the database even for a row the
    session holds already (populate_existing), as a unit's session keeps a
    row it loaded across its commits: a time the session holds from before
    a self-heal moved the row back would leave out a folder whose time
    rises."""
    query = _in_root(
        session.select(ArtifactSortTime), project_id=project_id, conversation_id=conversation_id, slugs=slugs
    ).execution_options(populate_existing=True)
    return {row.slug: row.sort_time for row in session.exec(query).all()}


def _rows(
    session: ScopedSession, *, project_id: UUID, conversation_id: UUID | None, slugs: Collection[str]
) -> dict[str, ArtifactSortTime]:
    """The rows of `slugs` in one root, locked until the caller commits.
    Every writer locks its rows here. First the rows the root's rows point
    at, in the order every delete takes them (lock_root_parents), so a
    writer never deadlocks a project or conversation delete that cascades
    into its rows. Then the rows (FOR UPDATE, taken in slug order), read
    from the database even when the session holds them already, as a unit's
    session keeps a row it loaded across its commits (populate_existing), so
    a writer that waited reads the time the other one committed, and
    record_sort_times never puts an older time back.
    _write adds its new rows in slug order too (code point order, as the
    column compares on PostgreSQL), and move_sort_times gives its rows their
    new slugs in that order, so two writers of the same folders, rows that
    exist or rows each one adds, queue on the first they share rather than
    deadlock. SQLite ignores FOR UPDATE; the PostgreSQL service tests pin
    the locks and the order (tests/test_artifact_sort_times_postgres.py)."""
    lock_root_parents(session, project_id=project_id, conversation_id=conversation_id)
    query = (
        _in_root(session.select(ArtifactSortTime), project_id=project_id, conversation_id=conversation_id, slugs=slugs)
        .order_by(ArtifactSortTime.slug)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    return {row.slug: row for row in session.exec(query).all()}


def _write(session: ScopedSession, *, project_id: UUID, conversation_id: UUID | None, times: Mapping[str, float]) -> int:
    """Write `times` and commit; the rows written. An unlocked read first
    (_times_held) leaves out each folder whose row holds the same or a newer
    time, so a write that changes nothing locks no row. The rest are locked
    in slug order (_rows) and checked again under the lock, since another
    writer may have written them since the read."""
    held = _times_held(session, project_id=project_id, conversation_id=conversation_id, slugs=times)
    rising = {slug: sort_time for slug, sort_time in times.items() if slug not in held or sort_time > held[slug]}
    if not rising:
        session.commit()
        return 0
    existing = _rows(session, project_id=project_id, conversation_id=conversation_id, slugs=rising)
    written = 0
    for slug, sort_time in sorted(rising.items()):
        row = existing.get(slug)
        if row is None:
            session.add(
                ArtifactSortTime(project_id=project_id, conversation_id=conversation_id, slug=slug, sort_time=sort_time)
            )
            written += 1
        elif sort_time > row.sort_time:
            row.sort_time = sort_time
            session.add(row)
            written += 1
    session.commit()
    return written


def record_sort_times(
    session: ScopedSession, *, project_id: UUID, times: Mapping[str, float], conversation_id: UUID | None = None
) -> int:
    """Write the time of each folder in `times`, folders of `project_id`'s
    shared root, or of `conversation_id`'s legacy root, and return how many
    rows it wrote. The newest time wins: a row whose time is the same or
    newer, another writer's included, is left as it is, so nothing is
    written, and no row locked, for a folder that did not move. A name the
    list would not open (artifact_identity.is_listable_folder_name) gets no
    row, and neither does a folder whose project or conversation is gone."""
    if not keeps_locations(session.scope):
        return 0
    times = {slug: sort_time for slug, sort_time in times.items() if is_listable_folder_name(slug)}
    if not times:
        return 0
    for _attempt in range(2):
        try:
            return _write(session, project_id=project_id, conversation_id=conversation_id, times=times)
        except IntegrityError:
            # Another writer added a folder's row after this one's read: the
            # batch again, whose read finds that row. A second failure means
            # the project or conversation is gone, or another writer won
            # again.
            session.rollback()
    # Each folder on its own, so one that cannot be written never keeps the
    # others out.
    written = 0
    for slug, sort_time in sorted(times.items()):
        try:
            written += _write(session, project_id=project_id, conversation_id=conversation_id, times={slug: sort_time})
        except IntegrityError:
            session.rollback()
    return written


def sort_time_rows(session: ScopedSession, *, project_id: UUID) -> dict[str, SortTimeRow]:
    """The rows of `project_id`'s shared root, by folder. A self-heal reads
    them before it lists the root, and passes them to reconcile_sort_times
    as `droppable`. Empty in a scope that keeps no rows."""
    if not keeps_locations(session.scope):
        return {}
    query = session.select(ArtifactSortTime).where(
        ArtifactSortTime.project_id == project_id, ArtifactSortTime.conversation_id.is_(None)
    )
    return {row.slug: SortTimeRow(row.id, row.sort_time) for row in session.exec(query).all()}


def reconcile_sort_times(
    session: ScopedSession,
    *,
    project_id: UUID,
    listed: Mapping[str, float],
    kept: Collection[str],
    droppable: Mapping[str, SortTimeRow],
    complete: bool,
) -> None:
    """Note what a full listing of `project_id`'s shared root found: `listed`,
    each folder whose time it read, and `kept`, every folder whose row must
    stay: those it found holding an id, and those it could not read. Each
    listed time is written (record_sort_times), so a folder whose
    `metadata.json` moved outside a turn's named edits moves up the list.

    The rows `droppable` holds, read before the listing began
    (sort_time_rows), decide two more writes. Each applies only to a row
    that is still the one `droppable` holds: same id, same time. A row
    another writer wrote while the listing ran stays as that writer left
    it: one it added, one a task move brought under that slug, which keeps
    its source's id (move_sort_times), and one a turn end moved to a newer
    time.
    - A listed time older than its row's is written too. The folder now
      holds an older `metadata.json`, such as a folder replaced by an older
      copy under the same name, and a walk sorts it by that time.
    - When the listing read the root itself (`complete`), a folder in
      neither `listed` nor `kept` is gone, or the list could show no card
      for it, and its row is dropped."""
    if not keeps_locations(session.scope):
        return
    gone = set(droppable).difference(listed, kept) if complete else set()
    older = {slug for slug, sort_time in listed.items() if slug in droppable and sort_time < droppable[slug].sort_time}
    if gone or older:
        for slug, row in _rows(session, project_id=project_id, conversation_id=None, slugs=gone | older).items():
            if SortTimeRow(row.id, row.sort_time) != droppable[slug]:
                continue
            if slug in gone:
                session.delete(row)
            else:
                row.sort_time = listed[slug]
                session.add(row)
        session.commit()
    record_sort_times(session, project_id=project_id, times=listed)


def forget_sort_time(session: ScopedSession, *, location: Location) -> None:
    """Drop the row of `location`, a folder just deleted."""
    if not keeps_locations(session.scope):
        return
    rows = _rows(session, project_id=location.project_id, conversation_id=location.conversation_id, slugs=[location.slug])
    for row in rows.values():
        session.delete(row)
    session.commit()


def move_sort_times(session: ScopedSession, *, project_id: UUID, moved: Mapping[str, str], to_project_id: UUID) -> None:
    """Follow folders moved from `project_id`'s shared root to
    `to_project_id`'s: `moved` maps each old slug to its new one. A moved
    folder keeps its row, its id and its time. A destination row under a
    new slug goes first: the move chose each new slug where the
    destination's disk held no folder, so such a row names a folder that is
    gone. One commit."""
    if not keeps_locations(session.scope) or not moved:
        return
    stale = _rows(session, project_id=to_project_id, conversation_id=None, slugs=list(moved.values()))
    rows = _rows(session, project_id=project_id, conversation_id=None, slugs=list(moved))
    for row in stale.values():
        session.delete(row)
    # The deletes reach the database before the moved rows take their
    # slugs: within one flush the ORM sends updates before deletes.
    session.flush()
    # One flush per row, in the order of the new slugs: a single flush
    # would send the updates in primary key order, and a writer that adds
    # rows under the same slugs adds them in slug order (_write), so the
    # two would take the unique index's entries in opposite orders.
    for slug, row in sorted(rows.items(), key=lambda item: moved[item[0]]):
        row.project_id, row.slug = to_project_id, moved[slug]
        session.add(row)
        session.flush()
    session.commit()


class SortTimeSentinel(BaseModel):
    """The sentinel's value, as JSON in the settings row SENTINEL_KEY."""

    #: Projects with no rows from the backfill: those whose held roots it
    #: could not read, those that moved or went away during their walk, and
    #: those the held-root sentinel listed as unread when it ran
    #: (HeldRootSentinel.unread), whose held rows may not name every root
    #: yet. The list reads every folder of their roots. Each later pass walks
    #: the ones the held-root sentinel no longer lists, and rewrites this
    #: list. The startup task runs one every
    #: ARTIFACT_BACKFILL_UNREAD_RETRY_SECONDS while this list is not empty
    #: (server._run_artifact_backfills).
    unread: list[UUID] = []


def read_sort_sentinel(session: ScopedSession) -> SortTimeSentinel | None:
    """The backfill's sentinel, or None before a pass has walked every
    project. A value that is not a SortTimeSentinel, as one written by hand,
    also reads as None (read_sentinel_value): the list then reads every
    folder of its roots instead of failing, and the next pass rewrites the
    value."""
    return read_sentinel_value(
        session,
        key=SENTINEL_KEY,
        model=SortTimeSentinel,
        logger=logger,
        unreadable_event="artifact_sort_times_sentinel_unreadable",
        unreadable_message=(
            "artifact_sort_times sentinel holds a value it cannot read; the list reads every folder of its roots"
        ),
    )


#: The most rows newest_folders reads of a run of rows that share one time,
#: when its limit cuts that run. Past it, the list reads every folder of
#: its roots unless the rows newer than the run fill it.
_TIED_ROWS = 2_000


def newest_folders(session: ScopedSession, *, roots: Collection[HeldRoot], limit: int) -> NewestFolders | None:
    """The newest `limit` rows of `roots`, newest first, or None when the
    list has to read every folder of its roots: this scope keeps no rows,
    or their backfill has written no sentinel it can read
    (read_sort_sentinel). The roots of a project the sentinel lists are
    left out of the rows and named in NewestFolders.unread.

    When the limit cuts a run of rows that share one time, a second query
    reads the rest of that run (NewestFolders.tied), up to _TIED_ROWS rows
    of it: the first query cuts such a run by slug, while a walk of every
    folder puts the folders of the earlier root first, so a list needs the
    whole run to order it as a walk does. One read of the sentinel and one
    query, or two when the limit cuts a run."""
    if not keeps_locations(session.scope):
        return None
    sentinel = read_sort_sentinel(session)
    if sentinel is None:
        return None
    unread = frozenset(sentinel.unread).intersection(root.project_id for root in roots)
    timed = [root for root in roots if root.project_id not in unread]
    shared = sorted({root.project_id for root in timed if root.conversation_id is None}, key=str)
    legacy = sorted({(root.project_id, root.conversation_id) for root in timed if root.conversation_id is not None}, key=str)
    clauses = []
    if shared:
        clauses.append(and_(ArtifactSortTime.conversation_id.is_(None), ArtifactSortTime.project_id.in_(shared)))
    if legacy:
        clauses.append(tuple_(ArtifactSortTime.project_id, ArtifactSortTime.conversation_id).in_(legacy))
    if not clauses:
        return NewestFolders(rows=(), complete=True, unread=unread, tied=(), ties_read=True)
    query = (
        session.select(ArtifactSortTime)
        .where(or_(*clauses))
        # A total order, so PostgreSQL's top-N sort cuts a run of tied
        # times the same way on every read. slug compares by code point
        # (the column's collation), so within one root the cut keeps the
        # folders a walk puts first (sorted by name). Across roots a walk
        # puts the earlier root first, which the rows cannot know.
        .order_by(ArtifactSortTime.sort_time.desc(), ArtifactSortTime.slug, ArtifactSortTime.id)
        .limit(limit + 1)
    )
    found = session.exec(query).all()
    rows = found[:limit]
    tied: list[ArtifactSortTime] = []
    ties_read = True
    # The row past the limit shares the last kept row's time: the limit
    # cut a run of tied times, so read the rest of it, in the same order.
    if len(found) > limit and found[limit].sort_time == rows[-1].sort_time:
        run = session.exec(
            session.select(ArtifactSortTime)
            .where(or_(*clauses), ArtifactSortTime.sort_time == rows[-1].sort_time)
            .order_by(ArtifactSortTime.slug, ArtifactSortTime.id)
            .limit(_TIED_ROWS + 1)
        ).all()
        ties_read = len(run) <= _TIED_ROWS
        if ties_read:
            # By folder, not by row id: between the two queries a writer may
            # drop a kept row and add the same folder's row again under a
            # new id, and the list must not open that folder twice.
            kept = {Location(row.project_id, row.conversation_id, row.slug) for row in rows}
            tied = [row for row in run if Location(row.project_id, row.conversation_id, row.slug) not in kept]

    def sort_times(read: Iterable[ArtifactSortTime]) -> tuple[SortTime, ...]:
        return tuple(SortTime(row.project_id, row.conversation_id, row.slug, row.sort_time) for row in read)

    return NewestFolders(
        rows=sort_times(rows), complete=len(found) <= limit, unread=unread, tied=sort_times(tied), ties_read=ties_read
    )


def _times_in(source: ProjectArtifacts) -> dict[str, float]:
    """The time of every folder in `source`'s root the list would open
    (held_artifact_roots.listable_folders). A root that is missing, a link,
    or one the server may not read holds none, and so does a folder of it
    that is gone or that the server may not read. Any other error raises."""
    times: dict[str, float] = {}
    try:
        with opened_artifact_root(source) as root:
            # Listed in full first, so the listing is closed before the stats.
            for name in list(listable_folders(root)):
                try:
                    found = _folder_time(root, name)
                except OSError as exc:
                    if _strictly_holds_no_id(exc=exc):
                        continue
                    raise
                if found is not None:
                    times[name] = found
    except ValueError:
        return {}
    except OSError as exc:
        if _strictly_holds_no_id(exc=exc):
            return {}
        raise
    return times


class _HeldProject(NamedTuple):
    """A project the backfill walks, and the held roots its rows name: the
    shared root when a conversation is None, else that legacy root."""

    project: ProjectRow
    org_id: str | None
    conversation_ids: tuple[UUID | None, ...]


class _HeldRootRow(NamedTuple):
    """One row of the backfill's query: a held root (its conversation, None
    for the shared root) and its project's columns, in the query's order."""

    conversation_id: UUID | None
    id: UUID
    name: str
    path: str
    org_id: str | None


def _held_projects(rows: Iterable[_HeldRootRow]) -> list[_HeldProject]:
    """`rows`, one per held root ordered by project, grouped by project."""
    from cowork.services.artifact_roots import ProjectRow

    projects: list[_HeldProject] = []
    for _project_id, group in groupby(rows, key=lambda row: row.id):
        roots = list(group)
        first = roots[0]
        project = ProjectRow(id=UUID(str(first.id)), name=first.name, path=first.path, external=False)
        conversation_ids = tuple(
            UUID(str(root.conversation_id)) if root.conversation_id is not None else None for root in roots
        )
        projects.append(_HeldProject(project=project, org_id=first.org_id, conversation_ids=conversation_ids))
    return projects


@dataclass
class SortTimeBackfillSummary:
    #: Rows the pass wrote: a folder's new row, or a row moved to a newer
    #: time (record_sort_times).
    recorded: int = 0
    #: Projects whose held roots could not all be read, or that moved or went
    #: away during their walk; the sentinel lists them.
    unread: int = 0
    #: Projects the held-root sentinel lists, whose held rows may not name
    #: every root yet; the sentinel lists them too.
    held_pending: int = 0


def _time_project(
    engine,
    *,
    held: _HeldProject,
    first_pass: bool,
    shared_root: Path,
    summary: SortTimeBackfillSummary,
    stop: threading.Event | None,
) -> bool:
    """Record the time of every folder in the held roots of one project.
    False when its roots could not all be read, or when the project moved or
    went away during its walk: its folder is missing after the walk at the
    path the pass walked (walked_folder_is_gone), or its row no longer holds
    that path (walked_path_is_current), since the walk may then have found
    its roots gone or read another project's; it then records none.

    `first_pass` is True for a first or resumed pass, which also counts a
    folder already missing before its walk as moved, since a rename may
    have moved it and not yet committed. It is False for a pass over the
    sentinel's list, which counts only a folder there before its walk and
    missing after it as moved. That pass takes a folder already missing
    before its walk for one gone for good: the project counts as read, with
    no rows, so it leaves the list. `shared_root` is the resolved shared
    root (projects.resolved_shared_root), read once for the pass."""
    from cowork.services.artifact_roots import left_out_for_another_partition, recorded_source

    if not held.org_id:
        return True
    # In org tenancy an organization's partition of the shared root is its id
    # (projects.partition_of).
    if left_out_for_another_partition(held.project, partition=str(held.org_id), shared_root=shared_root):
        return True
    times: dict[UUID | None, dict[str, float]] = {}
    try:
        # A folder there before the walk and gone after it moved during the
        # walk, as a rename moves it before it commits the new path. A first
        # pass also counts a folder already gone before its walk, since a
        # rename may have moved it and not yet committed: the pass walks only
        # projects with a held root, whose folder held an artifact folder. A
        # rename moves the folder and commits within one request, so a later
        # pass over the sentinel's list reads such a project at its new path.
        # There a folder gone before the walk at a path the row still holds
        # is gone for good, as a wiped volume leaves it until the project is
        # opened again (ProjectService.ensure_dir_exists): the project counts
        # as read, with no rows, rather than listed again on every pass.
        there = first_pass or not walked_folder_is_gone(held.project)
        for conversation_id in held.conversation_ids:
            check_stop(stop)
            times[conversation_id] = _times_in(recorded_source(held.project, conversation_id))
        gone = there and walked_folder_is_gone(held.project)
    except OSError as exc:
        summary.unread += 1
        logger.warning(
            "artifact_sort_times backfill could not read a project's held roots; the list reads every folder of "
            "them, and the next pass reads them again",
            extra={
                **log_context(project_id=held.project.id),
                "event": "artifact_sort_times_backfill_unread",
                "error_type": type(exc).__name__,
            },
        )
        return False
    with Session(engine) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=str(held.org_id)))
        if gone or not walked_path_is_current(session, project=held.project):
            summary.unread += 1
            logger.warning(
                "artifact_sort_times backfill walked a project that moved or went away meanwhile; the list reads "
                "every folder of it, and the next pass reads it again",
                extra={**log_context(project_id=held.project.id), "event": "artifact_sort_times_backfill_moved"},
            )
            return False
        for conversation_id, root_times in times.items():
            summary.recorded += record_sort_times(
                session, project_id=held.project.id, conversation_id=conversation_id, times=root_times
            )
    return True


class _Sentinels(NamedTuple):
    held: HeldRootSentinel | None
    sort: SortTimeSentinel | None


def _read_sentinels(engine) -> _Sentinels:
    with Session(engine) as raw:
        session = ScopedSession(raw, SYSTEM_SCOPE)
        return _Sentinels(held=read_sentinel(session), sort=read_sort_sentinel(session))


def _skip_reason(sentinels: _Sentinels) -> str | None:
    """Why a pass walks nothing (log_skipped): `finished` (the sentinel lists
    no unread project) or `held_pending` (the held-root backfill has written
    no sentinel, so the held rows do not name every root yet). None when the
    pass walks."""
    if sentinels.sort is not None and not sentinels.sort.unread:
        return "finished"
    if sentinels.held is None:
        return "held_pending"
    return None


def run_sort_time_backfill(
    engine=None, *, try_lock=None, project_ids=None, stop: threading.Event | None = None
) -> SortTimeBackfillSummary | None:
    """Read the time of every folder in every held root once per environment,
    record it, then write the sentinel with the projects that have no rows
    from it (SortTimeSentinel). None when skipped or stopped. It waits for the
    held-root backfill's sentinel (held_artifact_roots), since the held rows
    name the roots it reads, and runs after it in the same startup task.

    One replica runs it at a time (an advisory lock). The first pass walks
    the projects with a held root in id order and leaves out those the
    held-root sentinel lists as unread. In the PROGRESS_KEY setting row it
    keeps the last project it finished, each one it could not read, and each
    one the held-root sentinel listed. A pass resumed after a stop or a
    restart walks the projects after the last one it finished, and each
    project the row lists that has held rows now and that the held-root
    sentinel no longer lists: a stop loses no project. Once it has walked
    the last project it deletes the progress row and writes the sentinel,
    listing the projects it could not read, those that moved or went away
    during their walk, and those the held-root sentinel lists. A later pass
    that finds that list, at a start or the startup task's next run, walks
    only its projects that the held-root sentinel no longer lists, keeps no
    progress, and rewrites the list; one that finds it empty walks nothing.
    Such a pass counts a project whose folder is missing before its walk as
    read, with no rows (_time_project), so a folder gone for good leaves the
    list once.
    A folder's row keeps the newest time any writer read
    (record_sort_times). Raises when a database write fails for any reason
    but a row another writer recorded first. ``project_ids`` limits the
    pass to those projects; it exists for tests, whose database is shared
    across modules. Production passes nothing.
    """
    from cowork.services.projects import resolved_shared_root

    if get_app_settings().tenancy_mode != "org":
        log_skipped(logger, event_prefix="artifact_sort_times", reason="not_org")
        return None
    engine = engine or get_engine(get_app_settings().database.uri)
    lock = try_lock or advisory_try_lock
    reason = _skip_reason(_read_sentinels(engine))
    if reason is not None:
        log_skipped(logger, event_prefix="artifact_sort_times", reason=reason)
        return None
    with lock(engine, _LOCK_KEY) as acquired:
        if not acquired:
            log_skipped(logger, event_prefix="artifact_sort_times", reason="locked")
            return None
        # Again under the lock: another replica may have finished a pass, or
        # the held-root backfill rewritten its list, since the read above.
        sentinels = _read_sentinels(engine)
        reason = _skip_reason(sentinels)
        if reason is not None:
            log_skipped(logger, event_prefix="artifact_sort_times", reason=reason)
            return None
        held_unread = frozenset(sentinels.held.unread)
        query = (
            select(HeldArtifactRoot.conversation_id, Project.id, Project.name, Project.path, Project.org_id)
            .join(Project, Project.id == HeldArtifactRoot.project_id)
            .order_by(Project.id)
        )
        if sentinels.sort is not None:
            # Once the sentinel exists, the pass walks only the projects it
            # lists, and keeps no progress: the list is short.
            reason = "unread"
            progress = BackfillProgress()
            query = query.where(Project.id.in_(sentinels.sort.unread))
        else:
            reason = "first"
            progress = read_progress(engine, PROGRESS_KEY)
            if progress.after is not None:
                reason = "resumed"
                query = query.where(or_(Project.id > progress.after, Project.id.in_(progress.unread)))
        if project_ids is not None:
            query = query.where(Project.id.in_(list(project_ids)))
        with Session(engine) as raw:
            rows = [_HeldRootRow(*row) for row in raw.exec(query).all()]
        # A project the held-root sentinel lists has no held rows from its
        # backfill, so its held rows may not name every root yet. It stays in the progress,
        # so a resumed first pass walks it once the held-root backfill has
        # read it, even when the progress is past its id, and in the
        # sentinel this pass writes.
        projects = [held for held in _held_projects(rows) if held.project.id not in held_unread]
        progress = progress.within({held.project.id for held in projects})
        progress = BackfillProgress(after=progress.after, unread=sorted(held_unread.union(progress.unread), key=str))
        logger.warning(
            "artifact_sort_times backfill starts its walk",
            extra={"event": "artifact_sort_times_backfill_started", "reason": reason, "count": len(projects)},
        )
        summary = SortTimeBackfillSummary(held_pending=len(held_unread))
        # Read once for every project the pass walks, not once per project.
        shared_root = resolved_shared_root()
        try:
            for held in projects:
                check_stop(stop)
                read = _time_project(
                    engine,
                    held=held,
                    first_pass=sentinels.sort is None,
                    shared_root=shared_root,
                    summary=summary,
                    stop=stop,
                )
                progress = progress.finished(project_id=held.project.id, read=read)
                if sentinels.sort is None:
                    save_progress(engine, PROGRESS_KEY, progress=progress)
        except BackfillStopped:
            logger.warning(
                "artifact_sort_times backfill stopped before it finished; the next start resumes it",
                extra={"event": "artifact_sort_times_backfill_stopped"},
            )
            return None
        for outcome, count in (
            ("recorded", summary.recorded),
            ("unread", summary.unread),
            ("held_pending", summary.held_pending),
        ):
            logger.warning(
                "artifact_sort_times backfill finished its walk",
                extra={"event": "artifact_sort_times_backfill_summary", "outcome": outcome, "count": count},
            )
        if sentinels.sort is None:
            # Progress first, as the held-root backfill does: a crash between
            # the two leaves neither, and the next start walks every project
            # again, rather than a progress row that would skip projects once
            # the sentinel is deleted for a rerun.
            delete_progress(engine, PROGRESS_KEY)
        unread = sorted(progress.unread, key=str)
        write_sentinel(engine, SENTINEL_KEY, value=SortTimeSentinel(unread=unread).model_dump_json())
    return summary
