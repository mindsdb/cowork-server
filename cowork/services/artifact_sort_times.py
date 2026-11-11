"""The artifact_sort_times rows (cowork/models/artifact_sort_time.py): the time
the artifacts list sorts each artifact folder by.

Org tenancy only. With the held-root rows (held_artifact_roots), they let the
list choose its newest folders in the database and open only those
(newest_folders), instead of reading every folder's `metadata.json` in the
roots it opens. The list trusts the rows once their backfill has finished,
and until then reads every folder.

A row holds the modification time of the folder's `metadata.json` as the
server last read it: at the turn end that created or changed the folder
(task_objects.record_new_artifacts, with the times turn_artifact_changes
read), at the self-heal's listing for a folder no row names
(artifact_autopublish._record_listing, which also drops the rows of folders
the listing found gone), and at the backfill below. A rewrite of
`metadata.json` the server makes on its own, such as an id widening
(artifact_identity.ensure_full_id), leaves the row as it was, so an old
artifact does not move up the list. A row moves with a task's folders
(move_sort_times) and goes with a deleted folder (forget_sort_time).
"""
from __future__ import annotations

import logging
import stat
import threading
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple
from uuid import UUID

from sqlalchemy import and_, or_
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from cowork.common.logger import log_context
from cowork.common.paths import dir_child_stat, dir_scandir, is_child_name
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.models.artifact_sort_time import ArtifactSortTime
from cowork.models.held_artifact_root import HeldArtifactRoot
from cowork.models.project import Project
from cowork.services.artifact_identity import (
    _opened_from_trusted_root,
    _strictly_holds_no_id,
    opened_artifact_root,
)
from cowork.services.artifact_locations import Location, keeps_locations
from cowork.services.artifact_owner_backfill import advisory_try_lock, sentinel_present, write_sentinel
from cowork.services.artifacts import ProjectArtifacts
from cowork.services.held_artifact_roots import (
    SENTINEL_KEY as HELD_SENTINEL_KEY,
    HeldRoot,
    _check_stop,
    _Stopped,
    sentinel_written,
)
from cowork.services.shared_resources import advisory_lock_key

logger = logging.getLogger(__name__)

#: Written once a backfill pass has read every held root; the list trusts the
#: rows from then on. The revision that made the table deletes it on a
#: downgrade (33c9ba9927ad_artifact_sort_times.py), which names it as a literal.
SENTINEL_KEY = "_artifact_sort_times_backfill_v1"
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


def _folder_time(root, name: str) -> float | None:
    """The modification time of `name`'s `metadata.json`, `name` a direct
    child folder of the pinned `root`, opened without following a link in its
    place (dir_child_stat). None when `metadata.json` is not a regular file:
    the list shows no card for such a folder."""
    found = dir_child_stat(root, name, "metadata.json")
    return found.st_mtime if stat.S_ISREG(found.st_mode) else None


def sort_times_below(*, base: Path, slugs: Iterable[str]) -> dict[str, float]:
    """The time of each of `slugs` that is an artifact folder directly below
    `base`, a project's artifacts folder opened from the trusted root. A name
    that is not one folder there, or that cannot be read, gets no time.
    Opens nothing when `slugs` is empty."""
    names = sorted({slug for slug in slugs if is_child_name(slug)})
    if not names:
        return {}
    times: dict[str, float] = {}
    try:
        with _opened_from_trusted_root(Path(base)) as root:
            for name in names:
                try:
                    found = _folder_time(root, name)
                except OSError:
                    continue
                if found is not None:
                    times[name] = found
    except (OSError, ValueError):
        return {}
    return times


def _rows(session: ScopedSession, *, project_id: UUID, conversation_id: UUID | None, slugs: Collection[str]):
    query = session.select(ArtifactSortTime).where(
        ArtifactSortTime.project_id == project_id, ArtifactSortTime.slug.in_(list(slugs))
    )
    if conversation_id is None:
        query = query.where(ArtifactSortTime.conversation_id.is_(None))
    else:
        query = query.where(ArtifactSortTime.conversation_id == conversation_id)
    return {row.slug: row for row in session.exec(query).all()}


def _write(session: ScopedSession, *, project_id: UUID, conversation_id: UUID | None, times: Mapping[str, float]) -> None:
    existing = _rows(session, project_id=project_id, conversation_id=conversation_id, slugs=times)
    for slug, sort_time in times.items():
        row = existing.get(slug)
        if row is None:
            row = ArtifactSortTime(project_id=project_id, conversation_id=conversation_id, slug=slug, sort_time=sort_time)
        else:
            row.sort_time = sort_time
        session.add(row)
    session.commit()


def record_sort_times(
    session: ScopedSession, *, project_id, times: Mapping[str, float], conversation_id=None
) -> None:
    """Write the time of each folder in `times`, folders of `project_id`'s
    shared root, or of `conversation_id`'s legacy root, replacing the time a
    row held. A folder another writer recorded meanwhile keeps that writer's
    time, and one whose project or conversation is gone gets no row."""
    if not keeps_locations(session.scope) or not times:
        return
    project_id = UUID(str(project_id))
    conversation_id = UUID(str(conversation_id)) if conversation_id is not None else None
    try:
        _write(session, project_id=project_id, conversation_id=conversation_id, times=times)
    except IntegrityError:
        session.rollback()
        for slug, sort_time in times.items():
            try:
                _write(session, project_id=project_id, conversation_id=conversation_id, times={slug: sort_time})
            except IntegrityError:
                session.rollback()


def reconcile_sort_times(
    session: ScopedSession, *, project_id, listed: Mapping[str, float], kept: Collection[str], complete: bool
) -> None:
    """Note what a full listing of `project_id`'s shared root found: `listed`,
    each folder whose time it read, and `kept`, every folder whose row must
    stay: those it found holding an id, and those it could not read. A
    folder no row names gets one with the listing's time; a row keeps its
    own, since a turn end or the backfill wrote it. When the listing read the
    root itself (`complete`), a row naming a folder in neither is dropped:
    the folder is gone, or the list could show no card for it."""
    if not keeps_locations(session.scope):
        return
    project_id = UUID(str(project_id))
    query = session.select(ArtifactSortTime).where(
        ArtifactSortTime.project_id == project_id, ArtifactSortTime.conversation_id.is_(None)
    )
    rows = {row.slug: row for row in session.exec(query).all()}
    missing = {slug: sort_time for slug, sort_time in listed.items() if slug not in rows}
    if complete:
        for slug, row in rows.items():
            if slug not in listed and slug not in kept:
                session.delete(row)
        session.commit()
    record_sort_times(session, project_id=project_id, times=missing)


def forget_sort_time(session: ScopedSession, *, location: Location) -> None:
    """Drop the row of `location`, a folder just deleted."""
    if not keeps_locations(session.scope):
        return
    rows = _rows(session, project_id=location.project_id, conversation_id=location.conversation_id, slugs=[location.slug])
    for row in rows.values():
        session.delete(row)
    session.commit()


def move_sort_times(session: ScopedSession, *, project_id, moved: Mapping[str, str], to_project_id) -> None:
    """Follow folders moved from `project_id`'s shared root to
    `to_project_id`'s: `moved` maps each old slug to its new one. A moved
    folder keeps its time."""
    if not keeps_locations(session.scope) or not moved:
        return
    rows = _rows(session, project_id=UUID(str(project_id)), conversation_id=None, slugs=list(moved))
    for slug, row in rows.items():
        row.project_id, row.slug = UUID(str(to_project_id)), moved[slug]
        session.add(row)
    session.commit()


def newest_folders(session: ScopedSession, *, roots: Collection[HeldRoot], limit: int) -> NewestFolders | None:
    """The newest `limit` rows of `roots`, newest first, or None when the
    caller has to read every folder of its roots: this scope keeps no rows,
    or their backfill has not finished. One query."""
    if not keeps_locations(session.scope) or not sentinel_written(session, key=SENTINEL_KEY):
        return None
    shared = sorted({root.project_id for root in roots if root.conversation_id is None}, key=str)
    legacy = sorted({root.conversation_id for root in roots if root.conversation_id is not None}, key=str)
    clauses = []
    if shared:
        clauses.append(and_(ArtifactSortTime.conversation_id.is_(None), ArtifactSortTime.project_id.in_(shared)))
    if legacy:
        clauses.append(ArtifactSortTime.conversation_id.in_(legacy))
    if not clauses:
        return NewestFolders(rows=(), complete=True)
    query = (
        session.select(ArtifactSortTime)
        .where(or_(*clauses))
        .order_by(ArtifactSortTime.sort_time.desc())
        .limit(limit + 1)
    )
    rows = session.exec(query).all()
    return NewestFolders(
        rows=tuple(SortTime(row.project_id, row.conversation_id, row.slug, row.sort_time) for row in rows[:limit]),
        complete=len(rows) <= limit,
    )


def _times_in(source: ProjectArtifacts) -> dict[str, float]:
    """The time of every folder in `source`'s root the list would open: a
    folder, not a link, whose name does not start with a dot
    (artifacts.list_artifacts). A root that is missing, a link, or one the
    server may not read holds none. Any other error raises."""
    times: dict[str, float] = {}
    try:
        with opened_artifact_root(source) as root:
            with dir_scandir(root) as entries:
                names = []
                for entry in entries:
                    if entry.name.startswith("."):
                        continue
                    try:
                        if not entry.is_symlink() and entry.is_dir(follow_symlinks=False):
                            names.append(entry.name)
                    except OSError as exc:
                        if not _strictly_holds_no_id(exc=exc):
                            raise
            for name in names:
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


@dataclass
class SortTimeBackfillSummary:
    #: Folders given a row (or whose row the pass rewrote).
    recorded: int = 0
    #: Roots that could not be read; the sentinel waits for them.
    unread: int = 0


def run_sort_time_backfill(
    engine=None, *, try_lock=None, project_ids=None, stop: threading.Event | None = None
) -> SortTimeBackfillSummary | None:
    """Read the time of every folder in every held root once per environment,
    record it, then write the sentinel. None when skipped or stopped. It waits
    for the held-root backfill (held_artifact_roots), whose rows name the
    roots, and runs after it in the same startup task.

    One replica runs it at a time (an advisory lock). A stop, a restart or a
    root it could not read leaves the sentinel unwritten, so the next start
    reads every root again; a folder's row then takes the time read last.
    ``project_ids`` limits the pass to those projects; it exists for tests,
    whose database is shared across modules. Production passes nothing.
    """
    from cowork.services.artifact_roots import ProjectRow, left_out_for_another_partition, recorded_source
    from cowork.services.projects import resolved_shared_root

    if get_app_settings().tenancy_mode != "org":
        return None
    engine = engine or get_engine(get_app_settings().database.uri)
    lock = try_lock or advisory_try_lock
    if sentinel_present(engine, SENTINEL_KEY) or not sentinel_present(engine, HELD_SENTINEL_KEY):
        return None
    with lock(engine, _LOCK_KEY) as acquired:
        if not acquired or sentinel_present(engine, SENTINEL_KEY):
            return None
        with Session(engine) as raw:
            query = (
                select(HeldArtifactRoot.org_id, HeldArtifactRoot.conversation_id, Project.id, Project.name, Project.path)
                .join(Project, Project.id == HeldArtifactRoot.project_id)
                .order_by(Project.id)
            )
            if project_ids is not None:
                query = query.where(Project.id.in_(list(project_ids)))
            roots = raw.exec(query).all()
        summary = SortTimeBackfillSummary()
        # Read once for every root the pass reads, not once per root.
        shared_root = resolved_shared_root()
        try:
            for org_id, conversation_id, project_id, name, path in roots:
                _check_stop(stop)
                project = ProjectRow(id=UUID(str(project_id)), name=name, path=path, external=False)
                # In org tenancy an organization's partition of the shared
                # root is its id (projects.partition_of).
                if left_out_for_another_partition(project, partition=str(org_id), shared_root=shared_root):
                    continue
                try:
                    times = _times_in(recorded_source(project, conversation_id))
                except OSError as exc:
                    summary.unread += 1
                    logger.warning(
                        "artifact_sort_times backfill could not read a root; the next start reads again",
                        extra={
                            **log_context(project_id=project.id, conversation_id=conversation_id),
                            "event": "artifact_sort_times_backfill_unread",
                            "error_type": type(exc).__name__,
                        },
                    )
                    continue
                with Session(engine) as raw:
                    record_sort_times(
                        ScopedSession(raw, TenantScope(org_mode=True, org_id=str(org_id))),
                        project_id=project.id, conversation_id=conversation_id, times=times,
                    )
                summary.recorded += len(times)
        except _Stopped:
            logger.warning(
                "artifact_sort_times backfill stopped before it finished; the next start reads again",
                extra={"event": "artifact_sort_times_backfill_stopped"},
            )
            return None
        for outcome, count in (("recorded", summary.recorded), ("unread", summary.unread)):
            logger.warning(
                "artifact_sort_times backfill finished its walk",
                extra={"event": "artifact_sort_times_backfill_summary", "outcome": outcome, "count": count},
            )
        if not summary.unread:
            write_sentinel(engine, SENTINEL_KEY)
    return summary
