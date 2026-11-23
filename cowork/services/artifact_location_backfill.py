"""One-time backfill of artifact_locations rows (cowork/models/artifact_location.py).

Version 2 also indexes published URLs for notifications sent before pushes carried artifact IDs.

Artifacts made before the table existed have no row, so their first lookup
lists the project's roots. This pass records a row for each of them, once per
environment, at cowork-server startup in org tenancy. It walks every org
project's shared root and legacy conversation roots, reads each folder's id
without writing, and records what it found per project
(record_checked_folders): an id held by one folder gets its row, and an id
held by two folders of a project is marked conflicted, so its lookups keep
listing the roots and answer 409 as before.

A project whose roots and folders it cannot all read gets no rows from it,
since a folder it could not read may hold an id another folder holds too.
Only a folder that is gone, is a link, holds no metadata that parses, or is
one the server may not read at all holds no id
(cowork.services.artifact_identity.identities_below).

It runs as one file store call (cowork.db.units.run_efs, no deadline), so it
holds one of the file store's slots for the whole walk. A started file store
call is never abandoned, so shutdown sets ``stop`` and the walk ends at the
next root or id. The pass walks projects in id order and records each project
it finishes, and each it could not read, in the
``_artifact_location_backfill_v2_progress`` setting row. So a stop or a
restart costs only the project the walk was in: the next start walks the
projects it has not finished and the ones it could not read, and drops from
the progress an unread project deleted since. An id it cannot write (a value
the database refuses, or a row another writer deleted) is skipped and
counted, and so is an id whose one folder has a name that is not valid
UTF-8; any other database error ends the pass. The sentinel is written, and
the progress row deleted, once a pass reaches the last project with none
left unread. To walk every project again, delete the
``_artifact_location_backfill_v2`` and ``_artifact_location_backfill_v2_progress``
setting rows and restart cowork-server.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from sqlalchemy import or_
from sqlalchemy.exc import DataError, IntegrityError
from sqlalchemy.orm.exc import StaleDataError
from sqlmodel import Session, select

from cowork.common.logger import log_context
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.models.project import Project
from cowork.models.setting import Setting
from cowork.services.artifact_identity import _strictly_holds_no_id, identities_below, opened_artifact_folder, storable_name
from cowork.services.artifact_share_links import record_share_link
from cowork.services.artifacts import _load_published_map_pinned
from cowork.services.artifact_locations import Location, RecordOutcome, record_checked_folders
from cowork.services.artifact_owner_backfill import (
    BackfillStopped,
    advisory_try_lock,
    check_stop,
    delete_progress,
    read_progress,
    save_progress,
    sentinel_present,
    write_sentinel,
)
from cowork.services.artifact_roots import (
    ProjectRow,
    left_out_for_another_partition,
    project_roots,
    recorded_location,
)
from cowork.services.projects import resolved_shared_root
from cowork.services.shared_resources import advisory_lock_key

logger = logging.getLogger(__name__)

SENTINEL_KEY = "_artifact_location_backfill_v2"
PROGRESS_KEY = "_artifact_location_backfill_v2_progress"
_LOCK_KEY = advisory_lock_key(b"cowork\0artifact_location_backfill_v2")


@dataclass
class LocationBackfillSummary:
    #: Ids whose row names their one folder.
    recorded: int = 0
    #: Ids held by two or more folders of a project, marked conflicted.
    duplicated: list[str] = field(default_factory=list)
    #: Ids whose row this pass left naming another folder, or conflicted.
    conflicts: int = 0
    #: Projects whose roots or folders could not all be read; they got no rows.
    unread: int = 0
    #: Ids whose row the database refused or another writer deleted mid-write,
    #: and ids whose one folder has a name that is not valid UTF-8, which no
    #: row can store.
    skipped: int = 0


def location_index_ready(session: ScopedSession) -> bool:
    return session.exec(session.select(Setting).where(
        Setting.key == SENTINEL_KEY, Setting.scope.is_(None),
    )).first() is not None


def _published_urls(source, folder_name: str) -> list[str]:
    try:
        with opened_artifact_folder(source, folder_name) as folder:
            records = _load_published_map_pinned(folder, strict=True)
    except OSError as exc:
        if _strictly_holds_no_id(exc=exc):
            return []
        raise
    urls = []
    for record in records.values():
        if not isinstance(record, dict) or not record.get("published", True):
            continue
        url = record.get("url")
        if isinstance(url, str) and url.startswith("https://") and len(url) <= 2048 and storable_name(url):
            urls.append(url)
    return urls


def _found_locations(
    project: ProjectRow, *, stop: threading.Event | None,
) -> tuple[dict[str, list[Location]], set[tuple[str, str]]]:
    """Collect locations and active share URLs on the same resumable project pass."""
    found: dict[str, list[Location]] = {}
    links: set[tuple[str, str]] = set()
    for root in project_roots(project, strict=True):
        check_stop(stop)
        for artifact_id, folder_names in identities_below(root.source).items():
            for folder_name in folder_names:
                check_stop(stop)
                location = recorded_location(root.source, root.source.base / folder_name)
                if location is not None:
                    found.setdefault(artifact_id, []).append(location)
                    links.update((artifact_id, url) for url in _published_urls(root.source, folder_name))
    return found, links


def _record_project(
    engine,
    *,
    org_id: str,
    project_id: UUID,
    found: dict[str, list[Location]],
    summary: LocationBackfillSummary,
    stop: threading.Event | None,
) -> None:
    with Session(engine) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=org_id))
        for artifact_id, locations in found.items():
            check_stop(stop)
            try:
                # The walk read every root of the project.
                outcome = record_checked_folders(session, artifact_id=artifact_id, folders=locations)
            except IntegrityError:
                # The folder's project or conversation was deleted during the
                # walk.
                session.rollback()
                summary.conflicts += 1
                continue
            except (DataError, StaleDataError, ValueError) as exc:
                # A value the database or the row's model refuses, or a row a
                # concurrent delete removed before this write committed: this
                # id gets no row, and the walk goes on. Any other error ends
                # the pass.
                session.rollback()
                summary.skipped += 1
                logger.warning(
                    "artifact_location_backfill skipped an id it could not record",
                    extra={
                        **log_context(project_id=project_id),
                        "event": "artifact_location_backfill_skipped",
                        "error_type": type(exc).__name__,
                    },
                )
                continue
            if len(locations) > 1:
                summary.duplicated.append(artifact_id)
            elif outcome is RecordOutcome.RECORDED:
                summary.recorded += 1
            elif outcome is RecordOutcome.NOTHING:
                # The id's one folder has a name no row can store.
                summary.skipped += 1
            else:
                summary.conflicts += 1


def _walk_project(
    engine,
    *,
    project: ProjectRow,
    org_id: str | None,
    shared_root: Path,
    summary: LocationBackfillSummary,
    stop: threading.Event | None,
) -> bool:
    """Record the rows of one project. False when its roots or folders
    could not all be read, so it got no rows. `shared_root` is the resolved
    shared root (projects.resolved_shared_root), read once for the pass."""
    if not org_id:
        return True
    # In org tenancy an organization's partition of the shared root is its id
    # (projects.partition_of).
    if left_out_for_another_partition(project, partition=str(org_id), shared_root=shared_root):
        return True
    try:
        found, links = _found_locations(project, stop=stop)
    except OSError as exc:
        summary.unread += 1
        logger.warning(
            "artifact_location_backfill could not read a project; it gets no rows from this pass",
            extra={
                **log_context(project_id=project.id),
                "event": "artifact_location_backfill_unread",
                "error_type": type(exc).__name__,
            },
        )
        return False
    _record_project(engine, org_id=org_id, project_id=project.id, found=found, summary=summary, stop=stop)
    if links:
        with Session(engine) as raw:
            session = ScopedSession(raw, TenantScope(org_mode=True, org_id=org_id))
            for artifact_id, url in links:
                check_stop(stop)
                record_share_link(session, project.id, artifact_id, url)
    return True


def run_artifact_location_backfill(
    engine=None, *, try_lock=None, project_ids=None, stop: threading.Event | None = None
) -> LocationBackfillSummary | None:
    """Run the pass once per environment. None when skipped or stopped.

    Raises when the pass cannot run or a write fails (for example the
    database is unreachable, or no pooled connection frees in time); the
    sentinel is then not written and the next start goes on from the
    projects the kept progress has not finished. A project it could not read
    is counted in ``unread``, kept in the progress, and leaves the sentinel
    unwritten too. Recording a folder again changes nothing. ``project_ids``
    limits the pass to those projects; it exists for tests, whose database
    is shared across modules. Production passes nothing.
    """
    if get_app_settings().tenancy_mode != "org":
        return None
    engine = engine or get_engine(get_app_settings().database.uri)
    lock = try_lock or advisory_try_lock
    if sentinel_present(engine, SENTINEL_KEY):
        return None
    with lock(engine, _LOCK_KEY) as acquired:
        if not acquired:
            return None
        if sentinel_present(engine, SENTINEL_KEY):
            return None
        progress = read_progress(engine, PROGRESS_KEY)
        with Session(engine) as raw:
            query = select(Project.id, Project.name, Project.path, Project.org_id).order_by(Project.id)
            if progress.after is not None:
                query = query.where(or_(Project.id > progress.after, Project.id.in_(progress.unread)))
            if project_ids is not None:
                query = query.where(Project.id.in_(list(project_ids)))
            projects = raw.exec(query).all()
        # An unread project deleted since leaves the progress here: the query
        # never returns it again, so the walk would never finish it.
        returned = {UUID(str(row[0])) for row in projects}
        progress = progress.within(returned)
        summary = LocationBackfillSummary()
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
                save_progress(engine, PROGRESS_KEY, progress=progress)
        except BackfillStopped:
            logger.warning(
                "artifact_location_backfill stopped before it finished; the next start resumes it",
                extra={"event": "artifact_location_backfill_stopped"},
            )
            return None
        for outcome, count in (
            ("recorded", summary.recorded),
            ("duplicated", len(summary.duplicated)),
            ("conflicts", summary.conflicts),
            ("unread", summary.unread),
            ("skipped", summary.skipped),
        ):
            logger.warning(
                "artifact_location_backfill finished its walk",
                extra={"event": "artifact_location_backfill_summary", "outcome": outcome, "count": count},
            )
        if not progress.unread:
            # Progress first: a crash between the two leaves neither, and the
            # next start walks again, rather than a progress row that would
            # skip projects once the sentinel is deleted for a rerun.
            delete_progress(engine, PROGRESS_KEY)
            write_sentinel(engine, SENTINEL_KEY)
    return summary
