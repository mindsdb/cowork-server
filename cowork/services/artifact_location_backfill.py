"""One-time backfill of artifact_locations rows (cowork/models/artifact_location.py).

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
Only a folder that is gone, is a link, or holds no metadata that parses
holds no id (cowork.services.artifact_identity.identities_below).

It runs as one file store call (cowork.db.units.run_efs, no deadline), so it
holds one of the file store's slots for the whole walk. A started file store
call is never abandoned, so shutdown sets ``stop`` and the walk ends at the
next root or id. The sentinel is written only once every project was read
and recorded: a stop, a project it could not read, or a database error other
than an integrity violation leaves it unwritten, and the next start walks
again. To run it again, delete the ``_artifact_location_backfill_v1``
setting row and restart cowork-server.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from cowork.common.logger import log_context
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.models.project import Project
from cowork.services.artifact_identity import identities_below
from cowork.services.artifact_locations import Location, RecordOutcome, record_checked_folders
from cowork.services.artifact_owner_backfill import (
    _advisory_try_lock,
    _sentinel_present,
    _write_sentinel,
)
from cowork.services.artifact_roots import ProjectRow, _project_roots, recorded_location
from cowork.services.shared_resources import advisory_lock_key

logger = logging.getLogger(__name__)

SENTINEL_KEY = "_artifact_location_backfill_v1"
_LOCK_KEY = advisory_lock_key(b"cowork\0artifact_location_backfill_v1")


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


class _Stopped(Exception):
    """``stop`` was set during the walk."""


def _check_stop(stop: threading.Event | None) -> None:
    if stop is not None and stop.is_set():
        raise _Stopped


def _found_locations(project: ProjectRow, *, stop: threading.Event | None) -> dict[str, list[Location]]:
    """Every id in `project`'s roots and the folders that hold it. Raises
    OSError when a root or a folder cannot be read."""
    found: dict[str, list[Location]] = {}
    for root in _project_roots(project, strict=True):
        _check_stop(stop)
        for artifact_id, folder_names in identities_below(root.source).items():
            for folder_name in folder_names:
                location = recorded_location(root.source, root.source.base / folder_name)
                if location is not None:
                    found.setdefault(artifact_id, []).append(location)
    return found


def _record_project(
    engine,
    *,
    org_id: str,
    found: dict[str, list[Location]],
    summary: LocationBackfillSummary,
    stop: threading.Event | None,
) -> None:
    with Session(engine) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=org_id))
        for artifact_id, locations in found.items():
            _check_stop(stop)
            try:
                # The walk read every root of the project.
                outcome = record_checked_folders(session, artifact_id=artifact_id, folders=locations)
            except IntegrityError:
                # The folder's project or conversation was deleted during the
                # walk. Any other error ends the pass.
                session.rollback()
                summary.conflicts += 1
                continue
            if len(locations) > 1:
                summary.duplicated.append(artifact_id)
            elif outcome is RecordOutcome.RECORDED:
                summary.recorded += 1
            else:
                summary.conflicts += 1


def run_artifact_location_backfill(
    engine=None, *, try_lock=None, project_ids=None, stop: threading.Event | None = None
) -> LocationBackfillSummary | None:
    """Run the pass once per environment. None when skipped or stopped.

    Raises when the pass cannot run or a write fails (for example the
    database is unreachable, or no pooled connection frees in time); the
    sentinel is then not written and the next start retries. A project it
    could not read is counted in ``unread`` and leaves the sentinel unwritten
    too. Recording a folder again changes nothing. ``project_ids`` limits the
    pass to those projects; it exists for tests, whose database is shared
    across modules. Production passes nothing.
    """
    if get_app_settings().tenancy_mode != "org":
        return None
    engine = engine or get_engine(get_app_settings().database.uri)
    lock = try_lock or _advisory_try_lock
    if _sentinel_present(engine, SENTINEL_KEY):
        return None
    with lock(engine, _LOCK_KEY) as acquired:
        if not acquired:
            return None
        if _sentinel_present(engine, SENTINEL_KEY):
            return None
        with Session(engine) as raw:
            query = select(Project.id, Project.name, Project.path, Project.org_id)
            if project_ids is not None:
                query = query.where(Project.id.in_(list(project_ids)))
            projects = raw.exec(query).all()
        summary = LocationBackfillSummary()
        try:
            for project_id, name, path, org_id in projects:
                _check_stop(stop)
                if not org_id:
                    continue
                project = ProjectRow(id=UUID(str(project_id)), name=name, path=path, external=False)
                try:
                    found = _found_locations(project, stop=stop)
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
                    continue
                _record_project(engine, org_id=org_id, found=found, summary=summary, stop=stop)
        except _Stopped:
            logger.warning("artifact_location_backfill stopped before it finished; the next start resumes it")
            return None
        for outcome, count in (
            ("recorded", summary.recorded),
            ("duplicated", len(summary.duplicated)),
            ("conflicts", summary.conflicts),
            ("unread", summary.unread),
        ):
            logger.warning(
                "artifact_location_backfill finished its walk",
                extra={"event": "artifact_location_backfill_summary", "outcome": outcome, "count": count},
            )
        if not summary.unread:
            _write_sentinel(engine, SENTINEL_KEY)
    return summary
