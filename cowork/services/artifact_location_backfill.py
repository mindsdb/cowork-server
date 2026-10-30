"""One-time backfill of artifact_locations rows (cowork/models/artifact_location.py).

Artifacts made before the table existed have no row, so their first lookup
lists the project's roots. This pass records a row for each of them, once per
environment, at cowork-server startup in org tenancy. It walks every org
project's shared root and legacy conversation roots, reads each folder's id
without writing, and records the ids held by exactly one folder. An id held by
two folders gets no row: its lookups keep scanning, and answer 409 as before.

It runs as one file store call (cowork.db.units.run_efs, no deadline), so it
holds one of the file store's slots for the whole walk. A started file store
call is never abandoned, so shutdown sets ``stop`` and the walk ends at the
next project, without its sentinel; the next start walks again. To run it
again, delete the ``_artifact_location_backfill_v1`` setting row and restart
cowork-server.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from uuid import UUID

from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.models.project import Project
from cowork.services.artifact_identity import identities_below
from cowork.services.artifact_locations import Location, record_location
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
    recorded: int = 0
    duplicated: list[str] = field(default_factory=list)
    conflicts: int = 0


def _found_locations(project: ProjectRow) -> dict[str, list[Location]]:
    """Every id in `project`'s roots and the folders that hold it."""
    found: dict[str, list[Location]] = {}
    for root in _project_roots(project):
        for artifact_id, folder_names in identities_below(root.source).items():
            for folder_name in folder_names:
                location = recorded_location(root.source, root.source.base / folder_name)
                if location is not None:
                    found.setdefault(artifact_id, []).append(location)
    return found


def _backfill_project(engine, project: ProjectRow, org_id: str, summary: LocationBackfillSummary) -> None:
    found = _found_locations(project)
    with Session(engine) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=org_id))
        for artifact_id, locations in found.items():
            if len(locations) > 1:
                summary.duplicated.append(artifact_id)
                continue
            try:
                record_location(session, artifact_id, locations[0])
                summary.recorded += 1
            except Exception:
                # Recorded at another folder already, or a failed write: the
                # lookup scans for this id and records it then.
                session.rollback()
                summary.conflicts += 1


def run_artifact_location_backfill(
    engine=None, *, try_lock=None, project_ids=None, stop: threading.Event | None = None
) -> LocationBackfillSummary | None:
    """Run the pass once per environment. None when skipped or stopped.

    Raises only when the pass cannot run at all (for example the database is
    unreachable); the sentinel is then not written and the next start retries.
    ``project_ids`` limits the pass to those projects; it exists for tests,
    whose database is shared across modules. Production passes nothing.
    """
    if get_app_settings().tenancy_mode != "org":
        return None
    engine = engine or get_engine(get_app_settings().database.uri)
    lock = try_lock or _advisory_try_lock
    if _sentinel_present(engine, SENTINEL_KEY):
        return None
    with lock(engine, _LOCK_KEY) as acquired:
        if not acquired or _sentinel_present(engine, SENTINEL_KEY):
            return None
        with Session(engine) as raw:
            query = select(Project.id, Project.name, Project.path, Project.org_id)
            if project_ids is not None:
                query = query.where(Project.id.in_(list(project_ids)))
            projects = raw.exec(query).all()
        summary = LocationBackfillSummary()
        for project_id, name, path, org_id in projects:
            if stop is not None and stop.is_set():
                logger.warning("artifact_location_backfill stopped before it finished; the next start resumes it")
                return None
            if not org_id:
                continue
            project = ProjectRow(id=UUID(str(project_id)), name=name, path=path, external=False)
            _backfill_project(engine, project, org_id, summary)
        logger.warning(
            "artifact_location_backfill recorded=%d duplicated=%d conflicts=%d",
            summary.recorded, len(summary.duplicated), summary.conflicts,
        )
        _write_sentinel(engine, SENTINEL_KEY)
    return summary
