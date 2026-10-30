"""The artifact_locations rows (cowork/models/artifact_location.py).

Org tenancy only: there `COWORK_HOME` is the shared file store, where every
listing and stat goes to the file server. Every function here is a database
step and takes a scoped session; the file store steps that find or open the
folders run on their own threads (cowork/db/units.py, run_efs).

A row is written where the server makes, finds or moves an artifact folder
and deleted with the folder. A second folder claiming an id that already has
a row is a conflict, raised when that folder is recorded: the first row stays,
and lookups keep opening the first folder.
"""
from __future__ import annotations

import logging
from typing import NamedTuple
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from cowork.db.scoped import ScopedSession
from cowork.models.artifact_location import ArtifactLocation
from cowork.services.artifact_identity import ArtifactIdentityConflict

logger = logging.getLogger(__name__)


class Location(NamedTuple):
    """Where one artifact's folder is: the project's shared artifacts root
    when `conversation_id` is None, else that conversation's legacy root."""

    project_id: UUID
    conversation_id: UUID | None
    slug: str


def keeps_locations(session: ScopedSession) -> bool:
    """Whether this caller's artifacts have rows: org tenancy, with an org."""
    return bool(session.scope.org_mode and session.scope.org_id)


def _row(session: ScopedSession, artifact_id: str) -> ArtifactLocation | None:
    return session.exec(
        session.select(ArtifactLocation).where(ArtifactLocation.artifact_id == artifact_id)
    ).first()


def _location(row: ArtifactLocation) -> Location:
    return Location(row.project_id, row.conversation_id, row.slug)


def find_location(session: ScopedSession, artifact_id: str) -> Location | None:
    """The recorded folder of `artifact_id` in the caller's organization."""
    if not keeps_locations(session):
        return None
    row = _row(session, artifact_id)
    return _location(row) if row is not None else None


def record_location(
    session: ScopedSession, artifact_id: str, location: Location, *, replaces: Location | None = None,
) -> None:
    """Record that `artifact_id`'s folder is at `location`.

    An id with no row gets one, and recording the same place again changes
    nothing. An id whose row names another place raises
    ArtifactIdentityConflict and leaves that row as it was, unless the caller
    has just found the folder that row names no longer holds the id and says
    so in `replaces`; then the row moves to `location`.
    """
    if not keeps_locations(session):
        return
    row = _row(session, artifact_id)
    if row is None:
        session.add(ArtifactLocation(
            artifact_id=artifact_id,
            project_id=location.project_id,
            conversation_id=location.conversation_id,
            slug=location.slug,
        ))
        try:
            session.commit()
            return
        except IntegrityError:
            # Another writer recorded the id between the read and the insert.
            session.rollback()
            row = _row(session, artifact_id)
            if row is None:
                raise
    current = _location(row)
    if current == location:
        return
    if replaces is None or current != replaces:
        raise ArtifactIdentityConflict("Artifact identity is already recorded at another folder")
    row.project_id, row.conversation_id, row.slug = location
    session.add(row)
    session.commit()


def move_locations(
    session: ScopedSession, *, project_id: UUID, moved: dict[str, str], to_project_id: UUID,
) -> None:
    """Follow folders moved from `project_id`'s shared root to `to_project_id`'s:
    `moved` maps each old slug to its new one."""
    if not keeps_locations(session) or not moved:
        return
    rows = session.exec(
        session.select(ArtifactLocation).where(
            ArtifactLocation.project_id == project_id,
            ArtifactLocation.conversation_id.is_(None),
            ArtifactLocation.slug.in_(list(moved)),
        )
    ).all()
    for row in rows:
        row.project_id, row.slug = to_project_id, moved[row.slug]
        session.add(row)
    session.commit()


def forget_location(session: ScopedSession, artifact_id: str, location: Location) -> None:
    """Delete `artifact_id`'s row when it names `location`, the folder just
    deleted. A row naming another folder stays: that folder still holds the
    id."""
    if not keeps_locations(session):
        return
    row = _row(session, artifact_id)
    if row is not None and _location(row) == location:
        session.delete(row)
        session.commit()


def record_found_location(
    session: ScopedSession, artifact_id: str, location: Location, *, replaces: Location | None = None,
) -> None:
    """record_location for a folder a lookup or a turn found, where a conflict
    is logged rather than raised: the caller already has its answer."""
    try:
        record_location(session, artifact_id, location, replaces=replaces)
    except ArtifactIdentityConflict:
        logger.warning(
            "artifact id %s is recorded at another folder; the folder found in project %s stays unrecorded",
            artifact_id, location.project_id,
        )
