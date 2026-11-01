"""The artifact_locations rows (cowork/models/artifact_location.py).

Org tenancy only: there `COWORK_HOME` is the shared file store, where every
listing and stat goes to the file server. Every function here is a database
step and takes a scoped session; the file store steps that find or open the
folders run on their own threads (cowork/db/units.py, run_efs).

Only a writer that checked every root of a project (a lookup's scan, the
backfill's walk) writes a row, with record_checked_folders, since a writer
that saw fewer roots could miss another folder holding the id. A row moves
with a task's folders and is deleted with its folder. An id the server sees
held by two folders of one project is marked conflicted, and stays so while
the folder its row names holds the id: its lookups list the project's roots
and answer 409 when two of its folders are in the roots that decide the
lookup (the caller's own, or every member's when the caller's own hold
none), as before rows existed. A turn end sees only the folders it made, so
it writes no row; it marks an id conflicted when the id's row names another
folder of the project (record_made_folder).
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from enum import Enum
from typing import NamedTuple
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from cowork.common.logger import log_context
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.models.artifact_location import ArtifactLocation

logger = logging.getLogger(__name__)


class Location(NamedTuple):
    """Where one artifact's folder is: the project's shared artifacts root
    when `conversation_id` is None, else that conversation's legacy root."""

    project_id: UUID
    conversation_id: UUID | None
    slug: str


class LocationRow(NamedTuple):
    """An artifact id's row: the folder it names, and whether the id is
    conflicted, held by two folders of that folder's project."""

    location: Location
    conflicted: bool


class RecordOutcome(Enum):
    """What record_checked_folders left in the table for one id."""

    #: No row, and no folder to record.
    NOTHING = "nothing"
    #: The row names the one folder holding the id.
    RECORDED = "recorded"
    #: The row marks the id conflicted.
    CONFLICTED = "conflicted"
    #: The row is gone: its folder no longer holds the id, and no folder does.
    DROPPED = "dropped"
    #: The row names a folder the writer did not find, which may still hold
    #: the id, so it stays as it was.
    KEPT = "kept"


def keeps_locations(scope: TenantScope) -> bool:
    """Whether this caller's artifacts have rows: org tenancy, with an org."""
    return bool(scope.org_mode and scope.org_id)


def _row(session: ScopedSession, artifact_id: str) -> ArtifactLocation | None:
    return session.exec(
        session.select(ArtifactLocation).where(ArtifactLocation.artifact_id == artifact_id)
    ).first()


def _location(row: ArtifactLocation) -> Location:
    return Location(row.project_id, row.conversation_id, row.slug)


def _log_conflicted(location: Location) -> None:
    logger.warning(
        "an artifact id is held by two folders of one project; its lookups list the project's roots",
        extra=log_context(project_id=location.project_id, artifact_slug=location.slug),
    )


def _insert(
    session: ScopedSession, *, artifact_id: str, location: Location, conflicted: bool,
) -> ArtifactLocation | None:
    """Add `artifact_id`'s row. None once it is written, or the row another
    writer recorded between the caller's read and this insert."""
    if conflicted:
        _log_conflicted(location)
    session.add(ArtifactLocation(
        artifact_id=artifact_id,
        project_id=location.project_id,
        conversation_id=location.conversation_id,
        slug=location.slug,
        conflicted=conflicted,
    ))
    try:
        session.commit()
        return None
    except IntegrityError:
        session.rollback()
        row = _row(session, artifact_id)
        if row is None:
            raise
        return row


def _write(session: ScopedSession, row: ArtifactLocation, *, location: Location, conflicted: bool) -> None:
    if conflicted and not row.conflicted:
        _log_conflicted(location)
    row.project_id, row.conversation_id, row.slug = location
    row.conflicted = conflicted
    session.add(row)
    session.commit()


def find_location(session: ScopedSession, artifact_id: str) -> LocationRow | None:
    """The row of `artifact_id` in the caller's organization."""
    if not keeps_locations(session.scope):
        return None
    row = _row(session, artifact_id)
    return LocationRow(_location(row), row.conflicted) if row is not None else None


def record_made_folder(session: ScopedSession, *, artifact_id: str, location: Location) -> None:
    """Note that a turn made `location`, a folder holding `artifact_id`.

    A turn sees only the folders it made, and an older folder without a row
    may hold the same id, so it writes no row: the id's first lookup lists
    every root of the project and records what it finds. When the id's row
    already names another folder of `location`'s project, two folders hold
    the id, so the row is marked conflicted.
    """
    if not keeps_locations(session.scope):
        return
    row = _row(session, artifact_id)
    if row is None or row.conflicted:
        return
    current = _location(row)
    if current != location and current.project_id == location.project_id:
        _write(session, row, location=current, conflicted=True)


def record_checked_folders(
    session: ScopedSession,
    *,
    artifact_id: str,
    folders: Sequence[Location],
    replaces: Location | None = None,
) -> RecordOutcome:
    """Record what a writer found after it checked every root of one project
    for `artifact_id`: `folders`, every folder there holding the id.

    An id with no row gets one: for its one folder, or marked conflicted for
    two or more. An existing row keeps the folder it names unless the writer
    has just found that folder no longer holds the id and says so in
    `replaces`: then the row moves to what the writer found, or is dropped
    when it found nothing. Two folders found also mark a row conflicted when
    it names a folder of the same project. Otherwise the row stays as it was,
    since the folder it names may still hold the id: one in another project,
    or one the writer could not read. A conflicted row stays conflicted while
    the folder it names holds the id.
    """
    if not keeps_locations(session.scope):
        return RecordOutcome.NOTHING
    row = _row(session, artifact_id)
    if row is None:
        if not folders:
            return RecordOutcome.NOTHING
        conflicted = len(folders) > 1
        row = _insert(session, artifact_id=artifact_id, location=folders[0], conflicted=conflicted)
        if row is None:
            return RecordOutcome.CONFLICTED if conflicted else RecordOutcome.RECORDED
    current = _location(row)
    stale = replaces is not None and current == replaces
    if not folders:
        if not stale:
            return RecordOutcome.KEPT
        session.delete(row)
        session.commit()
        return RecordOutcome.DROPPED
    if len(folders) > 1:
        if stale:
            _write(session, row, location=folders[0], conflicted=True)
        elif current.project_id == folders[0].project_id:
            if not row.conflicted:
                _write(session, row, location=current, conflicted=True)
        else:
            return RecordOutcome.KEPT
        return RecordOutcome.CONFLICTED
    if current == folders[0]:
        return RecordOutcome.CONFLICTED if row.conflicted else RecordOutcome.RECORDED
    if not stale:
        return RecordOutcome.KEPT
    _write(session, row, location=folders[0], conflicted=False)
    return RecordOutcome.RECORDED


def move_locations(
    session: ScopedSession, *, project_id: UUID, moved: dict[str, str], to_project_id: UUID,
) -> None:
    """Follow folders moved from `project_id`'s shared root to `to_project_id`'s:
    `moved` maps each old slug to its new one."""
    if not keeps_locations(session.scope) or not moved:
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
    id. So does a conflicted row, since another folder of the project may
    still hold it; the next lookup's scan decides."""
    if not keeps_locations(session.scope):
        return
    row = _row(session, artifact_id)
    if row is not None and not row.conflicted and _location(row) == location:
        session.delete(row)
        session.commit()
