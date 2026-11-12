"""The queued_agent_repairs rows (cowork/models/queued_agent_repair.py).

The repair route records a row before it queues a repair in the artifact's
folder, and takes the row back when the repair is refused. A turn reads the
conversation's rows when it starts; its end checks only those folders for a
repair queued to it, then drops the rows it read whose folders it checked, by
id. A row recorded after the read is a new row, so it survives that drop. The
reads and drops at turn end are best-effort: without the rows, the diff lists
and checks every folder, as it did before rows existed.
"""
from __future__ import annotations

import logging
from functools import partial
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError

from cowork.common.logger import log_context
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.db.units import run_db
from cowork.models.queued_agent_repair import QueuedAgentRepair

logger = logging.getLogger(__name__)


class QueuedRepairs(BaseModel):
    """A conversation's queued_agent_repairs rows, as a turn read them."""

    model_config = ConfigDict(frozen=True)

    # Each folder's row id.
    row_ids: dict[str, UUID] = Field(default_factory=dict)

    @property
    def slugs(self) -> set[str]:
        return set(self.row_ids)

    def ids_for(self, slugs: set[str]) -> list[UUID]:
        """The ids of the rows whose folders are among ``slugs``."""
        return [row_id for slug, row_id in self.row_ids.items() if slug in slugs]


def _rows(session: ScopedSession, *, conversation_id: UUID, slug: str | None = None) -> list[QueuedAgentRepair]:
    query = session.select(QueuedAgentRepair).where(QueuedAgentRepair.conversation_id == conversation_id)
    if slug is not None:
        query = query.where(QueuedAgentRepair.slug == slug)
    return list(session.exec(query).all())


def _replace_row(session: ScopedSession, *, conversation_id: UUID, slug: str) -> UUID | None:
    """Replace the folder's rows with a fresh one, and return its id when the
    folder had no row. Raises IntegrityError when another request inserted a
    row for the folder after this one read the rows."""
    earlier = _rows(session, conversation_id=conversation_id, slug=slug)
    for row in earlier:
        session.delete(row)
    session.flush()
    row = QueuedAgentRepair(conversation_id=conversation_id, slug=slug)
    row_id = row.id
    session.add(row)
    session.commit()
    return None if earlier else row_id


def record_queued_repair(session: ScopedSession, *, conversation_id: UUID, slug: str) -> UUID | None:
    """Record that `slug`'s folder is about to hold a repair queued to
    `conversation_id`, as a fresh row that replaces any earlier one: a turn
    that read the earlier row drops only that row, so this one survives it.

    Returns the new row's id when the folder had no row, for a refused
    request to take back (take_back_queued_repair); None when a row was there
    already, which then stands for the folder as before.

    When another request records the folder between this one's read and its
    insert, this one replaces that request's row too and returns None. A
    take-back by the other request then deletes nothing, so a row stands for
    the folder whichever request's repair is queued."""
    try:
        return _replace_row(session, conversation_id=conversation_id, slug=slug)
    except IntegrityError:
        session.rollback()
    try:
        _replace_row(session, conversation_id=conversation_id, slug=slug)
    except IntegrityError:
        session.rollback()  # a third request's row stands for the folder
    return None


def read_queued_repairs(session: ScopedSession, *, conversation_id: UUID) -> QueuedRepairs:
    return QueuedRepairs(row_ids={row.slug: row.id for row in _rows(session, conversation_id=conversation_id)})


def forget_queued_repairs(session: ScopedSession, *, conversation_id: UUID, row_ids: list[UUID]) -> None:
    rows = session.exec(
        session.select(QueuedAgentRepair).where(
            QueuedAgentRepair.conversation_id == conversation_id,
            QueuedAgentRepair.id.in_(row_ids),
        )
    ).all()
    for row in rows:
        session.delete(row)
    session.commit()


async def queued_repair_folders(*, conversation_id: UUID, scope: TenantScope) -> QueuedRepairs | None:
    """The rows of the folders holding a repair queued to `conversation_id`,
    or None when they can't be read; the diff then checks every folder."""
    try:
        return await run_db(partial(read_queued_repairs, conversation_id=conversation_id), scope=scope)
    except Exception:
        logger.warning(
            "could not read a conversation's queued repairs", exc_info=True,
            extra=log_context(conversation_id=conversation_id),
        )
        return None


async def take_back_queued_repair(*, conversation_id: UUID, row_id: UUID | None, scope: TenantScope) -> None:
    """Delete the row a refused repair request added (record_queued_repair's
    id), so a refusal leaves no row behind. Nothing for None: the folder's
    earlier row stands for it as before. A row this can't delete costs the
    conversation's next turn end one more folder to check."""
    if row_id is None:
        return
    try:
        await run_db(
            partial(forget_queued_repairs, conversation_id=conversation_id, row_ids=[row_id]), scope=scope,
        )
    except Exception:
        logger.warning(
            "could not take back a refused repair's queued row", exc_info=True,
            extra=log_context(conversation_id=conversation_id),
        )


async def forget_checked_repairs(
    *, conversation_id: UUID, repairs: QueuedRepairs | None, checked: set[str], scope: TenantScope,
) -> None:
    """Drop the rows of `repairs`, read when the turn started, whose folders
    the turn end's diff checked (ArtifactChanges.checked_repairs). A row left
    behind costs the next turn end one more folder to check."""
    row_ids = repairs.ids_for(checked) if repairs is not None else []
    if not row_ids:
        return
    try:
        await run_db(
            partial(forget_queued_repairs, conversation_id=conversation_id, row_ids=row_ids), scope=scope,
        )
    except Exception:
        logger.warning(
            "could not drop a conversation's checked queued repairs", exc_info=True,
            extra=log_context(conversation_id=conversation_id),
        )
