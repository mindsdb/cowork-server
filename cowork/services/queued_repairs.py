"""The queued_agent_repairs rows (cowork/models/queued_agent_repair.py).

The repair route records a row before it queues a repair in the artifact's
folder. The end of the conversation's turn reads the rows, checks only those
folders for a repair queued to it, and drops the rows. The reads and drops at
turn end are best-effort: without the rows, the diff lists and checks every
folder, as it did before rows existed.
"""
from __future__ import annotations

import logging
from functools import partial
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from cowork.db.scoped import ScopedSession, TenantScope
from cowork.db.units import run_db
from cowork.models.queued_agent_repair import QueuedAgentRepair

logger = logging.getLogger(__name__)


def record_queued_repair(session: ScopedSession, *, conversation_id: UUID, slug: str) -> None:
    """Record that `slug`'s folder is about to hold a repair queued to
    `conversation_id`. Recording it twice changes nothing."""
    exists = session.exec(
        session.select(QueuedAgentRepair).where(
            QueuedAgentRepair.conversation_id == conversation_id, QueuedAgentRepair.slug == slug,
        )
    ).first()
    if exists is not None:
        return
    session.add(QueuedAgentRepair(conversation_id=conversation_id, slug=slug))
    try:
        session.commit()
    except IntegrityError:
        session.rollback()  # another request recorded it first


def queued_repair_slugs(session: ScopedSession, *, conversation_id: UUID) -> set[str]:
    return {
        row.slug
        for row in session.exec(
            session.select(QueuedAgentRepair).where(QueuedAgentRepair.conversation_id == conversation_id)
        ).all()
    }


def forget_queued_repairs(session: ScopedSession, *, conversation_id: UUID, slugs: set[str]) -> None:
    rows = session.exec(
        session.select(QueuedAgentRepair).where(
            QueuedAgentRepair.conversation_id == conversation_id,
            QueuedAgentRepair.slug.in_(list(slugs)),
        )
    ).all()
    for row in rows:
        session.delete(row)
    session.commit()


async def queued_repair_folders(conversation_id, *, scope: TenantScope) -> set[str] | None:
    """The folders holding a repair queued to `conversation_id`, or None when
    the rows can't be read; the diff then checks every folder."""
    try:
        return await run_db(
            partial(queued_repair_slugs, conversation_id=UUID(str(conversation_id))), scope=scope
        )
    except Exception:
        logger.warning("could not read the queued repairs of conversation %s", conversation_id, exc_info=True)
        return None


async def forget_checked_repairs(conversation_id, slugs: set[str] | None, *, scope: TenantScope) -> None:
    """Drop the rows a turn end has checked. A row left behind costs the next
    turn end one more folder to check."""
    if not slugs:
        return
    try:
        await run_db(
            partial(forget_queued_repairs, conversation_id=UUID(str(conversation_id)), slugs=set(slugs)),
            scope=scope,
        )
    except Exception:
        logger.warning("could not drop the queued repairs of conversation %s", conversation_id, exc_info=True)
