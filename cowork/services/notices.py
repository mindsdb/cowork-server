"""Each person's in-app notification inbox.

A feature posts a notice for one person (``post_notice``); the app lists them
under its bell, newest first, with an unread count, and marks them read. A
person only ever sees their own: the scoped session keeps the org, and every
query here adds the person, as schedules do.

Notices are kept for ``RETENTION`` and at most ``MAX_PER_PERSON`` each; older
ones go when a new one is posted, so there is no sweeper to run.

Scheduled runs are the first source (``announce_scheduled_run``). A run that
didn't finish posts a notice. A finished run posts nothing yet: the per-account
choice to hear about those isn't stored on the server yet.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Any
from uuid import UUID

import sqlalchemy as sa

from cowork.common.logger import get_logger
from cowork.db.scoped import ScopedSession, scope_for_identity
from cowork.models.notice import Notice

logger = get_logger(__name__)

RETENTION = timedelta(days=30)
MAX_PER_PERSON = 200
TITLE_MAX = 255

SCHEDULED_RUN_FAILED = "scheduled_run.failed"


class NoticeService:
    def __init__(self, session: ScopedSession) -> None:
        self.session = session

    def _owner_filter(self, stmt: Any) -> Any:
        # The scoped session keeps the org; the person is ours to add. No
        # person in scope means the desktop, where there is one.
        user_id = self.session.scope.user_id
        return stmt.where(Notice.user_id == user_id) if user_id else stmt

    def list_notices(self, *, limit: int = 50) -> list[Notice]:
        stmt = self._owner_filter(self.session.select(Notice))
        return list(self.session.exec(stmt.order_by(Notice.created_at.desc()).limit(limit)).all())

    def unread_count(self) -> int:
        stmt = self._owner_filter(self.session.select(Notice)).where(Notice.read_at.is_(None))
        return self.session.exec(stmt.with_only_columns(sa.func.count(Notice.id))).one()

    def mark_read(self, notice_id: UUID) -> bool:
        stmt = self._owner_filter(self.session.select(Notice)).where(Notice.id == notice_id)
        notice = self.session.exec(stmt).first()
        if notice is None:
            return False
        if notice.read_at is None:
            notice.read_at = datetime.now(timezone.utc)
            self.session.add(notice)
            self.session.commit()
        return True

    def mark_all_read(self) -> int:
        stmt = self._owner_filter(self.session.select(Notice)).where(Notice.read_at.is_(None))
        now = datetime.now(timezone.utc)
        unread = list(self.session.exec(stmt).all())
        for notice in unread:
            notice.read_at = now
            self.session.add(notice)
        self.session.commit()
        return len(unread)

    def post(self, *, kind: str, title: str, data: dict[str, Any]) -> Notice:
        """Post a notice for the person in scope, and drop their oldest past
        the limits."""
        notice = Notice(
            user_id=self.session.scope.user_id, kind=kind, title=title[:TITLE_MAX], data=data,
        )
        self.session.add(notice)
        self.session.flush()
        self._prune()
        self.session.commit()
        return notice

    def _prune(self) -> None:
        stmt = self._owner_filter(self.session.select(Notice)).order_by(Notice.created_at.desc())
        cutoff = datetime.now(timezone.utc) - RETENTION
        for index, notice in enumerate(self.session.exec(stmt).all()):
            created = notice.created_at
            if created is not None and created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            if index >= MAX_PER_PERSON or (created is not None and created < cutoff):
                self.session.delete(notice)


def post_notice(
    session: ScopedSession, *, kind: str, title: str, data: dict[str, Any],
) -> None:
    NoticeService(session).post(kind=kind, title=title, data=data)


async def announce_scheduled_run(
    *, org_id: str | None, user_id: str | None, schedule_id: UUID, run_id: UUID | None,
    conversation_id: UUID | str | None, schedule_name: str, outcome: str | None,
) -> None:
    """Post a notice to a schedule's owner that its run didn't finish. Never
    raises: a notice never fails a run."""
    from cowork.db.units import run_db

    data = {
        "schedule_id": str(schedule_id),
        "run_id": str(run_id) if run_id else None,
        "conversation_id": str(conversation_id) if conversation_id else None,
        "outcome": outcome,
    }
    try:
        await run_db(
            partial(post_notice, kind=SCHEDULED_RUN_FAILED, title=schedule_name, data=data),
            scope=scope_for_identity(org_id=org_id, user_id=user_id),
        )
    except Exception:
        logger.warning(
            "Could not post a scheduled run's notice", exc_info=True,
            extra={"event": "notice_post_failed", "schedule_id": str(schedule_id)},
        )
