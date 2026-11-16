"""Each person's notification inbox.

Features post notices; the API lists them, counts the unread ones and marks
them read. A notice stays until it is pruned: notices older than
``RETENTION`` go, and so do all but the newest ``KEEP_PER_PERSON`` of a
person's notices. Pruning happens when a notice is posted, so an inbox
nobody posts to keeps what it has.

Scheduled runs are the first source (``post_scheduled_run_notice``). The
notice names the task, never why it failed: the reason can carry details
from the run, and notices are shown in places others may see, such as the
operating system's notification.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlmodel import Session, select

from cowork.db.scoped import ScopedSession, scope_for_identity, unsafe_unscoped_session
from cowork.models.notice import Notice
from cowork.models.schedule import Schedule, ScheduleRun
from cowork.schemas.schedules import AttemptOutcome, RunStatus

logger = logging.getLogger(__name__)

RETENTION = timedelta(days=30)
KEEP_PER_PERSON = 200

SCHEDULED_RUN_FAILED = "scheduled_run.failed"
SCHEDULED_RUN_STOPPED = "scheduled_run.stopped"
SCHEDULED_RUN_SUCCEEDED = "scheduled_run.succeeded"
_SCHEDULED_RUN_KINDS = (SCHEDULED_RUN_FAILED, SCHEDULED_RUN_STOPPED, SCHEDULED_RUN_SUCCEEDED)

# The outcome a run that ended at a step or token limit records. Read as a
# string so the inbox doesn't depend on when that outcome is added.
_STOPPED_AT_LIMIT = "stopped_at_limit"


def _owner_clause(org_id: str | None, user_id: str | None) -> list[Any]:
    """Match one person's notices, NULL ids included (desktop rows)."""
    org = Notice.org_id.is_(None) if org_id is None else Notice.org_id == org_id  # type: ignore[union-attr]
    user = Notice.user_id.is_(None) if user_id is None else Notice.user_id == user_id  # type: ignore[union-attr]
    return [org, user]


def post_notice(
    session: Session, *, org_id: str | None, user_id: str | None, kind: str, subject: str,
    target: dict[str, Any], source_id: str | None = None, now: datetime | None = None,
) -> Notice:
    """Add a notice to one person's inbox and prune it. Commits.

    Stamps ``created_at`` here rather than leaving it to the database, whose
    clock may tick in whole seconds: the inbox orders by it, and pruning
    must never take the notice just posted."""
    now = now or datetime.now(timezone.utc)
    notice = Notice(
        org_id=org_id, user_id=user_id, kind=kind, subject=(subject or "")[:255],
        target=target, source_id=source_id, created_at=now,
    )
    session.add(notice)
    session.flush()
    _prune(session, org_id=org_id, user_id=user_id, now=now)
    session.commit()
    session.refresh(notice)
    return notice


def _prune(session: Session, *, org_id: str | None, user_id: str | None, now: datetime) -> None:
    """Drop one person's notices past RETENTION, then all but the newest
    KEEP_PER_PERSON."""
    owner = _owner_clause(org_id, user_id)
    session.execute(
        sa.delete(Notice)
        .where(*owner, Notice.created_at < now - RETENTION)  # type: ignore[operator]
        .execution_options(synchronize_session=False)
    )
    keep = (
        sa.select(Notice.id)
        .where(*owner)
        .order_by(Notice.created_at.desc(), Notice.id.desc())  # type: ignore[union-attr]
        .limit(KEEP_PER_PERSON)
    )
    session.execute(
        sa.delete(Notice)
        .where(*owner, Notice.id.not_in(keep.scalar_subquery()))  # type: ignore[union-attr]
        .execution_options(synchronize_session=False)
    )


def _scheduled_run_kind(run: ScheduleRun) -> str | None:
    """What a finished run's notice says happened, or None for a run that
    posts nothing: a trial (its author is watching it), one still running,
    and one the user cancelled."""
    if run.trial or run.status == RunStatus.running:
        return None
    if run.outcome == AttemptOutcome.cancelled:
        return None
    if run.outcome == _STOPPED_AT_LIMIT:
        return SCHEDULED_RUN_STOPPED
    if run.status == RunStatus.success:
        return SCHEDULED_RUN_SUCCEEDED
    return SCHEDULED_RUN_FAILED


def _wants(session: Session, schedule: Schedule, kind: str) -> bool:
    """Whether the schedule's owner asked to hear about this kind of ending.
    Failures and stops are on unless turned off; successes are off unless
    turned on."""
    from cowork.services.settings import SettingService

    settings = SettingService(
        session, scope_for_identity(org_id=schedule.org_id, user_id=schedule.created_by),
    ).load()
    if kind == SCHEDULED_RUN_SUCCEEDED:
        return settings.schedule_notify_succeeded
    return settings.schedule_notify_failed


def post_scheduled_run_notice(scoped: ScopedSession, *, run_id: UUID) -> Notice | None:
    """Post the notice a finished scheduled run calls for, at most once per
    run. A database unit: run it with SYSTEM_SCOPE, since it reads the run
    and its schedule wherever they are and writes to the schedule owner's
    inbox."""
    session = unsafe_unscoped_session(scoped)
    run = session.get(ScheduleRun, run_id)
    if run is None:
        return None
    kind = _scheduled_run_kind(run)
    if kind is None:
        return None
    schedule = session.get(Schedule, run.schedule_id)
    if schedule is None:
        return None
    already = session.exec(
        select(Notice.id).where(
            Notice.source_id == str(run.id),
            Notice.kind.in_(_SCHEDULED_RUN_KINDS),  # type: ignore[attr-defined]
        )
    ).first()
    if already is not None or not _wants(session, schedule, kind):
        return None
    return post_notice(
        session, org_id=schedule.org_id, user_id=schedule.created_by, kind=kind,
        subject=schedule.title or "",
        target={
            "scheduleId": str(schedule.id),
            "runId": str(run.id),
            "conversationId": str(run.conversation_id) if run.conversation_id else None,
        },
        source_id=str(run.id),
    )


class NoticeService:
    """The caller's own inbox. The scoped session bounds it to their
    organization or partition; the user filter is applied here, as user_id
    has no automatic scoping."""

    def __init__(self, session: ScopedSession) -> None:
        self.session = session

    def _own(self, stmt):
        user_id = self.session.scope.user_id
        if user_id is not None:
            stmt = stmt.where(Notice.user_id == user_id)
        return stmt

    def list_notices(self, limit: int = 50) -> list[Notice]:
        stmt = self._own(self.session.select(Notice)).order_by(
            Notice.created_at.desc(), Notice.id.desc(),  # type: ignore[union-attr]
        ).limit(limit)
        return list(self.session.exec(stmt).all())

    def unread_count(self) -> int:
        stmt = self._own(
            self.session.select(Notice).where(Notice.read_at.is_(None))  # type: ignore[union-attr]
        )
        return len(self.session.exec(stmt).all())

    def mark_read(self, notice_id: UUID) -> bool:
        notice = self.session.exec(self._own(self.session.select(Notice).where(Notice.id == notice_id))).first()
        if notice is None:
            return False
        if notice.read_at is None:
            notice.read_at = datetime.now(timezone.utc)
            self.session.add(notice)
            self.session.commit()
        return True

    def mark_all_read(self) -> int:
        unread = self.session.exec(
            self._own(self.session.select(Notice).where(Notice.read_at.is_(None)))  # type: ignore[union-attr]
        ).all()
        now = datetime.now(timezone.utc)
        for notice in unread:
            notice.read_at = now
            self.session.add(notice)
        self.session.commit()
        return len(unread)
