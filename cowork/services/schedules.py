from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from datetime import timezone as timezone_
from enum import Enum
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from cowork.common.logger import get_logger
from cowork.db.scoped import ScopedSession, unsafe_unscoped_session
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.models.schedule import (
    Schedule,
    ScheduleEvent,
    ScheduleOccurrence,
    ScheduleRun,
    ScheduleSkippedPeriod,
    ScheduleVersion,
)
from cowork.schedule_timing import normalize_next_run
from cowork.schemas.schedules import AttemptOutcome, Cadence, OccurrenceStatus, RunStatus


logger = get_logger(__name__)

# The settings a version holds: an edit that changes any of them makes a new
# version. A rename, a new start time or a pause does not.
VERSION_FIELDS = ("prompt", "model", "project_id", "cadence", "timezone")


def record_schedule_event(session: ScopedSession, schedule_id: UUID, kind: str, detail: dict | None = None) -> None:
    """Stage a lifecycle event. The actor is the request's user, or NULL when
    the scheduler itself acts (or on the desktop, which has no user id)."""
    session.add(ScheduleEvent(
        schedule_id=schedule_id, kind=kind, actor=getattr(session.scope, "user_id", None), detail=detail,
    ))


def _new_version(session: ScopedSession, schedule: Schedule, number: int) -> ScheduleVersion:
    version = ScheduleVersion(
        schedule_id=schedule.id, number=number, prompt=schedule.prompt, model=schedule.model,
        project_id=schedule.project_id, cadence=schedule.cadence, timezone=schedule.timezone or "UTC",
    )
    session.add(version)
    return version


def current_version(session: ScopedSession, schedule: Schedule) -> ScheduleVersion:
    """The schedule's latest version, staged from its fields first if it has
    none (a row written before versions, or by hand). Read in the caller's
    transaction, so a claim holding the schedule's row lock sees the version
    of the last committed edit, never one half-written."""
    version = unsafe_unscoped_session(session).execute(
        sa.select(ScheduleVersion)
        .where(ScheduleVersion.schedule_id == schedule.id)
        .order_by(ScheduleVersion.number.desc())  # type: ignore[attr-defined]
        .limit(1)
    ).scalars().first()
    if version is None:
        version = _new_version(session, schedule, 1)
        session.flush()
    return version


class ScheduleService:
    def __init__(self, session: ScopedSession) -> None:
        self.session = session

    def _default_project_id(self) -> UUID | None:
        """The caller's default project. Imported lazily: projects imports this
        module's models, so a top-level import would cycle."""
        from cowork.services.projects import ProjectService
        return ProjectService(self.session).default_project_id()

    def _owned_select(self):
        """Schedules are personal. The scoped session enforces the org, but
        user_id has no automatic scoping (see PinService), so a member could
        otherwise list/read/pause/delete another member's schedules. The
        scheduler runs under LOCAL_SCOPE (org_mode False), so it is unaffected
        and still scans every org's schedules."""
        query = self.session.select(Schedule)
        if self.session.scope.org_mode:
            query = query.where(Schedule.created_by == self.session.scope.user_id)
        return query

    def _owned(self, schedule_id: UUID) -> "Schedule | None":
        return self.session.exec(
            self._owned_select().where(Schedule.id == schedule_id)
        ).first()

    def list_schedules(self, project_id: UUID | None = None) -> list[Schedule]:
        query = self._owned_select()
        if project_id is not None:
            query = query.where(Schedule.project_id == project_id)
        return list(self.session.exec(query).all())

    def due(self, now: datetime) -> list[Schedule]:
        """Enabled schedules whose slot has come, read through the
        (enabled, next_run_at) index instead of every schedule."""
        return list(
            self.session.exec(
                self._owned_select()
                .where(Schedule.enabled == True)  # noqa: E712
                .where(Schedule.next_run_at <= now)
            ).all()
        )

    def events(self, schedule_id: UUID, limit: int = 100) -> list[ScheduleEvent]:
        if self._owned(schedule_id) is None:
            return []
        return list(
            self.session.exec(
                self.session.select(ScheduleEvent)
                .where(ScheduleEvent.schedule_id == schedule_id)
                .order_by(ScheduleEvent.created_at.desc())  # type: ignore[union-attr]
                .limit(limit)
            ).all()
        )

    def current_version_number(self, schedule_id: UUID) -> int | None:
        return unsafe_unscoped_session(self.session).execute(
            sa.select(sa.func.max(ScheduleVersion.number)).where(ScheduleVersion.schedule_id == schedule_id)
        ).scalar()

    def get_schedule(self, schedule_id: UUID) -> Schedule:
        schedule = self._owned(schedule_id)
        if schedule is None:
            raise ValueError("Schedule not found")
        return schedule

    def create_schedule(
        self,
        title: str,
        prompt: str,
        cadence: str,
        next_run_at: datetime,
        model: str,
        timezone: str = "UTC",
        project_id: UUID | None = None,
        enabled: bool = True,
    ) -> Schedule:
        # Anchor the parent (same as conversations): the target project must be
        # visible in scope, or a foreign org's project id could be attached.
        target_project_id = project_id or self._default_project_id()
        if self.session.get(Project, target_project_id) is None:
            raise ValueError("Project not found")
        schedule = Schedule(
            title=title,
            prompt=prompt,
            cadence=cadence,
            next_run_at=normalize_next_run(cadence, next_run_at, timezone),
            model=model,
            timezone=timezone,
            project_id=target_project_id,
            enabled=enabled,
            paused_at=None if enabled else datetime.now(timezone_.utc),
        )
        self.session.add(schedule)
        self.session.flush()
        _new_version(self.session, schedule, 1)
        record_schedule_event(self.session, schedule.id, "created", {"version": 1})
        self.session.commit()
        self.session.refresh(schedule)
        return schedule

    def update_schedule(self, schedule_id: UUID, **kwargs) -> Schedule:
        """Apply an edit. ``enabled`` pauses or resumes, as those actions do;
        the editor sends the current value with every save, which changes
        nothing."""
        schedule = self.get_schedule(schedule_id)
        new_project_id = kwargs.get("project_id")
        if new_project_id is not None and self.session.get(Project, new_project_id) is None:
            raise ValueError("Project not found")
        enabled = kwargs.pop("enabled", None)
        # Compared with the state before the edit: a reschedule below can
        # turn a missed one-off back on, and the stale `enabled` the editor
        # sent must not pause it again.
        was_on = self._runs_on_its_own(schedule)
        given = {k for k, v in kwargs.items() if v is not None}
        changed = sorted(
            field for field in given
            if hasattr(schedule, field) and getattr(schedule, field) != kwargs[field]
        )
        for field in given:
            if hasattr(schedule, field):
                setattr(schedule, field, kwargs[field])
        if any(field in VERSION_FIELDS for field in changed):
            # Written in the edit's own commit, so a claim reads the old
            # version or this one, never a mix of the two.
            number = (self.current_version_number(schedule.id) or 0) + 1
            _new_version(self.session, schedule, number)
        if changed:
            record_schedule_event(
                self.session, schedule.id, "edited",
                {"fields": changed, "version": self.current_version_number(schedule.id)},
            )
        if {"cadence", "timezone", "next_run_at"} & given:
            # The slot follows the edit at once, rather than firing on the old
            # cadence or reading a start already in the past as a missed run.
            schedule.next_run_at = normalize_next_run(schedule.cadence, schedule.next_run_at, schedule.timezone)
            if (
                schedule.cadence == Cadence.once
                and schedule.missed_at is not None
                and _utc(schedule.next_run_at) > datetime.now(timezone_.utc)
            ):
                # A missed one-off given a new time is scheduled again.
                schedule.missed_at = None
                schedule.enabled = schedule.paused_at is None and schedule.auto_paused_at is None
        self.session.add(schedule)
        if enabled is not None and enabled != was_on:
            return self._pause(schedule) if not enabled else self._resume(schedule)
        self.session.commit()
        self.session.refresh(schedule)
        return schedule

    @staticmethod
    def _runs_on_its_own(schedule: Schedule) -> bool:
        return schedule.paused_at is None and schedule.auto_paused_at is None and schedule.enabled

    def delete_schedule(self, schedule_id: UUID) -> bool:
        schedule = self._owned(schedule_id)
        if schedule is None:
            return False
        # Runs are removed by the Schedule.runs delete-orphan cascade. Deleting
        # them by hand here left the parent DELETE to be emitted first (no
        # relationship told the unit of work otherwise), which trips the
        # schedule_runs foreign key on Postgres — a raw 500 the desktop's
        # unenforced SQLite FKs hid.
        self.session.delete(schedule)
        self.session.commit()
        return True

    def release_conversation(self, conversation_id: UUID) -> None:
        """Let go of a conversation that is being deleted, keeping the rows.

        Two columns point at a conversation: a run records the one it produced,
        and a schedule records the one its last run produced. Neither is the
        run's own data, and both are nullable, so tidying a chat releases the
        link and leaves the run history it is an audit trail of. Every reader
        tolerates the empty pointer: both response schemas declare the fields
        `UUID | None`, and the desktop already renders a run with no
        conversation.

        Staged into the caller's transaction, never committed here: whoever is
        deleting the conversation commits once, so a crash cannot leave a
        conversation whose contents are gone.

        Keyed on the conversation id alone, and run on the raw session on
        purpose. Both `select` helpers would narrow this: the scoped layer adds
        an org filter for `Schedule`, and `_owned_select` adds an owner filter
        on top. A conversation id is already the narrowest possible key, and a
        project delete cascades conversations belonging to several members, so
        either filter can only hide a row that still has to be released. Hiding
        one puts the foreign-key violation straight back. A core UPDATE also
        keeps the flush hook out of it, which would otherwise adopt a row whose
        org_id is NULL into whoever triggered the delete.
        """
        raw = unsafe_unscoped_session(self.session)
        raw.execute(
            sa.update(ScheduleRun)
            .where(ScheduleRun.conversation_id == conversation_id)
            .values(conversation_id=None)
        )
        raw.execute(
            sa.update(Schedule)
            .where(Schedule.last_result_conversation_id == conversation_id)
            .values(last_result_conversation_id=None)
        )

    def pause_schedule(self, schedule_id: UUID) -> Schedule:
        return self._pause(self.get_schedule(schedule_id))

    def _pause(self, schedule: Schedule) -> Schedule:
        """Stop everything that would start on its own: future slots, a retry
        waiting to run, and a slot claimed but not yet started. An attempt
        already running finishes; nothing follows it."""
        now = datetime.now(timezone_.utc)
        schedule.enabled = False
        if schedule.paused_at is None:
            schedule.paused_at = now
        self.session.add(schedule)
        self.session.flush()
        cancelled = ScheduleOccurrenceService(self.session).cancel_pending(schedule.id, error="Paused before it ran.")
        record_schedule_event(self.session, schedule.id, "paused", {"cancelled_pending": cancelled})
        self.session.commit()
        self.session.refresh(schedule)
        return schedule

    def resume_schedule(self, schedule_id: UUID) -> Schedule:
        return self._resume(self.get_schedule(schedule_id))

    def _resume(self, schedule: Schedule) -> Schedule:
        """Turn the schedule back on, clearing a pause, an auto-pause and its
        failure streak. The run history stays. Nothing runs at once: a
        recurring schedule waits for its next slot. A one-off whose time has
        passed cannot wait for anything; run it now or give it a new time."""
        now = datetime.now(timezone_.utc)
        was_auto_paused = schedule.auto_paused_at is not None
        if schedule.cadence == Cadence.once and _utc(schedule.next_run_at) <= now:
            raise ScheduleConflict(
                "This one-time task's time has passed. Run it now, or edit it to give it a new time."
            )
        schedule.enabled = True
        schedule.paused_at = None
        schedule.auto_paused_at = None
        schedule.missed_at = None
        schedule.consecutive_failures = 0
        schedule.next_run_at = normalize_next_run(schedule.cadence, schedule.next_run_at, schedule.timezone, now=now)
        self.session.add(schedule)
        self.session.flush()
        # A retry that was being scheduled as the pause landed never starts
        # while paused (start_attempt checks); drop it, so the next slot can.
        ScheduleOccurrenceService(self.session).cancel_pending(schedule.id, error="Paused before it ran.")
        record_schedule_event(self.session, schedule.id, "resumed", {"from_auto_pause": was_auto_paused})
        self.session.commit()
        self.session.refresh(schedule)
        return schedule

    def last_skipped(self, schedule_id: UUID) -> ScheduleSkippedPeriod | None:
        return self.session.exec(
            self.session.select(ScheduleSkippedPeriod)
            .where(ScheduleSkippedPeriod.schedule_id == schedule_id)
            .order_by(ScheduleSkippedPeriod.created_at.desc())  # type: ignore[union-attr]
            .limit(1)
        ).first()


class ScheduleConflict(Exception):
    """An action the schedule's state does not allow; the message says why."""


class ScheduleRunService:
    def __init__(self, session: ScopedSession) -> None:
        self.session = session

    def create_run(self, schedule_id: UUID, is_manual: bool = False) -> ScheduleRun:
        run = ScheduleRun(
            schedule_id=schedule_id,
            started_at=datetime.now(timezone.utc),
            status=RunStatus.running,
            is_manual=is_manual,
        )
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)
        return run

    def has_running_run(self, schedule_id: UUID) -> bool:
        """
        Check if the schedule has a running non-manual (cron) run.
        The scheduler's due-check gates on has_active_run instead, so a
        manual run in flight also defers the slot (PR #181 review).
        """
        run = self.session.exec(
            self.session.select(ScheduleRun)
            .where(ScheduleRun.schedule_id == schedule_id)
            .where(ScheduleRun.status == RunStatus.running)
            .where(ScheduleRun.is_manual == False)
            .limit(1)
        ).first()
        return run is not None

    def has_active_run(self, schedule_id: UUID) -> bool:
        """Any in-flight run, manual or cron. Drives the UI "running" state
        (unlike has_running_run, which only guards cron overlap)."""
        # Anchor: ScheduleRun has no org column, so gate on the parent
        # schedule being visible in scope (request-reachable via _serialize).
        _stmt = self.session.select(Schedule).where(Schedule.id == schedule_id)
        if self.session.scope.org_mode:
            _stmt = _stmt.where(Schedule.created_by == self.session.scope.user_id)
        if self.session.exec(_stmt).first() is None:
            return False
        run = self.session.exec(
            self.session.select(ScheduleRun)
            .where(ScheduleRun.schedule_id == schedule_id)
            .where(ScheduleRun.status == RunStatus.running)
            .limit(1)
        ).first()
        return run is not None

    def last_successful_finish(self, schedule_id: UUID, *, manual_only: bool = False) -> datetime | None:
        """When the schedule's most recent successful run finished, or None if
        it never succeeded. Used by the scheduler's freshness guard to skip a
        due slot right after a manual run.
        """
        query = (
            self.session.select(ScheduleRun)
            .where(ScheduleRun.schedule_id == schedule_id)
            .where(ScheduleRun.status == RunStatus.success)
        )
        if manual_only:
            query = query.where(ScheduleRun.is_manual == True)  # noqa: E712
        run = self.session.exec(
            query
            .order_by(ScheduleRun.finished_at.desc())  # type: ignore[union-attr]
            .limit(1)
        ).first()
        if run is None or run.finished_at is None:
            return None
        finished = run.finished_at
        return finished if finished.tzinfo else finished.replace(tzinfo=timezone.utc)

    def still_exists(self, conversation_id: UUID | None) -> UUID | None:
        """The conversation id back, or None once the conversation is gone.

        A run holds its conversation id in a local for the whole run and writes
        it back at the end, so a user who deletes that chat mid-run leaves every
        later write pointing at a row that no longer exists. On Postgres each
        one raises, and the scheduler swallows the exception, which strands the
        run at `running`: `has_active_run` then reports the schedule busy and
        the due-check skips it on every poll until a restart reaps it. Writing
        NULL instead loses only the link to a conversation the user deleted on
        purpose.

        Read on the raw session so the org filter cannot answer "gone" for a
        conversation that is merely out of scope, which would drop a live link.

        A core SELECT, not `Session.get`: the session factory runs with
        `expire_on_commit=False`, so `get` answers from the identity map with
        no query at all, and the scheduler's session is the one that created
        this conversation. `get` would report it alive for the whole run,
        however long ago another session deleted it. Only a statement asks
        the database.
        """
        if conversation_id is None:
            return None
        raw = unsafe_unscoped_session(self.session)
        row = raw.execute(
            sa.select(Conversation.id).where(Conversation.id == conversation_id)
        ).first()
        return conversation_id if row is not None else None

    def set_run_conversation(self, run_id: UUID, conversation_id: UUID) -> None:
        """Attach the run's conversation as soon as it is known — before the
        turn executes — so the UI can open a run that is still in flight."""
        run = self.session.get(ScheduleRun, run_id)
        if run is None:
            return
        run.conversation_id = self.still_exists(conversation_id)
        self.session.add(run)
        try:
            self.session.commit()
        except IntegrityError:
            # The delete landed between still_exists' read and this commit,
            # and its release already nulled the column in the database. The
            # run simply has no conversation to point at.
            self.session.rollback()

    def finish_run(
        self,
        run_id: UUID,
        conversation_id: UUID | None = None,
        error: str | None = None,
        status: RunStatus | None = None,
    ) -> ScheduleRun:
        run = self.session.get(ScheduleRun, run_id)
        if run is None:
            raise ValueError("ScheduleRun not found")
        now = datetime.now(timezone.utc)
        run.finished_at = now
        started_at = run.started_at if run.started_at.tzinfo else run.started_at.replace(tzinfo=timezone.utc)
        run.duration_ms = int((now - started_at).total_seconds() * 1000)
        run.status = status or (RunStatus.failed if error else RunStatus.success)
        run.error = error
        if conversation_id is not None:
            run.conversation_id = self.still_exists(conversation_id)
        self.session.add(run)
        try:
            self.session.commit()
        except IntegrityError:
            # The delete landed between still_exists' read and this commit,
            # and its release already nulled the column in the database.
            # Rewrite without the pointer rather than stranding the run at
            # `running`, which would wedge the schedule until a restart.
            self.session.rollback()
            return self.finish_run(run_id, error=error, status=status)
        self.session.refresh(run)
        return run

    def end_run(
        self,
        run_id: UUID,
        *,
        status: RunStatus,
        error: str | None,
        outcome: AttemptOutcome,
        conversation_id: UUID | None = None,
    ) -> bool:
        """Stage the end of a run that is still ``running``. False, writing
        nothing, when it has already ended: a reaper ended it while its
        holder was out of touch, and that record stands. The caller commits."""
        run = self.session.get(ScheduleRun, run_id)
        if run is None:
            return False
        now = datetime.now(timezone.utc)
        values: dict = dict(
            finished_at=now,
            duration_ms=int((now - _utc(run.started_at)).total_seconds() * 1000),
            status=status,
            error=error,
            outcome=outcome,
        )
        if conversation_id is not None:
            values["conversation_id"] = self.still_exists(conversation_id)
        result = unsafe_unscoped_session(self.session).execute(
            sa.update(ScheduleRun)
            .where(ScheduleRun.id == run_id)
            .where(ScheduleRun.status == RunStatus.running)
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        return result.rowcount == 1

    def reap_orphaned_runs(
        self,
        error: str = "Run orphaned by a server restart before it completed.",
        started_before: datetime | None = None,
    ) -> int:
        """Mark every run still in ``running`` as ``failed``.

        A crash/restart while a run is in flight leaves its ``ScheduleRun`` in
        ``running`` forever. Because the scheduler's due-check skips schedules
        with a running run, a single stale row wedges that schedule
        permanently. Called once on boot to release those runs. Returns the
        number of runs reaped.

        ``started_before`` limits it to runs that began before then, and
        leaves runs that belong to an occurrence to the scheduler's lease
        reaper. Where other processes share the database, a run newer than
        that may still be live on one of them, and an occurrence's run is live
        for as long as its holder renews the lease, however old it is.
        """
        query = self.session.select(ScheduleRun).where(ScheduleRun.status == RunStatus.running)
        if started_before is not None:
            query = query.where(ScheduleRun.started_at < started_before).where(
                ScheduleRun.occurrence_id.is_(None)  # type: ignore[union-attr]
            )
        runs = self.session.exec(query).all()
        now = datetime.now(timezone.utc)
        for run in runs:
            started_at = (
                run.started_at
                if run.started_at.tzinfo
                else run.started_at.replace(tzinfo=timezone.utc)
            )
            run.finished_at = now
            run.duration_ms = int((now - started_at).total_seconds() * 1000)
            run.status = RunStatus.failed
            run.error = error
            run.outcome = (
                AttemptOutcome.interrupted
                if run.execution_started_at is not None
                else AttemptOutcome.failed_before_execution
            )
            self.session.add(run)
            # Settle the occurrence too, without a retry: a run cut off by a
            # quit may be resumed days later, and replaying a stale slot then
            # would surprise more than it helps.
            if run.occurrence_id is not None:
                ScheduleOccurrenceService(self.session).settle(
                    run.occurrence_id, OccurrenceStatus.failed, error=error
                )
        self.session.commit()
        return len(runs)

    def run_context(self, runs: list[ScheduleRun]) -> dict[UUID, dict]:
        """For each run at an occurrence: what fired it, the slot it ran for,
        and the version it ran. Read through the occurrence, which every
        attempt at it shares, rather than copied onto each run."""
        occurrence_ids = {r.occurrence_id for r in runs if r.occurrence_id is not None}
        if not occurrence_ids:
            return {}
        rows = unsafe_unscoped_session(self.session).execute(
            sa.select(
                ScheduleOccurrence.id, ScheduleOccurrence.trigger_source, ScheduleOccurrence.scheduled_for,
                ScheduleVersion.number,
            )
            .select_from(ScheduleOccurrence)
            .outerjoin(ScheduleVersion, ScheduleVersion.id == ScheduleOccurrence.version_id)
            .where(ScheduleOccurrence.id.in_(occurrence_ids))  # type: ignore[union-attr]
        ).all()
        return {
            row[0]: {"trigger": row[1], "scheduled_for": _utc(row[2]), "version": row[3]}
            for row in rows
        }

    def last_finished(self, schedule_id: UUID) -> ScheduleRun | None:
        """The schedule's most recent run that has ended, manual or cron."""
        return self.session.exec(
            self.session.select(ScheduleRun)
            .where(ScheduleRun.schedule_id == schedule_id)
            .where(ScheduleRun.status != RunStatus.running)
            .order_by(ScheduleRun.started_at.desc())  # type: ignore[union-attr]
            .limit(1)
        ).first()

    def list_runs(self, schedule_id: UUID, limit: int = 100) -> list[ScheduleRun]:
        # Anchor on parent visibility (child table, request-reachable).
        _stmt = self.session.select(Schedule).where(Schedule.id == schedule_id)
        if self.session.scope.org_mode:
            _stmt = _stmt.where(Schedule.created_by == self.session.scope.user_id)
        if self.session.exec(_stmt).first() is None:
            return []
        return list(
            self.session.exec(
                self.session.select(ScheduleRun)
                .where(ScheduleRun.schedule_id == schedule_id)
                .order_by(ScheduleRun.started_at.desc())  # type: ignore[union-attr]
                .limit(limit)
            ).all()
        )


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class StartedAttempt:
    """An attempt this caller now holds: the token makes every later write
    by it conditional on still holding the occurrence."""

    run_id: UUID
    schedule_id: UUID
    number: int
    token: str


@dataclass(frozen=True)
class ExpiredLease:
    occurrence_id: UUID
    schedule_id: UUID
    token: str | None


class Settlement(str, Enum):
    """Where an attempt's end left its occurrence."""

    retry_scheduled = "retry_scheduled"
    settled = "settled"
    # The caller no longer held the occurrence: it was reaped or settled
    # elsewhere, and nothing was written.
    lost = "lost"


def _new_token() -> str:
    return str(uuid4())


class ScheduleOccurrenceService:
    """Claims, attempts and settles schedule occurrences.

    Runs on the scheduler's unscoped session: an occurrence has no org column
    and is only ever reached through its schedule, which the scheduler has
    already resolved.

    Every change to an active occurrence is a conditional UPDATE: on its
    attempt count when an attempt starts, and on its lease token afterwards.
    Two pollers, a reaper and a late-finishing holder can each read the same
    row, but only the first write whose condition still holds lands; the rest
    change nothing and learn they lost. Postgres re-checks the condition after
    waiting for a concurrent writer's row lock; SQLite serializes writers.
    """

    def __init__(self, session: ScopedSession) -> None:
        self.session = session

    def _update(self, stmt) -> bool:
        """Run a conditional UPDATE that targets one row; True when it landed."""
        result = unsafe_unscoped_session(self.session).execute(
            stmt.execution_options(synchronize_session=False)
        )
        return result.rowcount == 1

    def _active(self, occurrence_id: UUID, token: str | None = None):
        conditions = [
            ScheduleOccurrence.id == occurrence_id,
            ScheduleOccurrence.status == OccurrenceStatus.active,
        ]
        if token is not None:
            conditions.append(ScheduleOccurrence.lease_token == token)
        return sa.update(ScheduleOccurrence).where(*conditions)

    def claim(
        self,
        schedule_id: UUID,
        *,
        occurrence_key: str,
        trigger_source: str,
        scheduled_for: datetime,
        max_attempts: int,
        lease_seconds: float,
        version_id: UUID | None = None,
    ) -> ScheduleOccurrence | None:
        """Insert the occurrence and commit it with whatever the caller staged.

        Returns None when the key was already claimed or the schedule already
        has an active occurrence; the caller's staged changes are rolled back
        with the losing insert, so a slot advanced alongside a failed claim
        stays where the winner left it.

        The claim holds a lease from the start, so an occurrence whose process
        dies before its first attempt is reaped like one that died mid-run.
        """
        now = datetime.now(timezone.utc)
        occurrence = ScheduleOccurrence(
            schedule_id=schedule_id,
            occurrence_key=occurrence_key,
            trigger_source=trigger_source,
            scheduled_for=scheduled_for,
            status=OccurrenceStatus.active,
            max_attempts=max_attempts,
            next_attempt_at=now,
            lease_expires_at=now + timedelta(seconds=lease_seconds),
            lease_token=_new_token(),
            version_id=version_id,
        )
        self.session.add(occurrence)
        try:
            self.session.commit()
        except IntegrityError:
            self.session.rollback()
            return None
        self.session.refresh(occurrence)
        return occurrence

    def get(self, occurrence_id: UUID) -> ScheduleOccurrence | None:
        return self.session.get(ScheduleOccurrence, occurrence_id)

    def _read(self, occurrence_id: UUID):
        """The occurrence's row as the database holds it now, not as an
        earlier read left it in this session."""
        return unsafe_unscoped_session(self.session).execute(
            sa.select(ScheduleOccurrence.__table__).where(ScheduleOccurrence.id == occurrence_id)
        ).mappings().first()

    def pending_retry_at(self, schedule_id: UUID) -> datetime | None:
        """When the schedule's active occurrence will try again, or None when
        nothing is waiting to retry (idle, or an attempt is running)."""
        occurrence = self.session.exec(
            self.session.select(ScheduleOccurrence)
            .where(ScheduleOccurrence.schedule_id == schedule_id)
            .where(ScheduleOccurrence.status == OccurrenceStatus.active)
            .where(ScheduleOccurrence.attempts > 0)
            .where(ScheduleOccurrence.next_attempt_at.is_not(None))  # type: ignore[union-attr]
            .limit(1)
        ).first()
        return _utc(occurrence.next_attempt_at) if occurrence else None

    def has_active(self, schedule_id: UUID) -> bool:
        return self.session.exec(
            self.session.select(ScheduleOccurrence)
            .where(ScheduleOccurrence.schedule_id == schedule_id)
            .where(ScheduleOccurrence.status == OccurrenceStatus.active)
            .limit(1)
        ).first() is not None

    def start_attempt(
        self, occurrence_id: UUID, *, is_manual: bool, lease_seconds: float
    ) -> StartedAttempt | None:
        """Take the occurrence and record its next attempt, or None when there
        is none to start: the occurrence settled, an attempt is already
        running, its retry is not due yet, or another poller took it first."""
        row = self._read(occurrence_id)
        now = datetime.now(timezone.utc)
        if (
            row is None
            or row["status"] != OccurrenceStatus.active
            or row["next_attempt_at"] is None
            or _utc(row["next_attempt_at"]) > now
        ):
            return None
        number = row["attempts"] + 1
        token = _new_token()
        stmt = self._active(occurrence_id)
        if row["trigger_source"] != "manual":
            # Only an explicit pause stops it: a one-off is disabled by its
            # own claim and its retries still run.
            stmt = stmt.where(_not_paused(ScheduleOccurrence.schedule_id))
        took = self._update(
            stmt
            .where(ScheduleOccurrence.attempts == row["attempts"])
            .where(ScheduleOccurrence.next_attempt_at.is_not(None))  # type: ignore[union-attr]
            .where(ScheduleOccurrence.next_attempt_at <= now)  # type: ignore[operator]
            .values(
                attempts=number,
                next_attempt_at=None,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
                lease_token=token,
            )
        )
        if not took:
            self.session.rollback()
            return None
        run = ScheduleRun(
            schedule_id=row["schedule_id"],
            started_at=now,
            status=RunStatus.running,
            is_manual=is_manual,
            occurrence_id=occurrence_id,
            attempt=number,
        )
        self.session.add(run)
        try:
            self.session.commit()
        except IntegrityError:
            # (occurrence_id, attempt) is taken: another poller won the race.
            self.session.rollback()
            return None
        return StartedAttempt(run_id=run.id, schedule_id=row["schedule_id"], number=number, token=token)

    def renew(self, occurrence_id: UUID, token: str, *, lease_seconds: float) -> bool:
        """Extend the holder's lease. False when it no longer holds it."""
        renewed = self._update(
            self._active(occurrence_id, token).values(
                lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=lease_seconds),
            )
        )
        self.session.commit()
        return renewed

    def begin_execution(self, attempt: StartedAttempt, occurrence_id: UUID, *, lease_seconds: float) -> bool:
        """Record that the attempt is about to hand its turn to the agent, and
        renew its lease, in one commit. False, writing nothing, when the
        attempt no longer holds its occurrence: it must not run then, because
        whoever took the occurrence may be starting another attempt."""
        now = datetime.now(timezone.utc)
        held = self._update(
            self._active(occurrence_id, attempt.token).values(
                lease_expires_at=now + timedelta(seconds=lease_seconds),
            )
        )
        if not held:
            self.session.rollback()
            return False
        self._update(
            sa.update(ScheduleRun)
            .where(ScheduleRun.id == attempt.run_id)
            .where(ScheduleRun.status == RunStatus.running)
            .values(execution_started_at=now)
        )
        self.session.commit()
        return True

    def request_stop(self, run_id: UUID) -> None:
        self._update(
            sa.update(ScheduleRun)
            .where(ScheduleRun.id == run_id)
            .where(ScheduleRun.stop_requested_at.is_(None))  # type: ignore[union-attr]
            .values(stop_requested_at=datetime.now(timezone.utc))
        )
        self.session.commit()

    def end_attempt(
        self,
        occurrence_id: UUID,
        token: str,
        *,
        outcome: AttemptOutcome,
        error: str | None,
        retry_delay: float | None,
    ) -> Settlement:
        """Stage where a finished attempt leaves its occurrence. The caller
        commits.

        Only an attempt that failed before execution, with attempts left and
        a ``retry_delay``, is tried again; every other ending settles the
        occurrence. Nothing is written when the caller no longer holds it.
        """
        row = self._read(occurrence_id)
        if row is None:
            return Settlement.lost
        retry = (
            outcome == AttemptOutcome.failed_before_execution
            and retry_delay is not None
            and row["attempts"] < row["max_attempts"]
            and not self._paused(row["schedule_id"])
        )
        stmt = self._active(occurrence_id, token)
        if retry:
            landed = self._update(stmt.values(
                error=error,
                lease_token=None,
                lease_expires_at=None,
                next_attempt_at=datetime.now(timezone.utc) + timedelta(seconds=retry_delay),
            ))
            return Settlement.retry_scheduled if landed else Settlement.lost
        status = _occurrence_status(outcome)
        if not self._update(stmt.values(**_settled(status, error))):
            return Settlement.lost
        self._count(row, status)
        return Settlement.settled

    def settle(self, occurrence_id: UUID, status: str, *, error: str | None = None) -> bool:
        """Settle an active occurrence whoever holds it. Only for a process
        that knows no one else can: the desktop's boot, before its scheduler
        starts. The caller commits."""
        row = self._read(occurrence_id)
        if row is None or not self._update(self._active(occurrence_id).values(**_settled(status, error))):
            return False
        self._count(row, status)
        return True

    def cancel_pending(self, schedule_id: UUID, *, error: str) -> int:
        """Stage cancelling what would start on its own: a cron retry waiting
        for its time, or a cron slot claimed whose first attempt has not
        started. A running attempt, and a run-now, are left alone. The caller
        commits."""
        result = unsafe_unscoped_session(self.session).execute(
            sa.update(ScheduleOccurrence)
            .where(ScheduleOccurrence.schedule_id == schedule_id)
            .where(ScheduleOccurrence.status == OccurrenceStatus.active)
            .where(ScheduleOccurrence.trigger_source != "manual")
            .where(sa.or_(
                ScheduleOccurrence.attempts == 0,
                ScheduleOccurrence.lease_token.is_(None),  # type: ignore[union-attr]
            ))
            .values(**_settled(OccurrenceStatus.cancelled, error))
            .execution_options(synchronize_session=False)
        )
        return result.rowcount

    def _paused(self, schedule_id: UUID) -> bool:
        return unsafe_unscoped_session(self.session).execute(
            sa.select(sa.literal(True)).where(~_not_paused(sa.literal(schedule_id, sa.Uuid())))
        ).first() is not None

    def _count(self, row, status: str) -> None:
        """Stage the schedule's failure streak for an occurrence that just
        settled, and pause the schedule when the streak reaches its limit.

        Runs once per occurrence: only the write that settled it calls this.
        A cron occurrence that settles failed extends the streak, however many
        attempts it took; a success of either kind ends it. A cancelled one
        leaves it as is, and so does a manual failure, which the user saw.
        """
        schedule_id = row["schedule_id"]
        raw = unsafe_unscoped_session(self.session)
        if status == OccurrenceStatus.success:
            raw.execute(sa.update(Schedule).where(Schedule.id == schedule_id).values(consecutive_failures=0))
            return
        if status != OccurrenceStatus.failed or row["trigger_source"] == "manual":
            return
        raw.execute(
            sa.update(Schedule).where(Schedule.id == schedule_id)
            .values(consecutive_failures=Schedule.consecutive_failures + 1)
        )
        paused = raw.execute(
            sa.update(Schedule)
            .where(Schedule.id == schedule_id)
            .where(Schedule.consecutive_failures >= MAX_FAILED_OCCURRENCES)
            .where(Schedule.auto_paused_at.is_(None))  # type: ignore[union-attr]
            .values(enabled=False, auto_paused_at=datetime.now(timezone.utc))
        ).rowcount
        if paused:
            self.session.add(ScheduleEvent(
                schedule_id=schedule_id, kind="auto_paused", actor=None,
                detail={"failed_runs": MAX_FAILED_OCCURRENCES, "occurrence_id": str(row["id"])},
            ))
            logger.warning(f"Schedule {schedule_id} paused after {MAX_FAILED_OCCURRENCES} failed runs in a row")

    def ready_for_attempt(self, now: datetime) -> list[UUID]:
        """Cron occurrences whose next attempt is due, oldest first. A manual
        occurrence is started by the request that claimed it, never here."""
        return list(
            self.session.exec(
                self.session.select(ScheduleOccurrence.id)
                .where(ScheduleOccurrence.status == OccurrenceStatus.active)
                .where(ScheduleOccurrence.trigger_source != "manual")
                .where(ScheduleOccurrence.next_attempt_at.is_not(None))  # type: ignore[union-attr]
                .where(ScheduleOccurrence.next_attempt_at <= now)  # type: ignore[operator]
                .order_by(ScheduleOccurrence.next_attempt_at)  # type: ignore[arg-type]
            ).all()
        )

    def expired_leases(self, now: datetime) -> list[ExpiredLease]:
        """Active occurrences that outlived their lease: the process holding
        them died before or during an attempt, or stopped renewing it. An
        occurrence waiting for a retry holds no lease."""
        rows = unsafe_unscoped_session(self.session).execute(
            sa.select(
                ScheduleOccurrence.id, ScheduleOccurrence.schedule_id, ScheduleOccurrence.lease_token,
            )
            .where(ScheduleOccurrence.status == OccurrenceStatus.active)
            .where(ScheduleOccurrence.lease_expires_at.is_not(None))  # type: ignore[union-attr]
            .where(ScheduleOccurrence.lease_expires_at < now)  # type: ignore[operator]
        ).all()
        return [ExpiredLease(occurrence_id=r[0], schedule_id=r[1], token=r[2]) for r in rows]

    def reap(
        self, lease: ExpiredLease, *, now: datetime, error: str, retry_delay: Callable[[int], float],
    ) -> Settlement:
        """Take back an occurrence whose holder stopped renewing it, in one
        commit with its running attempt.

        Lands only while the lease is still the expired one this caller saw:
        a holder that renewed, started an attempt or settled in between keeps
        it. An attempt that never reached the agent is tried again; one that
        did may have acted, so its occurrence settles without a replay.
        """
        run = self.running_attempt(lease.occurrence_id)
        executed = run is not None and run.execution_started_at is not None
        row = self._read(lease.occurrence_id)
        if row is None:
            return Settlement.lost
        outcome = AttemptOutcome.interrupted if executed else AttemptOutcome.failed_before_execution
        stmt = (
            self._active(lease.occurrence_id)
            .where(ScheduleOccurrence.lease_expires_at < now)  # type: ignore[operator]
            .where(
                ScheduleOccurrence.lease_token == lease.token
                if lease.token is not None
                else ScheduleOccurrence.lease_token.is_(None)  # type: ignore[union-attr]
            )
        )
        # A manual occurrence is only ever started by the request that
        # claimed it, so a retry of one would wait forever.
        retry = (
            not executed
            and row["trigger_source"] != "manual"
            and row["attempts"] < row["max_attempts"]
        )
        if retry:
            values = dict(
                error=error, lease_token=None, lease_expires_at=None,
                next_attempt_at=now + timedelta(seconds=retry_delay(row["attempts"])),
            )
        else:
            values = _settled(OccurrenceStatus.failed, error)
        if not self._update(stmt.values(**values)):
            self.session.rollback()
            return Settlement.lost
        if not retry:
            self._count(row, OccurrenceStatus.failed)
        if run is not None:
            self._update(
                sa.update(ScheduleRun)
                .where(ScheduleRun.id == run.id)
                .where(ScheduleRun.status == RunStatus.running)
                .values(status=RunStatus.failed, error=error, finished_at=now, outcome=outcome)
            )
        self.session.commit()
        return Settlement.retry_scheduled if retry else Settlement.settled

    def running_attempt(self, occurrence_id: UUID) -> ScheduleRun | None:
        return self.session.exec(
            self.session.select(ScheduleRun)
            .where(ScheduleRun.occurrence_id == occurrence_id)
            .where(ScheduleRun.status == RunStatus.running)
            .limit(1)
        ).first()


# Cron occurrences in a row that may settle failed before the schedule pauses
# itself. Retries within one occurrence count once.
MAX_FAILED_OCCURRENCES = 3


def _not_paused(schedule_id):
    """True while the schedule has no explicit pause or auto-pause."""
    return sa.exists().where(
        Schedule.id == schedule_id,
        Schedule.paused_at.is_(None),  # type: ignore[union-attr]
        Schedule.auto_paused_at.is_(None),  # type: ignore[union-attr]
    )


def _occurrence_status(outcome: AttemptOutcome) -> OccurrenceStatus:
    if outcome == AttemptOutcome.completed:
        return OccurrenceStatus.success
    if outcome == AttemptOutcome.cancelled:
        return OccurrenceStatus.cancelled
    return OccurrenceStatus.failed


def _settled(status: str, error: str | None) -> dict:
    return dict(
        status=status,
        error=error,
        next_attempt_at=None,
        lease_expires_at=None,
        lease_token=None,
        settled_at=datetime.now(timezone.utc),
    )
