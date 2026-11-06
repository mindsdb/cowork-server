from __future__ import annotations

from datetime import datetime, timedelta, timezone
from datetime import timezone as timezone_
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.attributes import set_attribute

from cowork.common.logger import get_logger, log_context
from cowork.db.scoped import ScopedSession, unsafe_unscoped_session
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.models.schedule import Schedule, ScheduleRun
from cowork.schedule_timing import advance_occurrence, normalize_next_run
from cowork.schemas.schedules import AttemptOutcome, Cadence, RunStatus

logger = get_logger(__name__)

# Cron slots in a row that may settle failed before the schedule pauses
# itself.
MAX_FAILED_SLOTS = 3


class ScheduleConflict(Exception):
    """An action the schedule's state does not allow; the message says why."""


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

    def get_schedule(self, schedule_id: UUID) -> Schedule:
        schedule = self._owned(schedule_id)
        if schedule is None:
            raise ValueError("Schedule not found")
        return schedule

    def lock_schedule(self, schedule_id: UUID) -> "Schedule | None":
        """The schedule as stored now, locked until the unit commits, so a
        check made on it still holds when the unit writes.

        SQLite ignores FOR UPDATE, so a write that changes nothing comes
        first: it takes SQLite's database write lock (and the row lock on
        Postgres), and the read after it sees whatever committed before.
        The write is `SET title = title`, done in the database, so a rename
        saved since `_owned` read the row is kept rather than written over.
        """
        schedule = self._owned(schedule_id)
        if schedule is None:
            return None
        # Through the ORM's instrumentation: SQLModel's own setattr would
        # validate the SQL expression as a string and refuse it.
        set_attribute(schedule, "title", Schedule.title)
        self.session.add(schedule)
        self.session.flush()
        return self.session.exec(
            self._owned_select()
            .where(Schedule.id == schedule_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).first()

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
            next_run_at=next_run_at,
            model=model,
            timezone=timezone,
            project_id=target_project_id,
            enabled=enabled,
            # `timezone` is this method's zone-name argument here.
            paused_at=None if enabled else datetime.now(timezone_.utc),
        )
        self.session.add(schedule)
        self.session.commit()
        self.session.refresh(schedule)
        return schedule

    def update_schedule(self, schedule_id: UUID, **kwargs) -> Schedule:
        """Apply an edit. ``enabled`` pauses or resumes, as those actions do;
        the editor sends the current value with every save, which changes
        nothing."""
        schedule = self.get_schedule(schedule_id)
        if "project_id" in kwargs and kwargs["project_id"] is None:
            # No project: the caller's default, as at create.
            kwargs["project_id"] = self._default_project_id()
        new_project_id = kwargs.get("project_id")
        if new_project_id is not None and self.session.get(Project, new_project_id) is None:
            raise ValueError("Project not found")
        enabled = kwargs.pop("enabled", None)
        # Compared with the state before the edit: a new time below can turn a
        # missed one-off back on, and the `enabled` the editor sent with the
        # old state must not pause it again.
        was_on = _runs_on_its_own(schedule)
        for field, value in kwargs.items():
            if value is not None and hasattr(schedule, field):
                setattr(schedule, field, value)
        if (
            schedule.missed_at is not None
            and _utc(schedule.next_run_at) > datetime.now(timezone.utc)
        ):
            # Only one-offs are marked missed. A future time schedules one
            # again, including when this edit changes it to a recurring task.
            schedule.missed_at = None
            schedule.enabled = schedule.paused_at is None and schedule.auto_paused_at is None
        self.session.add(schedule)
        if enabled is not None and enabled != was_on:
            return self._pause(schedule) if not enabled else self._resume(schedule)
        self.session.commit()
        self.session.refresh(schedule)
        return schedule

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
        """Stop future slots: nothing is due on a disabled schedule. A run
        already going finishes; nothing follows it."""
        schedule.enabled = False
        if schedule.paused_at is None:
            schedule.paused_at = datetime.now(timezone.utc)
        self.session.add(schedule)
        self.session.commit()
        self.session.refresh(schedule)
        return schedule

    def resume_schedule(self, schedule_id: UUID) -> Schedule:
        schedule = self.get_schedule(schedule_id)
        # A stale tab may repeat Resume after another request resumed it.
        # Leave its due slot and failure history untouched.
        return schedule if _runs_on_its_own(schedule) else self._resume(schedule)

    def _resume(self, schedule: Schedule) -> Schedule:
        """Turn the schedule back on, clearing a pause, an auto-pause and its
        failure streak. The run history stays.

        A recurring schedule waits for its next slot, with one exception: a
        slot left due before the pause, by a run whose agent did nothing,
        runs once more (see _slot_left_due). A one-off whose time has passed
        cannot wait for anything; run it now or give it a new time."""
        now = datetime.now(timezone.utc)
        if schedule.cadence == Cadence.once and _utc(schedule.next_run_at) <= now:
            raise ScheduleConflict(
                "This one-time task's time has passed. Run it now, or edit it to give it a new time."
            )
        schedule.enabled = True
        schedule.paused_at = None
        schedule.auto_paused_at = None
        schedule.missed_at = None
        schedule.consecutive_failures = 0
        schedule.next_run_at, _ = self._resumed_next_run(schedule, now)
        self.session.add(schedule)
        self.session.commit()
        self.session.refresh(schedule)
        return schedule

    def resume_plan(self, schedule_id: UUID) -> tuple[datetime, bool]:
        """What resuming now would do, without doing it: the time the task
        will next run, and whether that is a slot left due running right away."""
        return self._resumed_next_run(self.get_schedule(schedule_id), datetime.now(timezone.utc))

    def _resumed_next_run(self, schedule: Schedule, now: datetime) -> tuple[datetime, bool]:
        if self._slot_left_due(schedule, now):
            return _utc(schedule.next_run_at), True
        return normalize_next_run(schedule.cadence, schedule.next_run_at, schedule.timezone, now=now), False

    def _slot_left_due(self, schedule: Schedule, now: datetime) -> bool:
        """Whether a resumed recurring schedule keeps its due slot: the last
        cron run at it gave it back before its agent acted (a pause or a
        passing outage during that run), and no later slot has come due.
        Slots that came due while paused never run, and a stale slot goes
        with them. A slot whose run may have acted is used up, as it would
        have been without the pause."""
        if schedule.cadence == Cadence.once:
            return False
        slot = _utc(schedule.next_run_at)
        if slot > now or advance_occurrence(schedule.cadence, slot, schedule.timezone) <= now:
            return False
        # While the slot is still due, a cron run that started at or after it
        # was a run at it.
        last_outcome = self.session.exec(
            self.session.select(ScheduleRun.outcome)
            .where(ScheduleRun.schedule_id == schedule.id)
            .where(ScheduleRun.is_manual == sa.false())
            .where(ScheduleRun.started_at >= slot)
            .where(ScheduleRun.status != RunStatus.running)
            .order_by(ScheduleRun.started_at.desc())  # type: ignore[union-attr]
            .limit(1)
        ).first()
        return last_outcome == AttemptOutcome.failed_before_execution


def _runs_on_its_own(schedule: Schedule) -> bool:
    return schedule.paused_at is None and schedule.auto_paused_at is None and schedule.enabled


def count_settled_slot(session: ScopedSession, schedule_id: UUID, *, failed: bool) -> None:
    """Stage the failure streak for a cron slot that just settled, and pause
    the schedule when the streak reaches MAX_FAILED_SLOTS.

    Called by the write that settles the slot: a failure extends the streak,
    and a success ends it. Atomic updates, so two replicas settling different slots never
    lose a count. The caller commits.
    """
    raw = unsafe_unscoped_session(session)
    if not failed:
        raw.execute(
            sa.update(Schedule).where(Schedule.id == schedule_id).values(consecutive_failures=0)
            .execution_options(synchronize_session=False)
        )
        return
    raw.execute(
        sa.update(Schedule).where(Schedule.id == schedule_id)
        .values(consecutive_failures=Schedule.consecutive_failures + 1)
        .execution_options(synchronize_session=False)
    )
    paused = raw.execute(
        sa.update(Schedule)
        .where(Schedule.id == schedule_id)
        .where(Schedule.consecutive_failures >= MAX_FAILED_SLOTS)
        .where(Schedule.auto_paused_at.is_(None))  # type: ignore[union-attr]
        .values(enabled=False, auto_paused_at=datetime.now(timezone.utc))
        .execution_options(synchronize_session=False)
    ).rowcount
    if paused:
        logger.warning(
            "Schedule paused after %s failed runs in a row", MAX_FAILED_SLOTS,
            extra=log_context(schedule_id=schedule_id),
        )


class ScheduleBusy(Exception):
    """The schedule already has a run going, manual or cron."""


class SlotAlreadyClaimed(Exception):
    """Another run already holds this schedule's cron slot."""


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _record_lost_run(run: ScheduleRun) -> None:
    """Record what is known of a run whose process died. One that never
    claimed execution never started its turn; one that did may have acted.
    The claim survives deletion of the conversation."""
    run.outcome = (
        AttemptOutcome.failed_before_execution if run.execution_claimed_at is None else AttemptOutcome.interrupted
    )


class ScheduleRunService:
    def __init__(self, session: ScopedSession) -> None:
        self.session = session

    def create_run(
        self,
        schedule_id: UUID,
        is_manual: bool = False,
        *,
        slot_at: datetime | None = None,
        lease_seconds: float | None = None,
    ) -> ScheduleRun:
        """Insert the run. With ``slot_at``, the insert claims that cron slot:
        the unique index on (schedule_id, slot_at) refuses a second run for it,
        and that refusal raises SlotAlreadyClaimed. A schedule holds one
        `running` run at a time (uq_schedule_runs_one_running), so an insert
        while another run is going raises ScheduleBusy. With ``lease_seconds``, the
        run counts as alive until then unless renew_lease extends it."""
        now = datetime.now(timezone.utc)
        run = ScheduleRun(
            schedule_id=schedule_id,
            started_at=now,
            status=RunStatus.running,
            is_manual=is_manual,
            slot_at=slot_at,
            lease_expires_at=now + timedelta(seconds=lease_seconds) if lease_seconds is not None else None,
        )
        self.session.add(run)
        try:
            self.session.commit()
        except IntegrityError:
            self.session.rollback()
            if slot_at is not None and self._slot_holder(schedule_id, slot_at) is not None:
                raise SlotAlreadyClaimed(f"schedule {schedule_id} slot {slot_at.isoformat()} already has a run")
            if self._running_run(schedule_id) is not None:
                raise ScheduleBusy(f"schedule {schedule_id} already has a run going")
            raise
        self.session.refresh(run)
        return run

    def _running_run(self, schedule_id: UUID) -> ScheduleRun | None:
        return self.session.exec(
            self.session.select(ScheduleRun)
            .where(ScheduleRun.schedule_id == schedule_id)
            .where(ScheduleRun.status == RunStatus.running)
            .limit(1)
        ).first()

    def _slot_holder(self, schedule_id: UUID, slot_at: datetime) -> ScheduleRun | None:
        return self.session.exec(
            self.session.select(ScheduleRun)
            .where(ScheduleRun.schedule_id == schedule_id)
            .where(ScheduleRun.slot_at == slot_at)
            .limit(1)
        ).first()

    def renew_lease(self, run_id: UUID, lease_seconds: float) -> bool:
        """Extend a running run's lease. False once the run has finished or
        another replica reaped it, so its process no longer holds the claim."""
        run = self.session.get(ScheduleRun, run_id)
        if run is None or run.status != RunStatus.running:
            return False
        run.lease_expires_at = datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
        self.session.add(run)
        self.session.commit()
        return True

    def still_running(self, run_id: UUID) -> bool:
        """Whether the run is still `running`, i.e. no replica has reaped it.
        A core SELECT, for the reason still_exists gives."""
        raw = unsafe_unscoped_session(self.session)
        status = raw.execute(sa.select(ScheduleRun.status).where(ScheduleRun.id == run_id)).scalar()
        return status == RunStatus.running

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
        return self._running_run(schedule_id) is not None

    def last_successful_finish(self, schedule_id: UUID) -> datetime | None:
        """When the schedule's most recent successful run (manual or cron)
        finished, or None if it never succeeded. Used by the scheduler's
        freshness guard to skip a due slot right after e.g. a manual run.
        """
        run = self.session.exec(
            self.session.select(ScheduleRun)
            .where(ScheduleRun.schedule_id == schedule_id)
            .where(ScheduleRun.status == RunStatus.success)
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

    def set_run_conversation(self, run_id: UUID, conversation_id: UUID | None) -> bool:
        """Attach the run's conversation as soon as it is known — before the
        turn executes — so the UI can open a run that is still in flight.

        False when the run is gone or no longer `running`: another replica
        reaped it and may run its slot, so its turn must not start. The row is
        read FOR UPDATE, fresh from the database, and reap_lapsed_runs skips a
        row locked that way, so on PostgreSQL a reap and this link never both
        act on what the other has not yet committed."""
        run = self.session.exec(
            self.session.select(ScheduleRun)
            .where(ScheduleRun.id == run_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).first()
        if run is None or run.status != RunStatus.running:
            return False
        run.conversation_id = self.still_exists(conversation_id)
        # Persist before handing off: a crash after this point is ambiguous,
        # even when the user deletes the conversation. Never clear this claim.
        run.execution_claimed_at = run.execution_claimed_at or datetime.now(timezone.utc)
        self.session.add(run)
        try:
            self.session.commit()
        except IntegrityError:
            # The delete landed between still_exists' read and this commit,
            # and its release already nulled the column in the database. The
            # run simply has no conversation to point at. Retry the claim too:
            # rollback discarded it, and execution must not start without it.
            self.session.rollback()
            if conversation_id is None:
                raise
            return self.set_run_conversation(run_id, None)
        return True

    def finish_run(
        self,
        run_id: UUID,
        conversation_id: UUID | None = None,
        error: str | None = None,
        status: RunStatus | None = None,
        release_slot: bool = False,
        outcome: AttemptOutcome | None = None,
    ) -> ScheduleRun:
        """Record how the run ended. ``release_slot`` gives its cron slot back,
        for a run that left the slot due: the next poll can then claim it."""
        run = self.session.get(ScheduleRun, run_id)
        if run is None:
            raise ValueError("ScheduleRun not found")
        now = datetime.now(timezone.utc)
        run.finished_at = now
        started_at = run.started_at if run.started_at.tzinfo else run.started_at.replace(tzinfo=timezone.utc)
        run.duration_ms = int((now - started_at).total_seconds() * 1000)
        run.status = status or (RunStatus.failed if error else RunStatus.success)
        run.error = error
        run.lease_expires_at = None
        if outcome is not None:
            run.outcome = outcome
        if release_slot:
            run.slot_at = None
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
            return self.finish_run(
                run_id, error=error, status=status, release_slot=release_slot, outcome=outcome,
            )
        self.session.refresh(run)
        return run

    def reap_orphaned_runs(
        self,
        error: str = "Run orphaned by a server restart before it completed.",
        started_before: datetime | None = None,
    ) -> int:
        """Mark every run still in ``running`` as ``failed``.

        A crash/restart while a run is in flight leaves its ``ScheduleRun`` in
        ``running`` forever. Because the scheduler's due-check skips schedules
        with a running run, a single stale row wedges that schedule
        permanently. Returns the number of runs reaped.

        ``started_before`` limits it to runs that began before then. Where
        other processes share the database, a newer run may still be live on
        one of them, and reaping it would let the due-check fire its slot a
        second time.

        A reaped run gives its cron slot back, so the slot it left due fires
        again on the next poll, as it did before runs claimed slots.
        """
        query = self.session.select(ScheduleRun).where(ScheduleRun.status == RunStatus.running)
        if started_before is not None:
            query = query.where(ScheduleRun.started_at < started_before)
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
            run.slot_at = None
            run.lease_expires_at = None
            _record_lost_run(run)
            self.session.add(run)
        self.session.commit()
        return len(runs)

    def reap_lapsed_runs(self, *, now: datetime, error: str) -> list[ScheduleRun]:
        """Fail every running run whose lease ran out: its process stopped
        renewing it, so it died without recording how the run ended.

        A run that never claimed execution never started its turn, so it
        gives its cron slot back and the slot fires again. A run that did may
        have done its work, so it keeps the slot, and the caller consumes it
        rather than run that work twice. Returns those runs. Staged only: the
        caller commits.

        Rows are locked as they are read, skipping any row already locked: a
        run linking its conversation (set_run_conversation) or another
        replica's reap holds it, and the next poll reads it as committed.
        """
        runs = self.session.exec(
            self.session.select(ScheduleRun)
            .where(ScheduleRun.status == RunStatus.running)
            .where(ScheduleRun.lease_expires_at.is_not(None))  # type: ignore[union-attr]
            .where(ScheduleRun.lease_expires_at < now)  # type: ignore[operator]
            .with_for_update(skip_locked=True)
        ).all()
        kept: list[ScheduleRun] = []
        for run in runs:
            run.finished_at = now
            run.duration_ms = int((now - _utc(run.started_at)).total_seconds() * 1000)
            run.status = RunStatus.failed
            run.error = error
            run.lease_expires_at = None
            _record_lost_run(run)
            if run.execution_claimed_at is None:
                run.slot_at = None
            elif run.slot_at is not None:
                kept.append(run)
            self.session.add(run)
        return kept

    def last_finished_run(self, schedule_id: UUID) -> ScheduleRun | None:
        """Latest finished manual run or cron run that retained its occurrence."""
        parent = self.session.select(Schedule).where(Schedule.id == schedule_id)
        if self.session.scope.org_mode:
            parent = parent.where(Schedule.created_by == self.session.scope.user_id)
        if self.session.exec(parent).first() is None:
            return None
        return self.session.exec(
            self.session.select(ScheduleRun)
            .where(ScheduleRun.schedule_id == schedule_id)
            .where(ScheduleRun.status != RunStatus.running)
            .where(sa.or_(ScheduleRun.is_manual == sa.true(), ScheduleRun.slot_at.is_not(None)))
            .order_by(ScheduleRun.finished_at.desc(), ScheduleRun.started_at.desc())  # type: ignore[union-attr]
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
