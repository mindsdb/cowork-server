from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import TYPE_CHECKING, TypeVar
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError, InvalidRequestError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from cowork.common.datetime_utils import ensure_utc
from cowork.common.logger import get_logger, log_context
from cowork.db.scoped import (
    SYSTEM_SCOPE,
    MissingTenantScopeError,
    ScopedSession,
    scope_from_principal,
    unsafe_unscoped_session,
)
from cowork.db.units import run_db
from cowork.models.schedule import Schedule, ScheduleSkippedPeriod, ScheduleVersion
from cowork.schedule_timing import missed_slots, next_future_occurrence
from cowork.schemas.schedules import AttemptOutcome, Cadence, RunStatus, resolve_schedule_model
from cowork.services.schedules import (
    ScheduleOccurrenceService,
    current_version,
    ScheduleRunService,
    ScheduleService,
    Settlement,
    StartedAttempt,
)
from cowork.streaming.registry import registry
from cowork.streaming.stop import CANCEL_FLAG_TTL_SECONDS, remote_backend, stop_turn

if TYPE_CHECKING:
    from cowork.principal import Principal

logger = get_logger(__name__)

T = TypeVar("T")

_POLL_INTERVAL_SECONDS = 30
# Upper bound on a single run. A hung agent (stuck tool call, wedged stream)
# must not keep its ScheduleRun in `running` forever — that would block the
# schedule from ever firing again. On timeout the turn is stopped and the run
# recorded as interrupted. The registry's idle watchdog only bounds a turn
# that stops making progress; this bounds one that keeps going.
_MAX_RUN_DURATION_SECONDS = 600
# How long a run's last writes keep asking for a database connection while
# the pool is full. They must land: a run left at `running` reads as busy,
# and an outcome never recorded leaves its occurrence held until its lease
# lapses. Each try already waits POOL_TIMEOUT for a connection; the pause
# between tries only keeps a refusal that comes back at once from spinning.
_BOOKKEEPING_DEADLINE_SECONDS = _MAX_RUN_DURATION_SECONDS
_BOOKKEEPING_RETRY_PAUSE_SECONDS = 1.0
# A cron occurrence gets this many attempts before it settles failed, with
# these pauses before the second and later attempts. Only an attempt that
# failed before it acted is tried again. A manual run gets one: the user is
# watching and can press Run again.
_MAX_CRON_ATTEMPTS = 3
_RETRY_DELAYS_SECONDS = (60, 300)
_MANUAL_MAX_ATTEMPTS = 1
# Scheduled runs this process executes at once (see _Admission).
_MAX_CONCURRENT_RUNS = 3
# An attempt renews its lease this often while it runs. One whose lease
# lapses is presumed dead: its process exited, or has been out of touch with
# the database for several renewals.
_LEASE_SECONDS = 300
_LEASE_RENEW_SECONDS = 60
# How long a stop waits for the turn before saying it is late; the schedule
# stays held until the turn does stop. A pod cannot confirm it stopped, so
# its schedule is held for as long as the stop request stays live.
_STOP_GRACE_SECONDS = 30
_REMOTE_STOP_GRACE_SECONDS = CANCEL_FLAG_TTL_SECONDS
_scheduler_task: asyncio.Task | None = None

_RECURRING_CADENCES = {Cadence.hourly, Cadence.daily, Cadence.weekly, Cadence.weekdays}

# A one-off may start up to this late on its own. At this lateness or more
# (the app was not running when its time came) it is shown as missed, and
# runs only when the user presses Run now.
_ONCE_LATE_LIMIT = timedelta(hours=1)
# A recurring schedule that missed slots runs the latest one once, when that
# slot is no older than this; every other missed slot is passed over.
_MISSED_RUN_LOOKBACK = timedelta(days=7)

# Freshness guard (ENG-688): if a successful manual "run now" finished this
# recently before a due cron slot, the slot is skipped instead of executed,
# so both runs don't publish the same output twice. Only manual runs count:
# a cron run is its own slot. Hourly gets a tighter window so a run
# finishing mid-hour never suppresses the next slot.
_FRESHNESS_WINDOW_SECONDS = {
    Cadence.once: 60 * 60,
    Cadence.hourly: 30 * 60,
    Cadence.daily: 60 * 60,
    Cadence.weekdays: 60 * 60,
    Cadence.weekly: 60 * 60,
}


def _advance_next_run_at(schedule: Schedule, session) -> None:
    if schedule.cadence == Cadence.once:
        schedule.enabled = False
        session.add(schedule)
        return

    if schedule.cadence not in _RECURRING_CADENCES:
        return

    schedule.next_run_at = next_future_occurrence(
        schedule.cadence,
        schedule.next_run_at,
        schedule.timezone,
    )
    session.add(schedule)


def _apply_success_write_back(
    schedule: Schedule,
    run_service: ScheduleRunService,
    conversation_id: UUID | None,
    session,
) -> None:
    """Stage the bookkeeping for a run that completed: last run time, the
    pointer to the conversation it produced, and a clean error slate.

    The user can delete this chat while the run is still going, so check
    before pointing the schedule back at it. Staged only — the caller owns
    the commit, and its IntegrityError retry re-runs this with no
    conversation when the delete wins the race anyway.
    """
    schedule.last_run_at = datetime.now(timezone.utc)
    schedule.last_result_conversation_id = run_service.still_exists(conversation_id)
    schedule.last_error = None
    schedule.missed_runs = 0
    session.add(schedule)


def _locked_schedule(session: ScopedSession, schedule_id: UUID) -> Schedule | None:
    """The schedule as the database holds it now, locked until the caller's
    commit on Postgres, so a pause or another poller's write waits for it,
    or it for them. SQLite runs one writer at a time anyway."""
    return session.exec(
        session.select(Schedule)
        .where(Schedule.id == schedule_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    ).first()


def _catch_up(session: ScopedSession, schedule_id: UUID, now: datetime) -> None:
    """Bring one schedule that fell behind up to date, in one commit.

    A recurring schedule moves its slot to the latest missed one inside the
    lookback, which the due-check then runs once, or past now when there is
    none; the slots passed over are summarized. A one-off at or past the
    late limit is marked missed. Done under the schedule's lock and against
    its current row, so a repeat (this poll's next pass, or another poller)
    finds nothing left to do and records nothing twice.
    """
    schedule = _locked_schedule(session, schedule_id)
    if schedule is None or not schedule.enabled or schedule.paused_at or schedule.auto_paused_at:
        session.rollback()
        return
    seen = schedule.next_run_at
    next_run = ensure_utc(seen)
    if next_run > now:
        session.rollback()
        return
    if schedule.cadence == Cadence.once:
        if now - next_run < _ONCE_LATE_LIMIT:
            session.rollback()
            return
        moved = dict(enabled=False, missed_at=now)
        first = last = next_run
        count, reason = 1, "one_off_missed"
    elif schedule.cadence in _RECURRING_CADENCES:
        missed = missed_slots(
            schedule.cadence, next_run, schedule.timezone, now=now, lookback=_MISSED_RUN_LOOKBACK,
        )
        if not missed.skipped:
            session.rollback()
            return
        moved = dict(next_run_at=missed.run or missed.future)
        first, last = missed.first_skipped, missed.last_skipped
        count, reason = missed.skipped, "offline"
    else:
        session.rollback()
        return
    # Conditional on the slot this read: SQLite takes no row lock, so a
    # second poller that read the same row before this commit must not
    # record the same stretch again.
    landed = unsafe_unscoped_session(session).execute(
        sa.update(Schedule)
        .where(Schedule.id == schedule_id, Schedule.next_run_at == seen, Schedule.enabled == sa.true())
        .values(missed_runs=Schedule.missed_runs + count, **moved)
        .execution_options(synchronize_session=False)
    ).rowcount
    if not landed:
        session.rollback()
        return
    session.add(ScheduleSkippedPeriod(
        schedule_id=schedule_id, first_slot=first, last_slot=last, slot_count=count, reason=reason,
    ))
    session.commit()
    logger.info(f"Schedule {schedule_id}: passed over {count} missed slot(s) ({reason})")


def _handle_missed_runs(session: ScopedSession, now: datetime) -> None:
    """Bring every schedule that fell behind up to date (_catch_up).

    A schedule with an occurrence in flight is left alone: its slot is not
    due again until that occurrence settles, and the next poll then sees
    every slot that passed meanwhile.
    """
    occurrences = ScheduleOccurrenceService(session)
    behind = [s.id for s in ScheduleService(session).due(now)]
    session.commit()
    for schedule_id in behind:
        if not occurrences.has_active(schedule_id):
            _catch_up(session, schedule_id, now)


def _principal_for_schedule(schedule: Schedule) -> Principal | None:
    """Service principal for a scheduled run, derived from the schedule row.

    Delegates to `service_principal_for` from the scoped module, which handles
    the org vs. local mode logic. The custom error message below wraps it with
    schedule-specific context.
    """
    from cowork.db.scoped import service_principal_for

    try:
        return service_principal_for(schedule.org_id, schedule.created_by)
    except MissingTenantScopeError:
        raise MissingTenantScopeError(
            f"schedule {schedule.id} is missing org_id/created_by; "
            "cannot resolve a service principal to run it in org mode"
        )


@dataclass(frozen=True)
class _ScheduledTurn:
    """What a run reads off its schedule before the turn starts."""

    principal: Principal | None
    title: str
    prompt: str
    model: str | None
    project_id: UUID | None
    version: int


@dataclass(frozen=True)
class _TurnEvidence:
    """What a finished turn's own stream says about how it went."""

    # The terminal record's reason, or None when there is no terminal record.
    reason: str | None
    # Whether the stream holds anything beyond the turn's start, its progress
    # frames and its failure: an answer, a tool call, anything the agent did.
    acted: bool


@dataclass(frozen=True)
class _AttemptEnd:
    """How an attempt ended, by what is known of its execution."""

    outcome: AttemptOutcome
    error: str | None = None
    # Whether the same failure could go differently next time. Only an
    # attempt that failed before execution is ever tried again.
    retryable: bool = False


# Frames a turn writes before its agent can do anything, and its failure. A
# failed turn whose stream holds nothing else failed before it acted.
_PRE_EXECUTION_EVENTS = frozenset({
    "response.created", "response.in_progress", "response.queued", "response.failed",
})


class _Admission:
    """Scheduled runs this process executes at once, cron and run-now alike.

    A due slot past the cap stays due and a later poll claims it; a run-now
    past the cap is refused with a reason. Each process keeps its own count:
    this bounds one process's load, not the fleet's.
    """

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._lock = threading.Lock()
        self._held: set[UUID] = set()
        self._reserved = 0

    def free(self) -> int:
        with self._lock:
            return max(self._limit - len(self._held) - self._reserved, 0)

    def holds(self, occurrence_id: UUID) -> bool:
        with self._lock:
            return occurrence_id in self._held

    def admit(self, occurrence_id: UUID) -> None:
        """Count an occurrence the caller already sized against free()."""
        with self._lock:
            self._held.add(occurrence_id)

    def reserve(self) -> bool:
        """Hold a place for a run-now before it claims, or False when full."""
        with self._lock:
            if len(self._held) + self._reserved >= self._limit:
                return False
            self._reserved += 1
            return True

    def fill(self, occurrence_id: UUID) -> None:
        with self._lock:
            self._reserved -= 1
            self._held.add(occurrence_id)

    def unreserve(self) -> None:
        with self._lock:
            self._reserved -= 1

    def release(self, occurrence_id: UUID) -> None:
        with self._lock:
            self._held.discard(occurrence_id)


admission = _Admission(_MAX_CONCURRENT_RUNS)


def _slot_key(scheduled_for: datetime) -> str:
    return f"slot:{ensure_utc(scheduled_for).isoformat()}"


def _retry_delay(attempt: int) -> float:
    return _RETRY_DELAYS_SECONDS[min(attempt, len(_RETRY_DELAYS_SECONDS)) - 1]


def _claim_slot(session: ScopedSession, schedule: Schedule) -> UUID | None:
    """Claim the schedule's due slot and move it on, in one commit.

    The slot is consumed at the claim, not when the run ends: a slot whose run
    fails, is cancelled or never finishes cannot come due again. None when
    another poller already holds this slot, or the schedule still has an
    occurrence in flight; the slot then stays where it was.
    """
    scheduled_for = ensure_utc(schedule.next_run_at)
    schedule_id = schedule.id
    # Claimed against the row as it stands under its lock: a pause, or a
    # poller that already moved the slot on, leaves nothing to claim.
    schedule = _locked_schedule(session, schedule_id)
    if (
        schedule is None
        or not schedule.enabled
        or schedule.paused_at is not None
        or schedule.auto_paused_at is not None
        or ensure_utc(schedule.next_run_at) != scheduled_for
    ):
        session.rollback()
        return None
    version_id = current_version(session, schedule).id
    _advance_next_run_at(schedule, session)
    occurrence = ScheduleOccurrenceService(session).claim(
        schedule_id,
        occurrence_key=_slot_key(scheduled_for),
        trigger_source="cron",
        scheduled_for=scheduled_for,
        max_attempts=_MAX_CRON_ATTEMPTS,
        lease_seconds=_LEASE_SECONDS,
        version_id=version_id,
    )
    return occurrence.id if occurrence else None


def claim_manual_run(session: ScopedSession, schedule_id: UUID) -> UUID | None:
    """Claim a run-now occurrence, or None while the schedule is busy.

    Running a one-off now uses it up, missed or not: it is disabled in the
    same commit as the claim, so its own time does not fire it a second time.
    A refused claim rolls that back. A recurring schedule keeps its slot; the
    freshness guard skips a slot due right after a manual success.
    """
    schedule = _locked_schedule(session, schedule_id)
    if schedule is None:
        session.rollback()
        return None
    if schedule.cadence == Cadence.once:
        schedule.enabled = False
        schedule.missed_at = None
        session.add(schedule)
    occurrence = ScheduleOccurrenceService(session).claim(
        schedule_id,
        occurrence_key=f"manual:{uuid4()}",
        trigger_source="manual",
        scheduled_for=datetime.now(timezone.utc),
        max_attempts=_MANUAL_MAX_ATTEMPTS,
        lease_seconds=_LEASE_SECONDS,
        version_id=current_version(session, schedule).id,
    )
    return occurrence.id if occurrence else None


def _claim_for_schedule(session: ScopedSession, *, schedule_id: UUID, is_manual: bool) -> UUID | None:
    if is_manual:
        return claim_manual_run(session, schedule_id)
    return _claim_slot(session, ScheduleService(session).get_schedule(schedule_id))


def _start_attempt(session: ScopedSession, *, occurrence_id: UUID, is_manual: bool) -> StartedAttempt | None:
    return ScheduleOccurrenceService(session).start_attempt(
        occurrence_id, is_manual=is_manual, lease_seconds=_LEASE_SECONDS
    )


def _renew_lease(session: ScopedSession, *, occurrence_id: UUID, token: str) -> bool:
    return ScheduleOccurrenceService(session).renew(occurrence_id, token, lease_seconds=_LEASE_SECONDS)


def _begin_execution(session: ScopedSession, *, attempt: StartedAttempt, occurrence_id: UUID) -> bool:
    return ScheduleOccurrenceService(session).begin_execution(
        attempt, occurrence_id, lease_seconds=_LEASE_SECONDS
    )


def _request_stop(session: ScopedSession, *, run_id: UUID) -> None:
    ScheduleOccurrenceService(session).request_stop(run_id)


def _read_schedule(session: ScopedSession, *, schedule_id: UUID, occurrence_id: UUID) -> _ScheduledTurn:
    """What the attempt runs: its occurrence's version. The identity it runs
    as, and the title its conversation takes, are read from the schedule as
    it is now."""
    schedule = ScheduleService(session).get_schedule(schedule_id)
    # A scheduled run has no request, so it derives its tenant identity from
    # the schedule row (see _principal_for_schedule). None in local mode.
    try:
        principal = _principal_for_schedule(schedule)
    except MissingTenantScopeError:
        # A corrupt row (NULL org in org mode) can never resolve an
        # identity, so it can never run. Disable it, else the unadvanced
        # next_run_at keeps the slot due and every poll re-fires the same
        # failure. The re-raise records the run as failed.
        schedule.enabled = False
        session.add(schedule)
        session.commit()
        raise
    occurrence = ScheduleOccurrenceService(session).get(occurrence_id)
    version = (
        session.get(ScheduleVersion, occurrence.version_id)
        if occurrence is not None and occurrence.version_id is not None
        else None
    ) or current_version(session, schedule)
    return _ScheduledTurn(
        principal=principal,
        title=schedule.title,
        prompt=version.prompt,
        model=version.model,
        project_id=version.project_id,
        version=version.number,
    )


def _link_conversation(
    session: ScopedSession, *, run_id: UUID, scheduled: _ScheduledTurn, conversation_id: UUID | None,
) -> UUID:
    """The run's conversation, created first when the caller made none, and
    recorded on the run before the turn starts."""
    if conversation_id is None:
        from cowork.services.conversations import ConversationService

        # Conversation not pre-created by the caller (e.g. cron tick). Create
        # it under the schedule's OWN scope so org mode stamps the owning
        # org_id: the scheduler's SYSTEM_SCOPE is deliberately unscoped (it
        # scans every org), so creating through `session` would write an
        # invisible NULL-org row.
        conv_session = ScopedSession(
            unsafe_unscoped_session(session), scope_from_principal(scheduled.principal)
        )
        conversation_id = ConversationService(conv_session).create_conversation(
            topic=scheduled.title,
            project_id=scheduled.project_id,
        ).id
    ScheduleRunService(session).set_run_conversation(run_id, conversation_id)
    return conversation_id


def _run_status(outcome: AttemptOutcome) -> RunStatus:
    if outcome == AttemptOutcome.completed:
        return RunStatus.success
    if outcome == AttemptOutcome.cancelled:
        return RunStatus.cancelled
    return RunStatus.failed


def _stage_attempt_end(
    session: ScopedSession,
    *,
    attempt: StartedAttempt,
    occurrence_id: UUID,
    end: _AttemptEnd,
    retry_delay: float | None,
    conversation_id: UUID | None,
) -> Settlement:
    occurrences = ScheduleOccurrenceService(session)
    run_service = ScheduleRunService(session)
    settlement = occurrences.end_attempt(
        occurrence_id, attempt.token, outcome=end.outcome, error=end.error, retry_delay=retry_delay,
    )
    if settlement != Settlement.lost:
        # Read again: the schedule may have changed while the turn ran.
        schedule = ScheduleService(session).get_schedule(attempt.schedule_id)
        if end.outcome == AttemptOutcome.completed:
            _apply_success_write_back(schedule, run_service, conversation_id, session)
        elif end.outcome != AttemptOutcome.cancelled:
            schedule.last_error = end.error
            session.add(schedule)
    run_service.end_run(
        attempt.run_id,
        status=_run_status(end.outcome),
        error=end.error,
        outcome=end.outcome,
        conversation_id=conversation_id,
    )
    return settlement


def _record_attempt_end(session: ScopedSession, **kwargs) -> Settlement:
    """Record how the attempt ended, on its run, its occurrence and its
    schedule, in one commit.

    Lands on the occurrence and the schedule only while the attempt still
    holds its occurrence; a holder that was reaped while out of touch writes
    nothing there, so a settlement is never counted twice.
    """
    try:
        settlement = _stage_attempt_end(session, **kwargs)
        session.commit()
    except IntegrityError:
        # Only the conversation pointers stage a foreign-key write, so this is
        # the user's delete landing between still_exists' read and this
        # commit; the delete's release already nulled the columns. Redo it
        # without them so the bookkeeping and the settlement survive.
        session.rollback()
        settlement = _stage_attempt_end(session, **{**kwargs, "conversation_id": None})
        session.commit()
    return settlement


def _finish_run(session: ScopedSession, *, run_id: UUID, end: _AttemptEnd) -> None:
    """End the run alone, when its full record could not be written. Its
    occurrence keeps its lease and is reaped once that lapses."""
    if ScheduleRunService(session).end_run(
        run_id, status=_run_status(end.outcome), error=end.error, outcome=end.outcome,
    ):
        session.commit()


def _is_retryable(exc: BaseException) -> bool:
    """Whether trying the attempt again could go differently. A refused
    permission, a row that can never resolve an identity, or a schedule that
    is gone will fail the same way every time."""
    from cowork.services.product_permissions import ProductPermissionDenied

    if isinstance(exc, (ProductPermissionDenied, MissingTenantScopeError)):
        return False
    if isinstance(exc, ValueError) and "not found" in str(exc).lower():
        return False
    return True


async def _until_it_lands(unit: Callable[[ScopedSession], T], *, schedule_id: UUID) -> T:
    """Run one of a run's last bookkeeping units, trying again while no
    database connection frees, for up to _BOOKKEEPING_DEADLINE_SECONDS.

    A unit refused for a full pool never ran (cowork.db.units.run_db), so
    trying again cannot write anything twice.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _BOOKKEEPING_DEADLINE_SECONDS
    while True:
        try:
            return await run_db(unit, scope=SYSTEM_SCOPE)
        except PoolTimeoutError:
            if loop.time() >= deadline:
                raise
            logger.warning(
                f"Schedule {schedule_id}: no database connection freed to record its run; trying again"
            )
            await asyncio.sleep(_BOOKKEEPING_RETRY_PAUSE_SECONDS)


class _LeaseKeeper:
    """Renews an attempt's lease while it runs, so a long run, or one still
    stopping, is never taken for a dead one and its occurrence handed on."""

    def __init__(self, occurrence_id: UUID, token: str, schedule_id: UUID) -> None:
        self._occurrence_id = occurrence_id
        self._token = token
        self._schedule_id = schedule_id
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._renew())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)

    async def _renew(self) -> None:
        while True:
            await asyncio.sleep(_LEASE_RENEW_SECONDS)
            try:
                held = await run_db(
                    partial(_renew_lease, occurrence_id=self._occurrence_id, token=self._token),
                    scope=SYSTEM_SCOPE,
                )
            except Exception:
                # The next renewal tries again; the lease has slack for a few.
                logger.warning(f"Schedule {self._schedule_id}: could not renew its run's lease", exc_info=True)
                continue
            if not held:
                logger.warning(
                    f"Schedule {self._schedule_id}: occurrence {self._occurrence_id} was taken back "
                    "while its run was out of touch; its outcome will not be recorded on it"
                )
                return


async def execute_schedule(
    schedule_id: UUID,
    is_manual: bool = False,
    conversation_id: UUID | None = None,
) -> None:
    """Claim an occurrence of the schedule and run its first attempt.

    A cron claim takes the schedule's due slot; a manual claim is a fresh
    run-now. Nothing runs when the claim is lost: the slot is someone else's,
    or the schedule already has an occurrence in flight.
    """
    try:
        occurrence_id = await run_db(
            partial(_claim_for_schedule, schedule_id=schedule_id, is_manual=is_manual), scope=SYSTEM_SCOPE,
        )
    except PoolTimeoutError:
        # Nothing was written: a cron slot stays due and the next poll claims it.
        logger.warning(
            f"Schedule {schedule_id}: no database connection freed to claim its run; it did not run"
        )
        return
    if occurrence_id is None:
        logger.info(f"Schedule {schedule_id}: an occurrence is already in flight; not starting another")
        return
    await run_occurrence(occurrence_id, is_manual=is_manual, conversation_id=conversation_id)


async def run_occurrence(
    occurrence_id: UUID,
    *,
    is_manual: bool = False,
    conversation_id: UUID | None = None,
) -> None:
    """Run the next attempt at a claimed occurrence and record how it went."""
    admission.admit(occurrence_id)
    try:
        await _run_attempt(occurrence_id, is_manual=is_manual, conversation_id=conversation_id)
    finally:
        admission.release(occurrence_id)


async def _run_attempt(
    occurrence_id: UUID,
    *,
    is_manual: bool,
    conversation_id: UUID | None,
) -> None:
    """Run one attempt's turn and record how it went.

    Each read and write is its own database unit, so the run holds no pooled
    connection across the permission check or while its turn runs, and a
    wait for a connection never stalls the event loop. The attempt's lease is
    renewed for as long as it runs. Its end is recorded in one unit that
    tries again while the pool is full (_until_it_lands). An attempt that
    failed before execution hands its occurrence back for a later poll to try
    again, until its attempts are spent; any other failure settles it.
    """
    try:
        attempt = await run_db(
            partial(_start_attempt, occurrence_id=occurrence_id, is_manual=is_manual), scope=SYSTEM_SCOPE,
        )
    except PoolTimeoutError:
        # Nothing was written: the occurrence still waits for this attempt,
        # and the next poll (or its lease running out) starts it.
        logger.warning(
            f"Occurrence {occurrence_id}: no database connection freed to start its attempt; it did not run"
        )
        return
    if attempt is None:
        return
    schedule_id = attempt.schedule_id
    lease = _LeaseKeeper(occurrence_id, attempt.token, schedule_id)
    lease.start()
    end = _AttemptEnd(outcome=AttemptOutcome.uncertain, error="Run ended without saying how.")
    try:
        end, conversation_id = await _execute_attempt(
            attempt, occurrence_id, is_manual=is_manual, conversation_id=conversation_id,
        )
    except Exception as exc:
        # _execute_attempt answers every failure it can place; this is one
        # it could not, so nothing is assumed about what ran.
        logger.exception(
            "Schedule run failed: %s", exc,
            extra=log_context(schedule_id=schedule_id, conversation_id=conversation_id),
        )
    finally:
        retry_delay = (
            _retry_delay(attempt.number)
            if end.retryable and end.outcome == AttemptOutcome.failed_before_execution
            else None
        )
        try:
            await _until_it_lands(
                partial(
                    _record_attempt_end, attempt=attempt, occurrence_id=occurrence_id, end=end,
                    retry_delay=retry_delay, conversation_id=conversation_id,
                ),
                schedule_id=schedule_id,
            )
        except Exception:
            logger.exception(
                "Failed to record how the run ended",
                extra=log_context(schedule_id=schedule_id, conversation_id=conversation_id),
            )
            try:
                # The run must not strand at `running`; the occurrence keeps
                # its lease and is reaped once it lapses.
                await _until_it_lands(partial(_finish_run, run_id=attempt.run_id, end=end), schedule_id=schedule_id)
            except Exception:
                logger.exception(
                    "Failed to finish the run record",
                    extra=log_context(schedule_id=schedule_id, conversation_id=conversation_id),
                )
        await lease.stop()


async def _execute_attempt(
    attempt: StartedAttempt,
    occurrence_id: UUID,
    *,
    is_manual: bool,
    conversation_id: UUID | None,
) -> tuple[_AttemptEnd, UUID | None]:
    """Set the attempt's turn up, hand it to the agent, and say how it ended.

    Everything before the hand-off is known not to have acted, so its
    failures may be tried again. The hand-off is recorded first, so a process
    that dies past it leaves a run the reaper will not replay.
    """
    from cowork.handlers.responses import ResponsesHandler
    from cowork.schemas.responses import ResponsesRequest
    from cowork.services.product_permissions import require_product_permission

    schedule_id = attempt.schedule_id
    try:
        scheduled = await run_db(
            partial(_read_schedule, schedule_id=schedule_id, occurrence_id=occurrence_id), scope=SYSTEM_SCOPE,
        )

        await require_product_permission(scope_from_principal(scheduled.principal), "product.execute")

        conversation_id = await run_db(
            partial(_link_conversation, run_id=attempt.run_id, scheduled=scheduled, conversation_id=conversation_id),
            scope=SYSTEM_SCOPE,
        )

        # Stamp the run's identity on the Langfuse trace (existing pass-through
        # seam) so incident forensics don't have to reconstruct which schedule/
        # trigger produced a turn from timestamps.
        trigger = "manual" if is_manual else "cron"
        request = ResponsesRequest(
            # `schedule.model` is the "default" sentinel for every task the UI
            # creates, not a servable id. Resolve it to None so the harness
            # applies the account's configured default models instead of
            # overriding every role with the literal string.
            input=scheduled.prompt,
            model=resolve_schedule_model(scheduled.model),
            stream=True,
            conversation=str(conversation_id),
            trace_tags=["scheduled_task", f"trigger:{trigger}"],
            trace_metadata={
                "schedule_id": str(schedule_id),
                "schedule_run_id": str(attempt.run_id),
                "schedule_occurrence_id": str(occurrence_id),
                "trigger_type": trigger,
                "attempt": str(attempt.number),
                "schedule_version": str(scheduled.version),
            },
        )

        if not await run_db(
            partial(_begin_execution, attempt=attempt, occurrence_id=occurrence_id), scope=SYSTEM_SCOPE,
        ):
            # Taken back while it was setting up: whoever holds the
            # occurrence now decides what runs. This attempt never acted.
            logger.warning(f"Schedule {schedule_id}: run lost its occurrence before it started; not running it")
            return _AttemptEnd(
                outcome=AttemptOutcome.failed_before_execution,
                error="Run was taken over before it started.",
            ), conversation_id

        # The schedule-derived principal is what lets the turn (and the remote
        # backend's per-tenant key mint) run in org mode with no request in
        # flight. The handler opens its own short sessions. A raise here comes
        # before the turn's producer exists, so nothing has acted yet.
        stream = await ResponsesHandler(principal=scheduled.principal, interactive=False).handle(request)
    except Exception as exc:
        error = str(exc) or type(exc).__name__
        # The ids travel as record attributes: a database error's sanitized
        # line keeps those and drops the message.
        logger.exception(
            "Schedule run failed: %s", error,
            extra=log_context(schedule_id=schedule_id, conversation_id=conversation_id),
        )
        return _AttemptEnd(
            outcome=AttemptOutcome.failed_before_execution, error=error, retryable=_is_retryable(exc),
        ), conversation_id

    async def _drain() -> None:
        async for _ in stream:
            pass

    timed_out = False
    lost_track = False
    try:
        await asyncio.wait_for(_drain(), timeout=_MAX_RUN_DURATION_SECONDS)
    except asyncio.TimeoutError:
        timed_out = True
        logger.warning(
            f"Schedule {schedule_id}: run exceeded {_MAX_RUN_DURATION_SECONDS}s; stopping it",
            extra=log_context(schedule_id=schedule_id, conversation_id=conversation_id),
        )
    except Exception:
        # The tail broke, which says nothing about the turn: it may still be
        # running. Stop it rather than leave it running unwatched.
        lost_track = True
        logger.exception(
            "Lost the scheduled run's stream; stopping its turn",
            extra=log_context(schedule_id=schedule_id, conversation_id=conversation_id),
        )
    if timed_out or lost_track:
        await _stop_turn(str(conversation_id), run_id=attempt.run_id, schedule_id=schedule_id)

    # A cancel or a producer failure closes the stream normally from the
    # consumer's side (the producer runs detached and even swallows its own
    # CancelledError, so handle.task.cancelled() stays False). The buffer is
    # the only truthful account of how the turn ended — without it every run
    # is recorded as success.
    evidence = await _turn_evidence(str(conversation_id))
    return _classify(evidence, timed_out=timed_out), conversation_id


async def _stop_turn(conversation_id: str, *, run_id: UUID, schedule_id: UUID) -> None:
    """Stop the run's turn the way the Stop button does, saved as
    interrupted, and return once it has stopped.

    The occurrence stays held (its lease is renewed) until then, so no retry
    or run-now starts beside a turn still running here. A turn running in a
    pod cannot confirm it stopped: the occurrence is held for as long as the
    stop request stays live, and the run's outcome makes no claim either way.
    """
    try:
        await run_db(partial(_request_stop, run_id=run_id), scope=SYSTEM_SCOPE)
    except Exception:
        logger.warning(f"Schedule {schedule_id}: could not record the stop request", exc_info=True)
    handle = registry.get(conversation_id)
    if handle is None or not handle.is_running:
        return
    stopping = asyncio.ensure_future(stop_turn(handle, timed_out=True))
    try:
        await asyncio.wait_for(asyncio.shield(stopping), timeout=_STOP_GRACE_SECONDS)
    except asyncio.TimeoutError:
        logger.error(
            f"Schedule {schedule_id}: its turn did not stop within {_STOP_GRACE_SECONDS}s; "
            "holding the schedule until it does"
        )
    except Exception:
        logger.exception(f"Schedule {schedule_id}: stopping its turn failed; holding the schedule until it ends")
    await asyncio.wait({handle.task})
    if remote_backend():
        await asyncio.sleep(_REMOTE_STOP_GRACE_SECONDS)


_DURATION_ERROR = (
    f"Run exceeded the {_MAX_RUN_DURATION_SECONDS // 60}-minute limit and was stopped. "
    "Some of its actions may have completed — open the run to check before running it again."
)
_INTERRUPTED_ERROR = (
    "Run was interrupted before it finished. Some of its actions may have completed — "
    "open the run to check before running it again."
)
_FAILED_DURING_ERROR = "Run did not complete — open the run's task for details."
_FAILED_BEFORE_ERROR = "Run failed before it started work — open the run's task for details."
_UNCERTAIN_ERROR = (
    "Run ended without a record of how it finished. Open the run to check what it did "
    "before running it again."
)


def _classify(evidence: _TurnEvidence, *, timed_out: bool) -> _AttemptEnd:
    """What the turn's ending says about whether it acted."""
    if timed_out:
        return _AttemptEnd(outcome=AttemptOutcome.interrupted, error=_DURATION_ERROR)
    reason = evidence.reason
    if reason == "completed":
        return _AttemptEnd(outcome=AttemptOutcome.completed)
    if reason == "cancelled":
        logger.info("Scheduled run was cancelled")
        return _AttemptEnd(outcome=AttemptOutcome.cancelled)
    if reason in ("interrupted", "restart"):
        return _AttemptEnd(outcome=AttemptOutcome.interrupted, error=_INTERRUPTED_ERROR)
    if reason is None:
        return _AttemptEnd(outcome=AttemptOutcome.uncertain, error=_UNCERTAIN_ERROR)
    if not evidence.acted:
        return _AttemptEnd(
            outcome=AttemptOutcome.failed_before_execution, error=_FAILED_BEFORE_ERROR, retryable=True,
        )
    return _AttemptEnd(outcome=AttemptOutcome.failed_during_execution, error=_FAILED_DURING_ERROR)


def _sse_events(record) -> list[str]:
    sse = record.data.get("sse") if isinstance(record.data, dict) else None
    if not isinstance(sse, str):
        return []
    return [line[len("event:"):].strip() for line in sse.splitlines() if line.startswith("event:")]


async def _turn_evidence(conversation_id: str) -> _TurnEvidence:
    """How the turn that just ended on this conversation went, from its
    buffer: the terminal reason, and whether anything beyond its start and
    failure was written. Unknown (no reason, acted) when unavailable.

    Only call after the turn's stream has ended: the buffer is closed then,
    so reading it returns at the terminal record."""
    handle = registry.get(conversation_id)
    if handle is None or not handle.buffer.is_closed:
        return _TurnEvidence(reason=None, acted=True)
    acted = False
    try:
        async for rec in handle.buffer.tail(0):
            if rec.is_terminal:
                return _TurnEvidence(reason=str(rec.data.get("reason") or "") or None, acted=acted)
            events = _sse_events(rec)
            if not events or any(event not in _PRE_EXECUTION_EVENTS for event in events):
                acted = True
    except Exception:
        logger.exception(f"Could not read terminal state for conversation {conversation_id}")
    return _TurnEvidence(reason=None, acted=True)


def _ran_recently(schedule: Schedule, run_service: ScheduleRunService, now: datetime) -> bool:
    window = _FRESHNESS_WINDOW_SECONDS.get(schedule.cadence)
    if not window:
        return False
    last = run_service.last_successful_finish(schedule.id, manual_only=True)
    return last is not None and (now - last).total_seconds() < window


def _due_schedules(session, now: datetime) -> list[Schedule]:
    """Enabled schedules whose slot is due and should actually execute.

    A due slot with a successful run inside the freshness window is skipped
    and advanced to its next occurrence instead of returned.
    """
    run_service = ScheduleRunService(session)
    occurrences = ScheduleOccurrenceService(session)
    due: list[Schedule] = []
    skipped = False
    for s in ScheduleService(session).due(now):
        # Gate on ANY occurrence in flight, manual included, or waiting for a
        # retry: the slot is deferred, not consumed, and once the occurrence
        # settles the freshness guard decides whether it still fires. The
        # claim enforces this too; checking first saves a losing insert.
        if not s.enabled or ensure_utc(s.next_run_at) > now:
            continue
        if occurrences.has_active(s.id) or run_service.has_active_run(s.id):
            continue
        # Read the row again: a run that ended after this poll read it has
        # already consumed its slot or disabled its one-off, and the poll's
        # session still holds the row as it was.
        try:
            session.refresh(s)
        except InvalidRequestError:
            continue  # deleted since this poll read it
        if not s.enabled or ensure_utc(s.next_run_at) > now:
            continue
        if _ran_recently(s, run_service, now):
            logger.info(
                f"Schedule {s.id}: skipping due slot — a successful run "
                "finished within the freshness window"
            )
            _advance_next_run_at(s, session)
            skipped = True
            continue
        due.append(s)
    if skipped:
        session.commit()
    return due


_LEASE_EXPIRED_ERROR = (
    "Run stopped responding and was abandoned. If it had started work, some of its "
    "actions may have completed — open the run to check before running it again."
)


def _reap_expired_leases(session: ScopedSession, now: datetime) -> None:
    """Take back occurrences whose holder stopped renewing its lease.

    An attempt that never handed its turn to the agent is tried again; one
    that did may have acted, so its occurrence settles without a replay. One
    this process is running is left alone, and one whose holder renewed or
    settled since this poll read it is too (ScheduleOccurrenceService.reap).
    """
    occurrences = ScheduleOccurrenceService(session)
    for lease in occurrences.expired_leases(now):
        if admission.holds(lease.occurrence_id):
            continue
        settlement = occurrences.reap(
            lease, now=now, error=_LEASE_EXPIRED_ERROR,
            retry_delay=lambda attempts: _retry_delay(attempts) if attempts else 0,
        )
        if settlement != Settlement.lost:
            logger.warning(
                f"Schedule {lease.schedule_id}: occurrence {lease.occurrence_id} lease expired; "
                f"reaped it ({settlement.value})"
            )


def _poll(session: ScopedSession, capacity: int = _MAX_CONCURRENT_RUNS) -> list[UUID]:
    """One poll's unit: reap dead occurrences, settle missed runs, then claim.

    Returns the occurrences to attempt now — retries that came due first,
    then slots claimed this poll — at most ``capacity`` of them.
    """
    now = datetime.now(timezone.utc)
    _reap_expired_leases(session, now)
    _handle_missed_runs(session, now)
    ready = [o for o in ScheduleOccurrenceService(session).ready_for_attempt(now) if not admission.holds(o)]
    ready = ready[: max(capacity, 0)]
    for schedule in _due_schedules(session, now):
        if len(ready) >= capacity:
            break
        claimed = _claim_slot(session, schedule)
        if claimed is not None:
            ready.append(claimed)
    return ready


async def _scheduler_loop() -> None:
    logger.info("Scheduler loop started")
    while True:
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        capacity = admission.free()
        try:
            # A unit in a worker thread: when the pool has no connection to
            # give, the poll is refused after POOL_TIMEOUT instead of stalling
            # the event loop, and the next poll tries again.
            ready = await run_db(partial(_poll, capacity=capacity), scope=SYSTEM_SCOPE)
        except PoolTimeoutError:
            logger.warning("Scheduler poll found no free database connection; the next poll retries")
            ready = []
        except Exception:
            logger.exception("Scheduler loop error during poll")
            ready = []

        for occurrence_id in ready:
            # Counted before the task starts, so the next poll sizes its
            # claims by it even if the task has not run yet.
            admission.admit(occurrence_id)
            asyncio.create_task(run_occurrence(occurrence_id, is_manual=False))


def boot_reap_cutoff(org_mode: bool, now: datetime | None = None) -> datetime | None:
    """Which running runs a starting server may reap: all of them on the
    desktop, where it is the only process, or in org mode, where another
    replica may still be running them, only runs from before occurrences that
    are older than a lease. Leases release the rest once they lapse."""
    if not org_mode:
        return None
    return (now or datetime.now(timezone.utc)) - timedelta(seconds=_LEASE_SECONDS)


def start_scheduler() -> None:
    global _scheduler_task
    if _scheduler_task is not None and not _scheduler_task.done():
        return
    _scheduler_task = asyncio.create_task(_scheduler_loop())
    logger.info("Scheduler background task created")
