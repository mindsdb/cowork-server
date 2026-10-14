from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import TYPE_CHECKING, TypeVar
from uuid import UUID, uuid4

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
from cowork.models.schedule import Schedule, ScheduleOccurrence
from cowork.schedule_timing import advance_occurrence, overdue_slots
from cowork.schemas.schedules import Cadence, MissedRunPolicy, OccurrenceStatus, RunStatus, resolve_schedule_model
from cowork.services.schedules import ScheduleOccurrenceService, ScheduleRunService, ScheduleService
from cowork.streaming.registry import registry

if TYPE_CHECKING:
    from cowork.principal import Principal

logger = get_logger(__name__)

T = TypeVar("T")

_POLL_INTERVAL_SECONDS = 30
# Upper bound on a single run. A hung agent (stuck tool call, wedged stream)
# must not keep its ScheduleRun in `running` forever — that would block the
# schedule from ever firing again. On timeout the run is recorded as failed.
_MAX_RUN_DURATION_SECONDS = 600
# How long a run's last writes, its outcome and its finish, keep asking for a
# database connection while the pool is full. They must land: a run left at
# `running` stops its schedule firing until a restart, and an outcome never
# recorded leaves its slot due, so the slot would run a second time. Each try
# already waits POOL_TIMEOUT for a connection; the pause between tries only
# keeps a refusal that comes back at once from spinning.
_BOOKKEEPING_DEADLINE_SECONDS = _MAX_RUN_DURATION_SECONDS
_BOOKKEEPING_RETRY_PAUSE_SECONDS = 1.0
# A cron occurrence gets this many attempts before it settles failed, with
# these pauses before the second and later attempts. A manual run gets one:
# the user is watching and can press Run again.
_MAX_CRON_ATTEMPTS = 3
_RETRY_DELAYS_SECONDS = (60, 300)
_MANUAL_MAX_ATTEMPTS = 1
# Occurrences this process runs at once. A due slot past the cap stays due and
# a later poll claims it, rather than every due task starting a turn together.
_MAX_CONCURRENT_RUNS = 3
# How long an attempt may hold its occurrence before a poll presumes the
# process running it died: the turn's own bound, plus the time its last writes
# may spend waiting for a connection, plus slack.
_LEASE_SECONDS = _MAX_RUN_DURATION_SECONDS + _BOOKKEEPING_DEADLINE_SECONDS + 60
_scheduler_task: asyncio.Task | None = None
# Occurrences this process is running now; the poll sizes its claims by it and
# never reaps one of its own.
_inflight: set[UUID] = set()

_RECURRING_CADENCES = {Cadence.hourly, Cadence.daily, Cadence.weekly, Cadence.weekdays}

# How late a slot may be and still count as on time rather than missed. The
# `skip` policy still runs a slot this late, and a one-off under it is
# disabled only once it is later than this. The poll runs every 30s, so an
# on-time slot is never close to it.
_LATE_GRACE_SECONDS = 60 * 60
# Missed slots older than this are never run, whatever the policy; they are
# counted in missed_runs only.
_MISSED_RUN_LOOKBACK = timedelta(days=7)
# The most missed slots `bounded_catch_up` will run after an outage.
_MAX_CATCH_UP = 10
# Cron occurrences in a row that may settle failed before the schedule pauses
# itself. Retries within one occurrence count once.
_MAX_FAILED_OCCURRENCES = 3

# Freshness guard (ENG-688): if a successful manual "run now" finished this
# recently before a due cron slot, the slot is skipped instead of executed,
# so both runs don't publish the same output twice. Only manual runs count: a
# cron run is its own slot, and counting it would skip the slots a catch-up
# runs back to back. Hourly gets a tighter window so a run finishing mid-hour
# never suppresses the next slot.
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

    # One slot on, not straight past now: a catch-up runs the missed slots in
    # turn, and the missed-run policy repositions anything else left behind.
    schedule.next_run_at = advance_occurrence(
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


def _select_missed(policy: str, slots: list[datetime], now: datetime) -> list[datetime]:
    """Which of a schedule's overdue slots (oldest first, all inside the
    lookback) its missed-run policy runs."""
    if not slots:
        return []
    if policy == MissedRunPolicy.skip:
        latest = slots[-1]
        return [latest] if (now - latest).total_seconds() <= _LATE_GRACE_SECONDS else []
    if policy == MissedRunPolicy.bounded_catch_up:
        return slots[-_MAX_CATCH_UP:]
    return slots[-1:]


def _record_skipped(session, schedule_id, slots: list[datetime]) -> None:
    """Stage one settled `skipped` occurrence per slot passed over, so the
    history shows what did not run. A slot already recorded is left as is."""
    if not slots:
        return
    keys = {_slot_key(slot): slot for slot in slots}
    taken = set(
        session.exec(
            session.select(ScheduleOccurrence.occurrence_key)
            .where(ScheduleOccurrence.schedule_id == schedule_id)
            .where(ScheduleOccurrence.occurrence_key.in_(list(keys)))  # type: ignore[attr-defined]
        ).all()
    )
    now = datetime.now(timezone.utc)
    for key, slot in keys.items():
        if key in taken:
            continue
        session.add(
            ScheduleOccurrence(
                schedule_id=schedule_id,
                occurrence_key=key,
                trigger_source="cron",
                scheduled_for=slot,
                status=OccurrenceStatus.skipped,
                settled_at=now,
            )
        )


def _handle_missed_once(session, schedule: Schedule, next_run: datetime, now: datetime) -> None:
    """A one-off whose time passed runs at the next opportunity, unless it is
    too late for its policy; then it is disabled and recorded as skipped."""
    late = now - next_run
    limit = (
        timedelta(seconds=_LATE_GRACE_SECONDS)
        if schedule.missed_run_policy == MissedRunPolicy.skip
        else _MISSED_RUN_LOOKBACK
    )
    if late <= limit:
        return
    schedule.missed_runs += 1
    schedule.enabled = False
    session.add(schedule)
    _record_skipped(session, schedule.id, [next_run])


def _handle_missed_recurring(session, schedule: Schedule, next_run: datetime, now: datetime) -> None:
    """Move a recurring schedule's slot to the first missed slot its policy
    runs, or past now when it runs none, and record the rest as skipped.

    The due-check then claims the selected slots one at a time, as each
    occurrence settles. Applying the policy again to a schedule it already
    moved changes nothing, so a poll that repeats it is harmless.
    """
    older, recent, future = overdue_slots(
        schedule.cadence,
        next_run,
        schedule.timezone,
        now=now,
        since=now - _MISSED_RUN_LOOKBACK,
    )
    selected = _select_missed(schedule.missed_run_policy, recent, now)
    passed_over = [slot for slot in recent if slot not in selected]
    if not older and not passed_over:
        return
    schedule.missed_runs += older + len(passed_over)
    schedule.next_run_at = selected[0] if selected else future
    session.add(schedule)
    _record_skipped(session, schedule.id, passed_over)


def _handle_missed_runs(session) -> None:
    """Apply each schedule's missed-run policy to the slots it missed.

    A schedule with an occurrence in flight is left alone: its slot is not
    due again until that occurrence settles, and the policy then sees every
    slot that passed meanwhile.
    """
    now = datetime.now(timezone.utc)
    occurrences = ScheduleOccurrenceService(session)
    for schedule in ScheduleService(session).list_schedules():
        if not schedule.enabled:
            continue
        next_run = ensure_utc(schedule.next_run_at)
        if next_run >= now or occurrences.has_active(schedule.id):
            continue
        if schedule.cadence == Cadence.once:
            _handle_missed_once(session, schedule, next_run, now)
        elif schedule.cadence in _RECURRING_CADENCES:
            _handle_missed_recurring(session, schedule, next_run, now)

    session.commit()


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


@dataclass(frozen=True)
class _RunOutcome:
    """How a finished turn is recorded on its run."""

    status: RunStatus | None = None
    error: str | None = None


@dataclass(frozen=True)
class _Attempt:
    """The attempt a run is: which schedule, and which try at its occurrence."""

    run_id: UUID
    schedule_id: UUID
    number: int


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
    _advance_next_run_at(schedule, session)
    occurrence = ScheduleOccurrenceService(session).claim(
        schedule_id,
        occurrence_key=_slot_key(scheduled_for),
        trigger_source="cron",
        scheduled_for=scheduled_for,
        max_attempts=_MAX_CRON_ATTEMPTS,
        lease_seconds=_LEASE_SECONDS,
    )
    return occurrence.id if occurrence else None


def claim_manual_run(session: ScopedSession, schedule_id: UUID) -> UUID | None:
    """Claim a run-now occurrence, or None while the schedule is busy.

    Running a one-off now uses it up: it is disabled in the same commit as the
    claim, so its scheduled slot does not fire a second time. A refused claim
    rolls that back and leaves the one-off as it was. A recurring schedule
    keeps its slot; the freshness guard skips a slot due right after a manual
    success.
    """
    schedule = session.get(Schedule, schedule_id)
    if schedule is not None and schedule.cadence == Cadence.once and schedule.enabled:
        schedule.enabled = False
        session.add(schedule)
    occurrence = ScheduleOccurrenceService(session).claim(
        schedule_id,
        occurrence_key=f"manual:{uuid4()}",
        trigger_source="manual",
        scheduled_for=datetime.now(timezone.utc),
        max_attempts=_MANUAL_MAX_ATTEMPTS,
        lease_seconds=_LEASE_SECONDS,
    )
    return occurrence.id if occurrence else None


def _claim_for_schedule(session: ScopedSession, *, schedule_id: UUID, is_manual: bool) -> UUID | None:
    if is_manual:
        return claim_manual_run(session, schedule_id)
    return _claim_slot(session, ScheduleService(session).get_schedule(schedule_id))


def _start_attempt(session: ScopedSession, *, occurrence_id: UUID, is_manual: bool) -> _Attempt | None:
    run = ScheduleOccurrenceService(session).start_attempt(
        occurrence_id, is_manual=is_manual, lease_seconds=_LEASE_SECONDS
    )
    if run is None:
        return None
    return _Attempt(run_id=run.id, schedule_id=run.schedule_id, number=run.attempt or 1)


def _read_schedule(session: ScopedSession, *, schedule_id: UUID) -> _ScheduledTurn:
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
    return _ScheduledTurn(
        principal=principal,
        title=schedule.title,
        prompt=schedule.prompt,
        model=schedule.model,
        project_id=schedule.project_id,
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


def _settle_occurrence(
    session: ScopedSession, *, occurrence_id: UUID | None, attempt: int, outcome: _RunOutcome, retryable: bool,
) -> None:
    """Stage where the attempt leaves its occurrence: settled, or waiting for
    its next attempt. The caller commits."""
    if occurrence_id is None:
        return
    occurrences = ScheduleOccurrenceService(session)
    if outcome.status == RunStatus.cancelled:
        # The user stopped it; trying again would undo that.
        occurrences.settle(occurrence_id, OccurrenceStatus.cancelled)
    elif outcome.status == RunStatus.failed:
        occurrences.fail_attempt(
            occurrence_id,
            error=outcome.error or "Run failed",
            retry_delay=_retry_delay(attempt) if retryable else None,
        )
    else:
        occurrences.settle(occurrence_id, OccurrenceStatus.success)
    _count_settlement(session, occurrence_id)


def _count_settlement(session: ScopedSession, occurrence_id: UUID) -> None:
    """Stage the schedule's failure streak for an occurrence that just settled,
    and pause the schedule when the streak reaches its limit.

    Only cron occurrences that settle failed extend the streak; a success of
    either kind ends it. A cancelled or skipped occurrence leaves it as is,
    and so does a manual failure, which the user saw happen.
    """
    occurrence = ScheduleOccurrenceService(session).get(occurrence_id)
    if occurrence is None or occurrence.status == OccurrenceStatus.active:
        return
    schedule = session.get(Schedule, occurrence.schedule_id)
    if schedule is None:
        return
    if occurrence.status == OccurrenceStatus.success:
        schedule.consecutive_failures = 0
    elif occurrence.status == OccurrenceStatus.failed and occurrence.trigger_source != "manual":
        schedule.consecutive_failures += 1
        if schedule.consecutive_failures >= _MAX_FAILED_OCCURRENCES and schedule.enabled:
            schedule.enabled = False
            schedule.auto_paused_at = datetime.now(timezone.utc)
            logger.warning(
                f"Schedule {schedule.id} paused after {schedule.consecutive_failures} failed runs in a row"
            )
    else:
        return
    session.add(schedule)


def _record_turn_outcome(
    session: ScopedSession,
    *,
    schedule_id: UUID,
    reason: str | None,
    conversation_id: UUID | None,
    occurrence_id: UUID | None = None,
    attempt: int = 1,
) -> _RunOutcome:
    schedule_service = ScheduleService(session)
    run_service = ScheduleRunService(session)
    # Read again: the schedule may have changed while the turn ran.
    schedule = schedule_service.get_schedule(schedule_id)
    outcome = _RunOutcome()
    if reason == "cancelled":
        outcome = _RunOutcome(status=RunStatus.cancelled)
        logger.info(f"Schedule {schedule_id} run was cancelled")
    elif reason is not None and reason != "completed":
        outcome = _RunOutcome(
            status=RunStatus.failed,
            error="Run did not complete — open the run's task for details.",
        )
        schedule.last_error = outcome.error
        session.add(schedule)
    else:
        _apply_success_write_back(schedule, run_service, conversation_id, session)
    _settle_occurrence(session, occurrence_id=occurrence_id, attempt=attempt, outcome=outcome, retryable=True)

    try:
        session.commit()
    except IntegrityError:
        # Only the success write-back stages a foreign-key write (the
        # last_result pointer), so this is the user's delete landing
        # between still_exists' read and this commit — the delete's
        # release already nulled the column. Redo the write-back without
        # the pointer so the bookkeeping and the settlement survive.
        session.rollback()
        schedule = schedule_service.get_schedule(schedule_id)
        _apply_success_write_back(schedule, run_service, None, session)
        _settle_occurrence(session, occurrence_id=occurrence_id, attempt=attempt, outcome=outcome, retryable=True)
        session.commit()
    return outcome


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


def _record_attempt_error(
    session: ScopedSession, *, schedule_id: UUID, occurrence_id: UUID, attempt: int, error: str, retryable: bool,
) -> None:
    schedule = ScheduleService(session).get_schedule(schedule_id)
    schedule.last_error = error
    session.add(schedule)
    _settle_occurrence(
        session,
        occurrence_id=occurrence_id,
        attempt=attempt,
        outcome=_RunOutcome(status=RunStatus.failed, error=error),
        retryable=retryable,
    )
    session.commit()


def _finish_run(
    session: ScopedSession, *, run_id: UUID, conversation_id: UUID | None, error: str | None, status: RunStatus | None,
) -> None:
    ScheduleRunService(session).finish_run(
        run_id, conversation_id=conversation_id, error=error, status=status
    )


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
    _inflight.add(occurrence_id)
    try:
        await _run_attempt(occurrence_id, is_manual=is_manual, conversation_id=conversation_id)
    finally:
        _inflight.discard(occurrence_id)


async def _run_attempt(
    occurrence_id: UUID,
    *,
    is_manual: bool,
    conversation_id: UUID | None,
) -> None:
    """Run one attempt's turn and record how it went.

    Each read and write is its own database unit, so the run holds no pooled
    connection across the permission check or while its turn runs, and a
    wait for a connection never stalls the event loop. A unit commits when
    it ends, so a failure in one leaves the earlier ones written. The last
    two, the turn's outcome and the run's finish, try again while the pool is
    full (_until_it_lands). An attempt that fails hands its occurrence back
    for a later poll to try again, until its attempts are spent.
    """
    from cowork.handlers.responses import ResponsesHandler
    from cowork.schemas.responses import ResponsesRequest
    from cowork.services.product_permissions import require_product_permission

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
    run_id = attempt.run_id

    error: str | None = None
    final_status: RunStatus | None = None
    try:
        scheduled = await run_db(partial(_read_schedule, schedule_id=schedule_id), scope=SYSTEM_SCOPE)

        await require_product_permission(scope_from_principal(scheduled.principal), "product.execute")

        conversation_id = await run_db(
            partial(_link_conversation, run_id=run_id, scheduled=scheduled, conversation_id=conversation_id),
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
            # overriding every role with the literal string (ENG-2353).
            input=scheduled.prompt,
            model=resolve_schedule_model(scheduled.model),
            stream=True,
            conversation=str(conversation_id),
            trace_tags=["scheduled_task", f"trigger:{trigger}"],
            trace_metadata={
                "schedule_id": str(schedule_id),
                "schedule_run_id": str(run_id),
                "schedule_occurrence_id": str(occurrence_id),
                "trigger_type": trigger,
                "attempt": str(attempt.number),
            },
        )
        async def _drain_run() -> None:
            # The schedule-derived principal is what lets the turn (and the
            # remote backend's per-tenant key mint) run in org mode with no
            # request in flight. The handler opens its own short sessions.
            stream = await ResponsesHandler(principal=scheduled.principal, interactive=False).handle(request)
            async for _ in stream:
                pass

        try:
            await asyncio.wait_for(_drain_run(), timeout=_MAX_RUN_DURATION_SECONDS)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"Run exceeded max duration of {_MAX_RUN_DURATION_SECONDS}s and was aborted."
            ) from exc

        # A cancel or a producer failure closes the stream normally from the
        # consumer's side (the producer runs detached and even swallows its
        # own CancelledError, so handle.task.cancelled() stays False). The
        # buffer's terminal record is the only truthful signal of how the
        # turn ended — without it every run is recorded as success.
        reason = await _turn_terminal_reason(str(conversation_id))
        outcome = await _until_it_lands(
            partial(
                _record_turn_outcome, schedule_id=schedule_id, reason=reason,
                conversation_id=conversation_id, occurrence_id=occurrence_id, attempt=attempt.number,
            ),
            schedule_id=schedule_id,
        )
        final_status, error = outcome.status, outcome.error

    except Exception as exc:
        error = str(exc) or type(exc).__name__
        # The ids travel as record attributes: a database error's sanitized
        # line keeps those and drops the message.
        logger.exception(
            "Schedule run failed: %s", error,
            extra=log_context(schedule_id=schedule_id, conversation_id=conversation_id),
        )
        try:
            await _until_it_lands(
                partial(
                    _record_attempt_error, schedule_id=schedule_id, occurrence_id=occurrence_id,
                    attempt=attempt.number, error=error, retryable=_is_retryable(exc),
                ),
                schedule_id=schedule_id,
            )
        except Exception:
            # The occurrence keeps its lease; a later poll reaps it.
            logger.exception(
                "Failed to record the attempt's error",
                extra=log_context(schedule_id=schedule_id, conversation_id=conversation_id),
            )
    finally:
        try:
            # This write must always land or the run strands at `running`.
            # Its own unit starts clean, whatever failed above.
            await _until_it_lands(
                partial(
                    _finish_run, run_id=run_id, conversation_id=conversation_id,
                    error=error, status=final_status,
                ),
                schedule_id=schedule_id,
            )
        except Exception:
            logger.exception(
                "Failed to finish the run record",
                extra=log_context(schedule_id=schedule_id, conversation_id=conversation_id),
            )


async def _turn_terminal_reason(conversation_id: str) -> str | None:
    """Terminal reason ("completed" | "cancelled" | "error" | …) of the turn
    that just ended on this conversation, or None when unavailable.

    Only call after the turn's stream has been fully drained: the buffer is
    closed then, so tailing from the last record returns immediately."""
    handle = registry.get(conversation_id)
    if handle is None or not handle.buffer.is_closed:
        return None
    try:
        buffer = handle.buffer
        async for rec in buffer.tail(max(buffer.latest_seq - 1, 0)):
            if rec.is_terminal:
                return str(rec.data.get("reason") or "") or None
    except Exception:
        logger.exception(
            f"Could not read terminal state for conversation {conversation_id}"
        )
    return None


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
    for s in ScheduleService(session).list_schedules():
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


_LEASE_EXPIRED_ERROR = "Run stopped responding and was abandoned."


def _reap_expired_leases(session: ScopedSession, now: datetime) -> None:
    """Hand back occurrences whose process died holding them.

    The attempt it was running is recorded failed, and the occurrence gets
    its next attempt as any failed attempt would. One this process is still
    running is left alone.
    """
    occurrences = ScheduleOccurrenceService(session)
    expired = [o for o in occurrences.expired_leases(now) if o.id not in _inflight]
    for occurrence in expired:
        run = occurrences.running_attempt(occurrence.id)
        if run is not None:
            run.status = RunStatus.failed
            run.error = _LEASE_EXPIRED_ERROR
            run.finished_at = now
            session.add(run)
        logger.warning(f"Schedule {occurrence.schedule_id}: occurrence {occurrence.id} lease expired; reaping it")
        occurrences.fail_attempt(
            occurrence.id,
            error=_LEASE_EXPIRED_ERROR,
            retry_delay=_retry_delay(occurrence.attempts) if occurrence.attempts else 0,
        )
        _count_settlement(session, occurrence.id)
    if expired:
        session.commit()


def _poll(session: ScopedSession, capacity: int = _MAX_CONCURRENT_RUNS) -> list[UUID]:
    """One poll's unit: reap dead occurrences, settle missed runs, then claim.

    Returns the occurrences to attempt now — retries that came due first,
    then slots claimed this poll — at most ``capacity`` of them.
    """
    now = datetime.now(timezone.utc)
    _reap_expired_leases(session, now)
    _handle_missed_runs(session)
    ready = [o for o in ScheduleOccurrenceService(session).ready_for_attempt(now) if o not in _inflight]
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
        capacity = _MAX_CONCURRENT_RUNS - len(_inflight)
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
            _inflight.add(occurrence_id)
            asyncio.create_task(run_occurrence(occurrence_id, is_manual=False))


def boot_reap_cutoff(org_mode: bool, now: datetime | None = None) -> datetime | None:
    """Which running runs a starting server may reap: all of them on the
    desktop, where it is the only process, or only those older than a lease
    in org mode, where another replica may still be running the rest. Leases
    release those once they lapse."""
    if not org_mode:
        return None
    return (now or datetime.now(timezone.utc)) - timedelta(seconds=_LEASE_SECONDS)


def start_scheduler() -> None:
    global _scheduler_task
    if _scheduler_task is not None and not _scheduler_task.done():
        return
    _scheduler_task = asyncio.create_task(_scheduler_loop())
    logger.info("Scheduler background task created")
