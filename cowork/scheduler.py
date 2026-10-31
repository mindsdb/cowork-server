from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import TYPE_CHECKING, TypeVar
from uuid import UUID

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
from cowork.models.schedule import Schedule
from cowork.schedule_timing import count_missed_occurrences, next_future_occurrence
from cowork.schemas.schedules import Cadence, RunStatus, resolve_schedule_model
from cowork.services.schedules import ScheduleRunService, ScheduleService, SlotAlreadyClaimed
from cowork.streaming.registry import ServerShuttingDown, registry

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
# How old a run still at `running` must be before a booting replica may reap
# it: its turn is cut off at _MAX_RUN_DURATION_SECONDS and its last writes
# (the outcome, then the finish) each give up after
# _BOOKKEEPING_DEADLINE_SECONDS, plus slack for the units before the turn
# starts. An error path with a third last write can use up that slack, which
# is harmless: the run's own finish overwrites a reaped status. Age is only a
# heuristic, since a database unit that stalls holds a run past any deadline,
# so only boot uses it, not every poll.
_ORPHANED_RUN_AGE_SECONDS = _MAX_RUN_DURATION_SECONDS + 2 * _BOOKKEEPING_DEADLINE_SECONDS + 10 * 60
# A run holds its claim this long unless its process renews it, which it does
# every _LEASE_RENEW_SECONDS for as long as the run lives. A run past its lease
# belongs to a process that died without recording how the run ended, so any
# replica's poll reaps it (_reap_lapsed_runs) within a lease and a poll.
_RUN_LEASE_SECONDS = 120
_LEASE_RENEW_SECONDS = 30
_LAPSED_RUN_ERROR = "Run was interrupted: the server running it stopped before it finished."
_scheduler_task: asyncio.Task | None = None
# A run's error when its turn was refused because the server was shutting
# down. The run never started, so the schedule's own last_error is left alone.
_SHUTDOWN_RUN_ERROR = "Run did not start: the server was shutting down."
# A run's error when a shutdown cancelled it before it recorded its turn's
# outcome. Saved as failed, not success, so a cron slot it did not consume
# stays due, and the freshness guard skips no slot because of it.
_INTERRUPTED_RUN_ERROR = "Run was interrupted: the server shut down before it finished."
# The runs the poll loop started and nothing awaits. A shutdown waits for them
# (wait_for_scheduled_runs) so each run's last writes land before the process
# exits. A manual run belongs to its request instead. Uvicorn's request drain
# waits for that request only up to --timeout-graceful-shutdown, then cancels
# it, and the run is saved as interrupted (_INTERRUPTED_RUN_ERROR).
_scheduled_runs: set[asyncio.Task] = set()

_RECURRING_CADENCES = {Cadence.hourly, Cadence.daily, Cadence.weekly, Cadence.weekdays}

# A one-off whose slot has just passed is still due this poll and must be run,
# not disabled — the poll runs `_handle_missed_runs` before `_due_schedules`,
# so disabling on any overdue amount kills the task before it can fire (the
# task then shows as "Paused" and never runs; ENG-1675). Only a one-off overdue
# by more than this catch-up window — the app was offline when its slot passed —
# is disabled without running, so a long-stale task doesn't fire unexpectedly on
# the next launch. Mirrors the recurring branch, which lets a single overdue slot
# (missed == 1) run rather than fast-forwarding past it.
#
# Independent of `_FRESHNESS_WINDOW_SECONDS[Cadence.once]` below despite the shared
# 1h value: this bounds how late a one-off may still run; the freshness window
# suppresses a slot after a recent successful run. Tune them separately.
_ONCE_CATCHUP_WINDOW_SECONDS = 60 * 60

# Freshness guard (ENG-688): if a successful run — typically a manual
# "run now" — finished this recently before a due cron slot, the slot is
# skipped instead of executed, so both runs don't publish the same output
# twice. Hourly gets a tighter window so consecutive slots never suppress
# each other even when a run finishes mid-hour.
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


def _handle_missed_runs(session) -> None:
    now = datetime.now(timezone.utc)
    schedules = ScheduleService(session).list_schedules()
    for schedule in schedules:
        if not schedule.enabled:
            continue

        next_run = ensure_utc(schedule.next_run_at)

        if next_run >= now:
            continue

        if schedule.cadence == Cadence.once:
            # A due one-off is left enabled so `_due_schedules` executes it on
            # this same tick. Only disable it without running when it is overdue
            # beyond the catch-up window — the app was offline when its slot
            # passed — so a long-stale one-off doesn't fire on the next launch.
            # Bump missed_runs like the recurring branch so an auto-disabled
            # one-off carries a signal it was skipped rather than run.
            if (now - next_run).total_seconds() > _ONCE_CATCHUP_WINDOW_SECONDS:
                schedule.missed_runs += 1
                schedule.enabled = False
                session.add(schedule)
            continue

        if schedule.cadence not in _RECURRING_CADENCES:
            continue

        missed, future = count_missed_occurrences(
            schedule.cadence,
            next_run,
            schedule.timezone,
            now=now,
        )
        # Only fast-forward when more than one occurrence was skipped (app
        # offline for multiple cadence periods). A single overdue slot
        # (missed == 1) is still due this poll — advancing here would skip
        # the run entirely. 
        if missed > 1:
            schedule.missed_runs += missed
            schedule.next_run_at = future
            session.add(schedule)

    session.commit()


def _principal_for_schedule(schedule: Schedule) -> Principal | None:
    """Service principal for a scheduled run, derived from the schedule row.

    Delegates to `service_principal_for` from the scoped module, which decides
    whether schedules are partitioned (org mode, or personal schedules on a
    trusted-identity deployment). The custom error message below wraps it with
    schedule-specific context.
    """
    from cowork.db.scoped import service_principal_for

    try:
        return service_principal_for(schedule.org_id, schedule.created_by)
    except MissingTenantScopeError:
        raise MissingTenantScopeError(
            f"schedule {schedule.id} is missing org_id/created_by; "
            "cannot resolve a service principal to run it as its creator"
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
class DueSlot:
    """A schedule due to run, and the cron slot its run claims."""

    schedule_id: UUID
    slot_at: datetime


class RunClaimLost(Exception):
    """Another replica reaped this run before its turn started, so the slot
    may run there; this run must not start its turn."""


@dataclass(frozen=True)
class _RunOutcome:
    """How a finished turn is recorded on its run."""

    status: RunStatus | None = None
    error: str | None = None


def _create_run(
    session: ScopedSession, *, schedule_id: UUID, is_manual: bool, slot_at: datetime | None,
) -> UUID:
    return ScheduleRunService(session).create_run(
        schedule_id, is_manual=is_manual, slot_at=slot_at, lease_seconds=_RUN_LEASE_SECONDS,
    ).id


def _renew_run_lease(session: ScopedSession, *, run_id: UUID) -> bool:
    return ScheduleRunService(session).renew_lease(run_id, _RUN_LEASE_SECONDS)


async def _renew_lease(run_id: UUID, *, schedule_id: UUID) -> None:
    """Renew the run's lease until cancelled, so no replica reaps it while it
    lives. A renewal the pool refuses is tried again at the next interval. A
    run that is no longer `running` was reaped by another replica, so renewing
    stops there."""
    while True:
        await asyncio.sleep(_LEASE_RENEW_SECONDS)
        try:
            renewed = await run_db(partial(_renew_run_lease, run_id=run_id), scope=SYSTEM_SCOPE)
        except Exception:
            logger.warning(
                "Could not renew a scheduled run's lease; trying again",
                exc_info=True,
                extra=log_context(schedule_id=schedule_id),
            )
            continue
        if not renewed:
            logger.warning(
                "A scheduled run lost its lease to another replica",
                extra=log_context(schedule_id=schedule_id),
            )
            return


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
    recorded on the run before the turn starts. Refuses with RunClaimLost when
    another replica has reaped the run, since that replica may run its slot.

    The check before the conversation exists saves making one for a run
    already reaped. The link itself is the claim check that counts: it
    records the conversation only on a run still `running`
    (ScheduleRunService.set_run_conversation), so a reap that lands in
    between still stops the turn."""
    if not ScheduleRunService(session).still_running(run_id):
        raise RunClaimLost(f"run {run_id} was reaped before its turn started")
    created = conversation_id is None
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
    if not ScheduleRunService(session).set_run_conversation(run_id, conversation_id):
        if created:
            _delete_unstarted_conversation(session, scheduled=scheduled, conversation_id=conversation_id)
        raise RunClaimLost(f"run {run_id} was reaped before its turn started")
    return conversation_id


def _delete_unstarted_conversation(
    session: ScopedSession, *, scheduled: _ScheduledTurn, conversation_id: UUID,
) -> None:
    """Delete the conversation _link_conversation made for a run whose turn
    never started, so it is not left empty in its owner's list. Under the
    schedule's own scope, as it was created."""
    from cowork.services.conversations import ConversationService

    conv_session = ScopedSession(
        unsafe_unscoped_session(session), scope_from_principal(scheduled.principal)
    )
    ConversationService(conv_session).delete_conversation(conversation_id)


def _record_turn_outcome(
    session: ScopedSession, *, schedule_id: UUID, reason: str | None, is_manual: bool, conversation_id: UUID,
) -> _RunOutcome:
    schedule_service = ScheduleService(session)
    run_service = ScheduleRunService(session)
    # Read again: the schedule may have changed while the turn ran.
    schedule = schedule_service.get_schedule(schedule_id)
    outcome = _RunOutcome()
    if reason == "cancelled":
        outcome = _RunOutcome(status=RunStatus.cancelled)
        logger.info("Schedule run cancelled", extra={"event": "schedule_cancelled", "schedule_id": schedule_id})
    elif reason is not None and reason != "completed":
        outcome = _RunOutcome(
            status=RunStatus.failed,
            error="Run did not complete — open the run's task for details.",
        )
        schedule.last_error = outcome.error
        session.add(schedule)
    else:
        _apply_success_write_back(schedule, run_service, conversation_id, session)

    # Always consume the cron slot: the schedule stays due otherwise and
    # the loop would immediately restart the run the user killed (a
    # cancelled/failed run isn't a success, so the freshness guard
    # wouldn't block the restart).
    if not is_manual:
        _advance_next_run_at(schedule, session)

    try:
        session.commit()
    except IntegrityError:
        # Only the success write-back stages a foreign-key write (the
        # last_result pointer), so this is the user's delete landing
        # between still_exists' read and this commit — the delete's
        # release already nulled the column. Redo the write-back without
        # the pointer so the bookkeeping and the consumed slot survive.
        session.rollback()
        schedule = schedule_service.get_schedule(schedule_id)
        _apply_success_write_back(schedule, run_service, None, session)
        if not is_manual:
            _advance_next_run_at(schedule, session)
        session.commit()
    return outcome


def _record_schedule_error(session: ScopedSession, *, schedule_id: UUID, error: str, consume_slot: bool) -> bool:
    """Record a run's error on its schedule. Returns whether it consumed the
    cron slot."""
    schedule = ScheduleService(session).get_schedule(schedule_id)
    schedule.last_error = error
    session.add(schedule)
    # A run that disabled its schedule has already stopped the re-fire. So has
    # a pause during the run; the run then gives its slot back
    # (execute_schedule), so a resumed schedule can claim the slot it left due.
    consumed = consume_slot and schedule.enabled
    if consumed:
        _advance_next_run_at(schedule, session)
    return consumed


def _finish_run(
    session: ScopedSession, *, run_id: UUID, conversation_id: UUID | None, error: str | None,
    status: RunStatus | None, release_slot: bool = False,
) -> None:
    ScheduleRunService(session).finish_run(
        run_id, conversation_id=conversation_id, error=error, status=status, release_slot=release_slot,
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
                "No database connection freed to record scheduled run; trying again",
                extra=log_context(schedule_id=schedule_id)
            )
            await asyncio.sleep(_BOOKKEEPING_RETRY_PAUSE_SECONDS)


async def execute_schedule(
    schedule_id: UUID,
    is_manual: bool = False,
    conversation_id: UUID | None = None,
    *,
    slot_at: datetime | None = None,
) -> None:
    """Run one schedule's turn and record how it went.

    Each read and write is its own database unit, so the run holds no pooled
    connection across the permission check or while its turn runs, and a
    wait for a connection never stalls the event loop. A unit commits when
    it ends, so a failure in one leaves the earlier ones written. The last
    two, the turn's outcome and the run's finish, try again while the pool is
    full (_until_it_lands).

    A cron run names its slot (``slot_at``), and inserting its run row claims
    that slot: when another replica's poll already started it, this run
    returns without doing anything. A run that leaves its slot due, because
    its turn never started or a shutdown cut it, gives the slot back, so the
    next poll can claim it again.
    """
    from cowork.handlers.responses import ResponsesHandler
    from cowork.schemas.responses import ResponsesRequest
    from cowork.services.product_permissions import ProductPermissionUnavailable, require_product_permission

    try:
        run_id = await run_db(
            partial(_create_run, schedule_id=schedule_id, is_manual=is_manual, slot_at=slot_at),
            scope=SYSTEM_SCOPE,
        )
    except PoolTimeoutError:
        # Nothing was written, so nothing is left at `running`: a cron slot
        # stays due and the next poll starts it.
        logger.warning(
            "No database connection freed to start scheduled run",
            extra=log_context(schedule_id=schedule_id)
        )
        return
    except SlotAlreadyClaimed:
        logger.info(
            "Schedule slot already started by another replica",
            extra=log_context(schedule_id=schedule_id),
        )
        return

    lease = asyncio.create_task(_renew_lease(run_id, schedule_id=schedule_id))
    slot_consumed = False
    error: str | None = None
    final_status: RunStatus | None = None
    created_conversation = conversation_id is None
    turn_started = False
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
                "trigger_type": trigger,
            },
        )
        async def _drain_run() -> None:
            # The schedule-derived principal is what lets the turn (and the
            # remote backend's per-tenant key mint) run in org mode with no
            # request in flight. The handler opens its own short sessions.
            stream = await ResponsesHandler(principal=scheduled.principal, interactive=False).handle(request)
            async for _ in stream:
                pass

        turn_started = True
        try:
            await asyncio.wait_for(_drain_run(), timeout=_MAX_RUN_DURATION_SECONDS)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"Run exceeded max duration of {_MAX_RUN_DURATION_SECONDS}s and was aborted."
            ) from exc
        except ServerShuttingDown:
            # This replica is ending its turns for a shutdown and refused the
            # run's turn before it started. Nothing ran, so the schedule keeps
            # its last_error, and _record_turn_outcome never consumes the cron
            # slot: it stays due for the next poll, on a replica still
            # serving. The finally below finishes the run row.
            error = _SHUTDOWN_RUN_ERROR
            logger.info(
                "Schedule run did not start; the server is shutting down",
                extra=log_context(schedule_id=schedule_id, conversation_id=conversation_id),
            )
            if created_conversation:
                try:
                    await run_db(
                        partial(
                            _delete_unstarted_conversation,
                            scheduled=scheduled, conversation_id=conversation_id,
                        ),
                        scope=SYSTEM_SCOPE,
                    )
                    conversation_id = None
                except Exception:
                    logger.warning(
                        "Could not delete the empty conversation of a run that did not start",
                        exc_info=True,
                        extra=log_context(schedule_id=schedule_id, conversation_id=conversation_id),
                    )
            return

        # A cancel or a producer failure closes the stream normally from the
        # consumer's side (the producer runs detached and even swallows its
        # own CancelledError, so handle.task.cancelled() stays False). The
        # buffer's terminal record is the only truthful signal of how the
        # turn ended — without it every run is recorded as success.
        reason = await _turn_terminal_reason(str(conversation_id))
        outcome = await _until_it_lands(
            partial(
                _record_turn_outcome, schedule_id=schedule_id, reason=reason,
                is_manual=is_manual, conversation_id=conversation_id,
            ),
            schedule_id=schedule_id,
        )
        final_status, error = outcome.status, outcome.error
        slot_consumed = not is_manual

    except RunClaimLost:
        # The lease ran out before the turn started, and another replica
        # reaped the run and gave its slot back. The slot belongs to whichever
        # poll claims it next, so this run records nothing on the schedule.
        error = _LAPSED_RUN_ERROR
        logger.warning(
            "Scheduled run lost its claim before its turn started",
            extra=log_context(schedule_id=schedule_id),
        )
    except asyncio.CancelledError:
        # A shutdown cut the run: Uvicorn's request drain cancels a manual
        # run's request at its bound, and the event loop's close cancels a
        # cron run still going. The finally saves it as failed.
        error = _INTERRUPTED_RUN_ERROR
        raise
    except Exception as exc:
        error = str(exc)
        # The ids travel as record attributes: a database error's sanitized
        # line keeps those and drops the message.
        logger.exception(
            "Schedule run failed: %s", error,
            extra=log_context(schedule_id=schedule_id, conversation_id=conversation_id),
        )
        # A run that raised consumes its cron slot like one that failed inside
        # its turn. Left due, the next poll starts it again, and a failure that
        # repeats (a timeout, a revoked permission) re-fires every poll. A busy
        # pool or an auth outage before the turn starts is the exception: it
        # clears on its own, the agent has done nothing yet, and skipping the
        # slot would drop a daily or weekly occurrence over a short blip.
        transient = not turn_started and isinstance(exc, (PoolTimeoutError, ProductPermissionUnavailable))
        try:
            slot_consumed = await _until_it_lands(
                partial(
                    _record_schedule_error, schedule_id=schedule_id, error=error,
                    consume_slot=not is_manual and not transient,
                ),
                schedule_id=schedule_id,
            )
        except Exception:
            logger.exception(
                "Failed to record the scheduled run's error",
                extra=log_context(schedule_id=schedule_id, conversation_id=conversation_id),
            )
    finally:
        lease.cancel()
        await asyncio.gather(lease, return_exceptions=True)
        try:
            # This write must always land or the run strands at `running` and
            # wedges the schedule. Its own unit starts clean, whatever failed
            # above.
            await _until_it_lands(
                partial(
                    _finish_run, run_id=run_id, conversation_id=conversation_id,
                    error=error, status=final_status,
                    release_slot=slot_at is not None and not slot_consumed,
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
            "Could not read conversation terminal state",
            extra=log_context(conversation_id=conversation_id)
        )
    return None


def _ran_recently(schedule: Schedule, run_service: ScheduleRunService, now: datetime) -> bool:
    window = _FRESHNESS_WINDOW_SECONDS.get(schedule.cadence)
    if not window:
        return False
    last = run_service.last_successful_finish(schedule.id)
    return last is not None and (now - last).total_seconds() < window


def _due_schedules(session, now: datetime) -> list[Schedule]:
    """Enabled schedules whose slot is due and should actually execute.

    A due slot with a successful run inside the freshness window is skipped
    and advanced to its next occurrence instead of returned.
    """
    run_service = ScheduleRunService(session)
    due: list[Schedule] = []
    skipped = False
    for s in ScheduleService(session).list_schedules():
        # Gate on ANY in-flight run, manual included: a manual run still
        # executing when the slot comes due would otherwise run alongside the
        # cron run and publish the same output twice. The slot is deferred,
        # not consumed — once the run finishes, the freshness guard decides
        # whether it still fires.
        if not s.enabled or ensure_utc(s.next_run_at) > now or run_service.has_active_run(s.id):
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
                "Skipping schedule slot after a recent successful run",
                extra=log_context(schedule_id=s.id)
            )
            _advance_next_run_at(s, session)
            skipped = True
            continue
        due.append(s)
    if skipped:
        session.commit()
    return due


def boot_reap_cutoff(now: datetime | None = None) -> datetime | None:
    """Which running runs a starting server may reap: all of them on the
    desktop, where it is the only process, or in org mode, where other
    replicas may still be running them, only runs old enough to be presumed
    dead."""
    from cowork.common.settings.app_settings import get_app_settings

    if get_app_settings().tenancy_mode != "org":
        return None
    return (now or datetime.now(timezone.utc)) - timedelta(seconds=_ORPHANED_RUN_AGE_SECONDS)


def _reap_lapsed_runs(session: ScopedSession, now: datetime) -> None:
    """Fail the runs whose process stopped renewing their lease.

    A run that never started its turn gave its slot back, so the slot fires
    again (ScheduleRunService.reap_lapsed_runs). A run whose turn may have
    started consumes its slot here instead, so the work isn't done twice. A
    paused schedule moves on too, so it resumes at its next occurrence rather
    than on a slot this run still holds.
    """
    for run in ScheduleRunService(session).reap_lapsed_runs(now=now, error=_LAPSED_RUN_ERROR):
        logger.warning(
            "Reaped a scheduled run whose server stopped renewing it",
            extra=log_context(schedule_id=run.schedule_id, conversation_id=run.conversation_id),
        )
        schedule = session.get(Schedule, run.schedule_id)
        if (
            schedule is not None
            and not run.is_manual
            and ensure_utc(schedule.next_run_at) == ensure_utc(run.slot_at)
        ):
            _advance_next_run_at(schedule, session)
    session.commit()


def _poll(session: ScopedSession) -> list[DueSlot]:
    """One poll's unit: reap runs whose server died, settle missed runs, then
    the schedules due now with the slot each one's run claims."""
    _reap_lapsed_runs(session, datetime.now(timezone.utc))
    _handle_missed_runs(session)
    return [
        DueSlot(schedule_id=schedule.id, slot_at=ensure_utc(schedule.next_run_at))
        for schedule in _due_schedules(session, datetime.now(timezone.utc))
    ]


async def _scheduler_loop() -> None:
    logger.info("Scheduler loop started")
    # Replicas a rollout starts together would otherwise poll in step, and two
    # replicas polling together can both start one due schedule.
    await asyncio.sleep(random.uniform(0, _POLL_INTERVAL_SECONDS))
    while True:
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        try:
            # A unit in a worker thread: when the pool has no connection to
            # give, the poll is refused after POOL_TIMEOUT instead of stalling
            # the event loop, and the next poll tries again.
            due = await run_db(_poll, scope=SYSTEM_SCOPE)
        except PoolTimeoutError:
            logger.warning("Scheduler poll found no free database connection; the next poll retries")
            due = []
        except Exception:
            logger.exception("Scheduler loop error during poll")
            due = []

        for slot in due:
            run = asyncio.create_task(execute_schedule(slot.schedule_id, is_manual=False, slot_at=slot.slot_at))
            _scheduled_runs.add(run)
            run.add_done_callback(_scheduled_runs.discard)


def start_scheduler() -> None:
    global _scheduler_task
    if _scheduler_task is not None and not _scheduler_task.done():
        return
    _scheduler_task = asyncio.create_task(_scheduler_loop())
    logger.info("Scheduler background task created")


def stop_scheduler() -> asyncio.Task | None:
    """Cancel the poll loop, so it starts no more runs, without waiting for it.

    A poll reading the database finishes that read first: run_db re-raises
    the cancel once the read ends, so the loop never reaches the line that
    starts a run. Returns the cancelled loop for the caller to wait on once
    it has done what must not wait behind that read, or None when no loop
    was running, as on a second call.
    """
    global _scheduler_task
    task, _scheduler_task = _scheduler_task, None
    if task is None or task.done():
        return None
    task.cancel()
    logger.info("Scheduler stopped")
    return task


async def wait_for_scheduled_runs(*, timeout_seconds: float) -> int:
    """Wait up to ``timeout_seconds`` for the runs the poll loop started.

    For a shutdown, after the server has ended their turns: each run then
    records the turn's outcome and finishes its run row. A row left at
    `running` stops its schedule firing until a boot reaps it
    (ScheduleRunService.reap_orphaned_runs). Returns how many runs were still
    going when the wait ended; they go on until the process exits.
    """
    runs = [run for run in _scheduled_runs if not run.done()]
    if not runs:
        return 0
    _, pending = await asyncio.wait(runs, timeout=timeout_seconds)
    return len(pending)
