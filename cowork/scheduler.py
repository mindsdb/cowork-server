from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import TYPE_CHECKING, TypeVar
from uuid import UUID

from sqlalchemy.exc import IntegrityError, InvalidRequestError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from cowork.common.datetime_utils import ensure_utc
from cowork.common.logger import get_logger
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
from cowork.services.schedules import ScheduleRunService, ScheduleService
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
_scheduler_task: asyncio.Task | None = None

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


def _create_run(session: ScopedSession, *, schedule_id: UUID, is_manual: bool) -> UUID:
    return ScheduleRunService(session).create_run(schedule_id, is_manual=is_manual).id


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


def _record_schedule_error(session: ScopedSession, *, schedule_id: UUID, error: str) -> None:
    schedule = ScheduleService(session).get_schedule(schedule_id)
    schedule.last_error = error
    session.add(schedule)


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
    """Run one schedule's turn and record how it went.

    Each read and write is its own database unit, so the run holds no pooled
    connection across the permission check or while its turn runs, and a
    wait for a connection never stalls the event loop. A unit commits when
    it ends, so a failure in one leaves the earlier ones written. The last
    two, the turn's outcome and the run's finish, try again while the pool is
    full (_until_it_lands).
    """
    from cowork.handlers.responses import ResponsesHandler
    from cowork.schemas.responses import ResponsesRequest
    from cowork.services.product_permissions import require_product_permission

    try:
        run_id = await run_db(
            partial(_create_run, schedule_id=schedule_id, is_manual=is_manual), scope=SYSTEM_SCOPE,
        )
    except PoolTimeoutError:
        # Nothing was written, so nothing is left at `running`: a cron slot
        # stays due and the next poll starts it.
        logger.warning(
            f"Schedule {schedule_id}: no database connection freed to start its run; it did not run"
        )
        return

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
                is_manual=is_manual, conversation_id=conversation_id,
            ),
            schedule_id=schedule_id,
        )
        final_status, error = outcome.status, outcome.error

    except Exception as exc:
        error = str(exc)
        logger.exception(f"Schedule {schedule_id} run failed: {error}")
        try:
            await run_db(
                partial(_record_schedule_error, schedule_id=schedule_id, error=error), scope=SYSTEM_SCOPE,
            )
        except Exception:
            pass
    finally:
        try:
            # This write must always land or the run strands at `running` and
            # wedges the schedule. Its own unit starts clean, whatever failed
            # above.
            await _until_it_lands(
                partial(
                    _finish_run, run_id=run_id, conversation_id=conversation_id,
                    error=error, status=final_status,
                ),
                schedule_id=schedule_id,
            )
        except Exception:
            logger.exception(f"Failed to finish run record for schedule {schedule_id}")


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


def _poll(session: ScopedSession) -> list[UUID]:
    """One poll's unit: settle missed runs, then the ids of the schedules due now."""
    _handle_missed_runs(session)
    return [schedule.id for schedule in _due_schedules(session, datetime.now(timezone.utc))]


async def _scheduler_loop() -> None:
    logger.info("Scheduler loop started")
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

        for schedule_id in due:
            asyncio.create_task(execute_schedule(schedule_id, is_manual=False))


def start_scheduler() -> None:
    global _scheduler_task
    if _scheduler_task is not None and not _scheduler_task.done():
        return
    _scheduler_task = asyncio.create_task(_scheduler_loop())
    logger.info("Scheduler background task created")
