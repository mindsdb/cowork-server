from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID

from cowork.common.datetime_utils import ensure_utc
from cowork.common.logger import get_logger
from cowork.db.session import get_open_session
from cowork.models.schedule import Schedule
from cowork.schedule_timing import count_missed_occurrences, next_future_occurrence
from cowork.schemas.schedules import Cadence, RunStatus
from cowork.services.schedules import ScheduleRunService, ScheduleService
from cowork.streaming.registry import registry

if TYPE_CHECKING:
    from cowork.principal import Principal

logger = get_logger(__name__)

_POLL_INTERVAL_SECONDS = 30
# Upper bound on a single run. A hung agent (stuck tool call, wedged stream)
# must not keep its ScheduleRun in `running` forever — that would block the
# schedule from ever firing again. On timeout the run is recorded as failed.
_MAX_RUN_DURATION_SECONDS = 600
_scheduler_task: asyncio.Task | None = None

_RECURRING_CADENCES = {Cadence.hourly, Cadence.daily, Cadence.weekly, Cadence.weekdays}

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
            # A one-off that was never executed — disable it without running
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

    A scheduled run has no HTTP request and so no gateway-injected principal, so
    the owning identity comes from the row itself: ``org_id`` scopes the turn's
    data and the per-tenant key the remote backend mints, and ``created_by``
    attributes the rows it writes.

    Local mode has no tenant context, so return None (unscoped). Org mode
    requires both ids; a NULL is corrupt data that would write rows the owner
    can't see, so fail loud instead.
    """
    from cowork.common.settings.app_settings import get_app_settings
    from cowork.db.scoped import MissingTenantScopeError
    from cowork.principal import Principal

    if get_app_settings().tenancy_mode != "org":
        return None
    if not schedule.org_id or not schedule.created_by:
        raise MissingTenantScopeError(
            f"schedule {schedule.id} is missing org_id/created_by; "
            "cannot resolve a service principal to run it in org mode"
        )
    return Principal(user_id=schedule.created_by, org_id=schedule.org_id)


async def execute_schedule(
    schedule_id: UUID,
    is_manual: bool = False,
    conversation_id: UUID | None = None,
) -> None:
    from cowork.handlers.responses import ResponsesHandler
    from cowork.schemas.responses import ResponsesRequest

    from cowork.db.scoped import MissingTenantScopeError, ScopedSession, SYSTEM_SCOPE
    session = ScopedSession(get_open_session(), SYSTEM_SCOPE)
    run_service = ScheduleRunService(session)
    schedule_service = ScheduleService(session)

    run = run_service.create_run(schedule_id, is_manual=is_manual)

    error: str | None = None
    final_status: RunStatus | None = None
    try:
        schedule = schedule_service.get_schedule(schedule_id)

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

        if conversation_id is None:
            # Conversation not pre-created by the caller (e.g. cron tick). Create
            # it under the schedule's OWN scope so org mode stamps the owning
            # org_id: the scheduler's SYSTEM_SCOPE is deliberately unscoped (it
            # scans every org), so creating through `session` would write an
            # invisible NULL-org row.
            from cowork.db.scoped import (
                ScopedSession,
                scope_from_principal,
                unsafe_unscoped_session,
            )
            from cowork.services.conversations import ConversationService

            conv_session = ScopedSession(
                unsafe_unscoped_session(session), scope_from_principal(principal)
            )
            conversation = ConversationService(conv_session).create_conversation(
                topic=schedule.title,
                project_id=schedule.project_id,
            )
            conversation_id = conversation.id

        run_service.set_run_conversation(run.id, conversation_id)

        # Stamp the run's identity on the Langfuse trace (existing pass-through
        # seam) so incident forensics don't have to reconstruct which schedule/
        # trigger produced a turn from timestamps.
        trigger = "manual" if is_manual else "cron"
        request = ResponsesRequest(
            input=schedule.prompt,
            model=schedule.model,
            stream=True,
            conversation=str(conversation_id),
            trace_tags=["scheduled_task", f"trigger:{trigger}"],
            trace_metadata={
                "schedule_id": str(schedule_id),
                "schedule_run_id": str(run.id),
                "trigger_type": trigger,
            },
        )
        async def _drain_run() -> None:
            # ResponsesHandler takes a RAW session (it wraps its own scope from
            # the principal); hand it the underlying session, not our scoped one.
            # The schedule-derived principal is what lets the turn (and the
            # remote backend's per-tenant key mint) run in org mode with no
            # request in flight.
            from cowork.db.scoped import unsafe_unscoped_session
            stream = await ResponsesHandler(
                unsafe_unscoped_session(session), principal=principal
            ).handle(request)
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

        # Refresh schedule in case it changed during execution
        schedule = schedule_service.get_schedule(schedule_id)
        if reason == "cancelled":
            final_status = RunStatus.cancelled
            logger.info(f"Schedule {schedule_id} run was cancelled")
        elif reason is not None and reason != "completed":
            final_status = RunStatus.failed
            error = "Run did not complete — open the run's task for details."
            schedule.last_error = error
            session.add(schedule)
        else:
            schedule.last_run_at = datetime.now(timezone.utc)
            schedule.last_result_conversation_id = conversation_id
            schedule.last_error = None
            schedule.missed_runs = 0
            session.add(schedule)

        # Always consume the cron slot: the schedule stays due otherwise and
        # the loop would immediately restart the run the user killed (a
        # cancelled/failed run isn't a success, so the freshness guard
        # wouldn't block the restart).
        if not is_manual:
            _advance_next_run_at(schedule, session)

        session.commit()

    except Exception as exc:
        error = str(exc)
        logger.exception(f"Schedule {schedule_id} run failed: {error}")
        try:
            schedule = schedule_service.get_schedule(schedule_id)
            schedule.last_error = error
            session.add(schedule)
            session.commit()
        except Exception:
            pass
    finally:
        try:
            run_service.finish_run(
                run.id, conversation_id=conversation_id, error=error, status=final_status
            )
        except Exception:
            logger.exception(f"Failed to finish run record for schedule {schedule_id}")
        session.close()


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


async def _scheduler_loop() -> None:
    logger.info("Scheduler loop started")
    while True:
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        from cowork.db.scoped import ScopedSession, SYSTEM_SCOPE
        session = ScopedSession(get_open_session(), SYSTEM_SCOPE)
        try:
            _handle_missed_runs(session)
            due = _due_schedules(session, datetime.now(timezone.utc))
        except Exception:
            logger.exception("Scheduler loop error during poll")
            due = []
        finally:
            session.close()

        for schedule in due:
            asyncio.create_task(execute_schedule(schedule.id, is_manual=False))


def start_scheduler() -> None:
    global _scheduler_task
    if _scheduler_task is not None and not _scheduler_task.done():
        return
    _scheduler_task = asyncio.create_task(_scheduler_loop())
    logger.info("Scheduler background task created")
