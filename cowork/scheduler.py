from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from uuid import UUID

from cowork.common.logger import get_logger
from cowork.db.session import get_open_session
from cowork.models.schedule import Schedule
from cowork.schemas.schedules import Cadence
from cowork.services.schedules import ScheduleRunService, ScheduleService

logger = get_logger(__name__)

_POLL_INTERVAL_SECONDS = 30
_scheduler_task: asyncio.Task | None = None

# Hourly is a fixed wall-clock-agnostic interval — it fires every 60
# minutes regardless of DST. Daily/weekly are *calendar* cadences: they
# preserve the local wall-clock time (e.g. "9:00 AM") across DST shifts,
# so they're computed in the schedule's stored zone instead (see
# ``_next_calendar_run``).
_HOURLY_DELTA = timedelta(hours=1)
_CALENDAR_DELTAS: dict[str, timedelta] = {
    Cadence.daily: timedelta(days=1),
    Cadence.weekly: timedelta(weeks=1),
}


def _as_utc(dt: datetime) -> datetime:
    """Treat a naive datetime as UTC (legacy rows stored without tzinfo)."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _zone(name: str) -> ZoneInfo:
    """Resolve a stored IANA name, falling back to UTC if it's unknown."""
    try:
        return ZoneInfo(name or "UTC")
    except (ZoneInfoNotFoundError, ValueError):
        logger.warning("Unknown schedule timezone %r — falling back to UTC", name)
        return ZoneInfo("UTC")


def _next_calendar_run(next_run_utc: datetime, cadence: str, tz_name: str, after: datetime) -> datetime:
    """Advance a daily/weekly run to the first occurrence strictly after ``after``.

    The arithmetic happens on the *naive* wall-clock in the schedule's
    zone so the local time-of-day is preserved across DST boundaries —
    a 9:00 AM task stays 9:00 AM whether the offset is PST or PDT. The
    result is converted back to a UTC instant for storage. A pure
    ``next_run_utc + timedelta(days=1)`` would drift by an hour each time
    the offset changes; this does not.
    """
    delta = _CALENDAR_DELTAS[cadence]
    tz = _zone(tz_name)
    # Drop to wall-clock in-zone, step by whole calendar units, re-localize.
    local_naive = next_run_utc.astimezone(tz).replace(tzinfo=None)
    while True:
        local_naive += delta
        candidate = local_naive.replace(tzinfo=tz).astimezone(timezone.utc)
        if candidate > after:
            return candidate


def _compute_next_run(next_run_utc: datetime, cadence: str, tz_name: str, after: datetime) -> datetime:
    """Next future run instant (UTC) for a recurring cadence after ``after``."""
    if cadence == Cadence.hourly:
        candidate = next_run_utc
        while candidate <= after:
            candidate += _HOURLY_DELTA
        return candidate
    return _next_calendar_run(next_run_utc, cadence, tz_name, after)


def _advance_next_run_at(schedule: Schedule, session) -> None:
    if schedule.cadence == Cadence.once:
        schedule.enabled = False
        session.add(schedule)
        return

    if schedule.cadence not in _CALENDAR_DELTAS and schedule.cadence != Cadence.hourly:
        return

    now = datetime.now(timezone.utc)
    schedule.next_run_at = _compute_next_run(
        _as_utc(schedule.next_run_at), schedule.cadence, schedule.timezone, now
    )
    session.add(schedule)


def _handle_missed_runs(session) -> None:
    now = datetime.now(timezone.utc)
    schedules = ScheduleService(session).list_schedules()
    for schedule in schedules:
        if not schedule.enabled:
            continue

        next_run = _as_utc(schedule.next_run_at)
        if next_run >= now:
            continue

        if schedule.cadence == Cadence.once:
            # A one-off that was never executed — disable it without running
            schedule.enabled = False
            session.add(schedule)
            continue

        if schedule.cadence not in _CALENDAR_DELTAS and schedule.cadence != Cadence.hourly:
            continue

        # Count how many scheduled occurrences slipped while the app was
        # off by walking the cadence forward from the stored next-run.
        future = _compute_next_run(next_run, schedule.cadence, schedule.timezone, now)
        missed = 0
        cursor = next_run
        while cursor < future:
            missed += 1
            cursor = _compute_next_run(cursor, schedule.cadence, schedule.timezone, cursor)
        if missed > 0:
            schedule.missed_runs += missed
            schedule.next_run_at = future
            session.add(schedule)

    session.commit()


async def execute_schedule(
    schedule_id: UUID,
    is_manual: bool = False,
    conversation_id: UUID | None = None,
) -> None:
    from cowork.api.v1.endpoints.responses import mark_stream_active, mark_stream_finished
    from cowork.handlers.responses import ResponsesHandler
    from cowork.schemas.responses import ResponsesRequest

    session = get_open_session()
    run_service = ScheduleRunService(session)
    schedule_service = ScheduleService(session)

    run = run_service.create_run(schedule_id, is_manual=is_manual)

    error: str | None = None
    try:
        schedule = schedule_service.get_schedule(schedule_id)

        if conversation_id is None:
            # Conversation not pre-created by the caller (e.g. cron tick).
            from cowork.services.conversations import ConversationService
            conversation = ConversationService(session).create_conversation(
                topic=schedule.title,
                project_id=schedule.project_id,
            )
            conversation_id = conversation.id

        # Mark as in-flight so the client doesn't inject synthetic
        # continuation prompts while the LLM is still generating.
        # (May already be marked if the caller pre-created the conversation.)
        mark_stream_active(str(conversation_id))

        request = ResponsesRequest(
            input=schedule.prompt,
            model=schedule.model,
            stream=False,
            conversation=str(conversation_id),
        )
        await ResponsesHandler(session).handle(request)

        # Refresh schedule in case it changed during execution
        schedule = schedule_service.get_schedule(schedule_id)
        schedule.last_run_at = datetime.now(timezone.utc)
        schedule.last_result_conversation_id = conversation_id
        schedule.last_error = None
        schedule.missed_runs = 0
        session.add(schedule)

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
        if conversation_id:
            mark_stream_finished(str(conversation_id))
        try:
            run_service.finish_run(run.id, conversation_id=conversation_id, error=error)
        except Exception:
            logger.exception(f"Failed to finish run record for schedule {schedule_id}")
        session.close()


async def _scheduler_loop() -> None:
    logger.info("Scheduler loop started")
    while True:
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        session = get_open_session()
        try:
            _handle_missed_runs(session)
            now = datetime.now(timezone.utc)
            schedules = ScheduleService(session).list_schedules()
            due = [
                s for s in schedules
                if s.enabled and _as_utc(s.next_run_at) <= now
            ]
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
