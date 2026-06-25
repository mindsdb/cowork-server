from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import UUID

from cowork.common.datetime_utils import ensure_utc
from cowork.common.logger import get_logger
from cowork.db.session import get_open_session
from cowork.models.schedule import Schedule
from cowork.schedule_timing import count_missed_occurrences, next_future_occurrence
from cowork.schemas.schedules import Cadence
from cowork.services.schedules import ScheduleRunService, ScheduleService

logger = get_logger(__name__)

_POLL_INTERVAL_SECONDS = 30
_scheduler_task: asyncio.Task | None = None

_RECURRING_CADENCES = {Cadence.hourly, Cadence.daily, Cadence.weekly}


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


async def execute_schedule(
    schedule_id: UUID,
    is_manual: bool = False,
    conversation_id: UUID | None = None,
) -> None:
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

        request = ResponsesRequest(
            input=schedule.prompt,
            model=schedule.model,
            stream=True,
            conversation=str(conversation_id),
        )
        stream = await ResponsesHandler(session).handle(request)
        async for _ in stream:
            pass

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
            run_service = ScheduleRunService(session)
            schedules = ScheduleService(session).list_schedules()
            due = [
                s for s in schedules
                if s.enabled
                and ensure_utc(s.next_run_at) <= now
                and not run_service.has_running_run(s.id)
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
