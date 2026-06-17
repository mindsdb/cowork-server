from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import UUID

from cowork.common.logger import get_logger
from cowork.db.session import get_open_session
from cowork.models.schedule import Schedule
from cowork.schemas.schedules import Cadence
from cowork.services.schedules import ScheduleRunService, ScheduleService

logger = get_logger(__name__)

_POLL_INTERVAL_SECONDS = 30
_scheduler_task: asyncio.Task | None = None

_CADENCE_DELTAS: dict[str, timedelta] = {
    Cadence.hourly: timedelta(hours=1),
    Cadence.daily: timedelta(days=1),
    Cadence.weekly: timedelta(weeks=1),
}


def _advance_next_run_at(schedule: Schedule, session) -> None:
    if schedule.cadence == Cadence.once:
        schedule.enabled = False
        session.add(schedule)
        return

    delta = _CADENCE_DELTAS.get(schedule.cadence)
    if delta is None:
        return

    now = datetime.now(timezone.utc)
    next_run = schedule.next_run_at
    if next_run.tzinfo is None:
        next_run = next_run.replace(tzinfo=timezone.utc)

    while next_run <= now:
        next_run += delta

    schedule.next_run_at = next_run
    session.add(schedule)


def _handle_missed_runs(session) -> None:
    now = datetime.now(timezone.utc)
    schedules = ScheduleService(session).list_schedules()
    for schedule in schedules:
        if not schedule.enabled:
            continue

        next_run = schedule.next_run_at
        if next_run.tzinfo is None:
            next_run = next_run.replace(tzinfo=timezone.utc)

        if next_run >= now:
            continue

        if schedule.cadence == Cadence.once:
            # A one-off that was never executed — disable it without running
            schedule.enabled = False
            session.add(schedule)
            continue

        delta = _CADENCE_DELTAS.get(schedule.cadence)
        if delta is None:
            continue

        overdue_seconds = (now - next_run).total_seconds()
        missed = int(overdue_seconds // delta.total_seconds())
        if missed > 0:
            schedule.missed_runs += missed
            # Fast-forward to the next future occurrence
            while next_run <= now:
                next_run += delta
            schedule.next_run_at = next_run
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
            # If the handler never ran (e.g. schedule lookup failed),
            # an eagerly-created turn buffer would linger as not-done
            # and any /tail follower would wait forever. Finish it.
            from cowork.services import stream_buffer
            buf = stream_buffer.get_buffer(str(conversation_id))
            if buf and not buf.done:
                buf.finish()
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
                if s.enabled and (
                    s.next_run_at.replace(tzinfo=timezone.utc)
                    if s.next_run_at.tzinfo is None
                    else s.next_run_at
                ) <= now
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
