from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Awaitable, Callable
from uuid import UUID, uuid4

from cowork.common.logger import get_logger
from cowork.db.session import get_open_session
from cowork.models.schedule import Schedule
from cowork.schemas.schedules import Cadence, MissedRunPolicy, ScheduleHealth
from cowork.services.schedules import ScheduleRunService, ScheduleService

logger = get_logger(__name__)

_POLL_INTERVAL_SECONDS = 30
_scheduler_task: asyncio.Task | None = None

# ── Failure policy tunables (module-level so tests can monkeypatch) ──
# A single run is retried in-process on transient failures with capped
# exponential backoff. Distinct from re-firing on the next cadence tick:
# these retries all belong to the *same* occurrence.
_MAX_ATTEMPTS = 3                 # total tries for one run (1 + 2 retries)
_RETRY_BASE_DELAY_SECONDS = 2.0   # backoff = base * 2**(attempt-1)
_RETRY_MAX_DELAY_SECONDS = 30.0   # cap per-retry sleep
_RUN_TIMEOUT_SECONDS = 600.0      # bound a single attempt; hung run -> failure
# After this many *consecutive* failed runs the task auto-pauses instead of
# failing silently forever.
_MAX_CONSECUTIVE_FAILURES = 5

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

# Upper bound on how many missed occurrences a ``catch_up`` task will replay
# on resume — protects against a months-closed app stampeding hundreds of runs.
_MAX_CATCH_UP = 10

# Error substrings we treat as permanent (no point retrying the same run).
# Everything else is assumed transient (network blips, rate limits, timeouts).
_PERMANENT_ERROR_MARKERS = (
    "not found",
    "invalid",
    "unauthorized",
    "forbidden",
    "validation",
)


def _as_utc(dt: datetime) -> datetime:
    """Treat a naive datetime as UTC (legacy rows stored without tzinfo)."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _is_transient(exc: Exception) -> bool:
    """Heuristic: should this failure be retried within the same run?

    Timeouts and generic connection errors are transient; clearly-permanent
    failures (bad config, auth, validation) are not — retrying them just
    burns the backoff budget to land on the same error.
    """
    if isinstance(exc, asyncio.TimeoutError):
        return True
    msg = str(exc).lower()
    return not any(marker in msg for marker in _PERMANENT_ERROR_MARKERS)


def _occurrence_key(schedule: Schedule) -> str:
    """Idempotency key for the occurrence currently due.

    Derived from the schedule id + the scheduled instant so the same
    occurrence can never be run twice — whether the scheduler ticks twice
    in quick succession or a catch-up overlaps a live tick. Manual runs use
    a fresh uuid key (see ``execute_schedule``) so they never collide with a
    scheduled occurrence.
    """
    return f"{schedule.id}@{_as_utc(schedule.next_run_at).isoformat()}"


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


def _committed_next_run_at(session, schedule_id: UUID) -> datetime | None:
    """The ``next_run_at`` currently *committed* to the DB for a schedule.

    Reads past any uncommitted in-session changes (the session runs with
    ``autoflush=False``) so a live tick can tell whether a concurrent
    missed-runs sweep already moved the cadence forward while its run was
    in-flight. Returns None if the row is gone.
    """
    from sqlmodel import select

    value = session.exec(
        select(Schedule.next_run_at).where(Schedule.id == schedule_id)
    ).first()
    return _as_utc(value) if value is not None else None


def _recurring_occurrences(schedule: Schedule, now: datetime) -> tuple[list[datetime], datetime]:
    """Missed scheduled instants for a recurring task and the next future one.

    Walks the cadence forward from the stored next-run, collecting every
    occurrence already due (``<= now``) and returning the first future
    occurrence to fast-forward to.
    """
    next_run = _as_utc(schedule.next_run_at)
    missed: list[datetime] = []
    cursor = next_run
    while cursor <= now:
        missed.append(cursor)
        cursor = _compute_next_run(cursor, schedule.cadence, schedule.timezone, cursor)
    return missed, cursor


def _handle_missed_runs(session) -> list[tuple[UUID, str]]:
    """Reconcile occurrences that came due while the scheduler was offline.

    The most-recent occurrence that's due always runs (the normal cadence
    tick / first run after reopening the app). The older *backlog* is what
    each schedule's ``missed_run_policy`` governs:

      skip      — drop the backlog; run only the current occurrence.
      run_once  — collapse the backlog into a single catch-up + current.
      catch_up  — replay every backlog occurrence (bounded, oldest-first)
                  + current.

    Either way ``next_run_at`` is fast-forwarded past everything due and
    ``missed_runs`` counts how many scheduled times elapsed.

    Returns ``(schedule_id, idempotency_key)`` pairs the caller should
    execute. Each key is occurrence-stable, so a catch-up that races a live
    cadence tick (or a second poll) cannot double-fire.
    """
    now = datetime.now(timezone.utc)
    to_run: list[tuple[UUID, str]] = []
    run_service = ScheduleRunService(session)
    schedules = ScheduleService(session).list_schedules()
    for schedule in schedules:
        if not schedule.enabled:
            continue

        next_run = _as_utc(schedule.next_run_at)
        if next_run > now:
            continue

        policy = schedule.missed_run_policy or MissedRunPolicy.skip

        if schedule.cadence == Cadence.once:
            # A one-off whose time passed. Under skip it's silently dropped;
            # otherwise it still gets to run exactly once on resume. The key
            # is the scheduled instant so a manual run-now can't double it.
            if policy != MissedRunPolicy.skip:
                key = _occurrence_key(schedule)
                if not run_service.run_exists(schedule.id, key):
                    to_run.append((schedule.id, key))
            # Either way the one-off is now spent: disable it.
            schedule.enabled = False
            session.add(schedule)
            continue

        if schedule.cadence not in _CALENDAR_DELTAS and schedule.cadence != Cadence.hourly:
            continue

        overdue, future = _recurring_occurrences(schedule, now)
        if not overdue:
            continue

        # Fast-forward the cadence past everything that's already due and
        # record how many scheduled times elapsed (drives the card's
        # "missed N runs" note).
        schedule.missed_runs += len(overdue)
        schedule.next_run_at = future
        session.add(schedule)

        # The *most recent* overdue occurrence is the run that's due right
        # now — it always fires, regardless of policy, so a normal cadence
        # tick (and the first run after reopening the app) actually runs.
        # Everything older is the offline *backlog*, governed by policy:
        #   skip      — drop the backlog (run only the current occurrence).
        #   run_once  — collapse the whole backlog into a single catch-up.
        #   catch_up  — replay every backlog occurrence, bounded + oldest-first.
        current = overdue[-1:]
        backlog = overdue[:-1]
        if policy == MissedRunPolicy.catch_up:
            replay = backlog[-_MAX_CATCH_UP:] + current
        elif policy == MissedRunPolicy.run_once:
            replay = backlog[-1:] + current
        else:  # skip
            replay = current

        for instant in replay:
            key = f"{schedule.id}@{instant.isoformat()}"
            if not run_service.run_exists(schedule.id, key):
                to_run.append((schedule.id, key))

    session.commit()
    return to_run


async def _attempt_with_retries(work: Callable[[], Awaitable[None]]) -> int:
    """Run ``work`` with a per-attempt timeout and capped backoff retries.

    Returns the number of attempts taken on success. Re-raises the last
    exception once the budget is exhausted or a permanent failure is hit —
    each attempt is bounded by ``_RUN_TIMEOUT_SECONDS`` so a hung run can't
    block the scheduler forever.
    """
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            await asyncio.wait_for(work(), timeout=_RUN_TIMEOUT_SECONDS)
            return attempt
        except Exception as exc:  # noqa: BLE001 — re-raised below
            last_exc = exc
            if attempt >= _MAX_ATTEMPTS or not _is_transient(exc):
                raise
            delay = min(
                _RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
                _RETRY_MAX_DELAY_SECONDS,
            )
            logger.warning(
                "Schedule run attempt %d/%d failed (%s); retrying in %.1fs",
                attempt, _MAX_ATTEMPTS, exc, delay,
            )
            await asyncio.sleep(delay)
    # Unreachable: the loop either returns or raises, but keep the type checker happy.
    assert last_exc is not None
    raise last_exc


def _apply_success(schedule: Schedule, conversation_id: UUID | None, session) -> None:
    schedule.last_run_at = datetime.now(timezone.utc)
    schedule.last_result_conversation_id = conversation_id
    schedule.last_error = None
    schedule.missed_runs = 0
    schedule.consecutive_failures = 0
    schedule.health = ScheduleHealth.ok
    session.add(schedule)


def _apply_failure(schedule: Schedule, error: str, session) -> None:
    """Record a failed run and auto-pause once failures pile up.

    A run that exhausted its in-process retries counts as one consecutive
    failure. After ``_MAX_CONSECUTIVE_FAILURES`` of them the task disables
    itself so it stops failing silently on every tick — the user resumes it
    once they've dealt with whatever ``last_error`` reports.
    """
    schedule.last_error = error
    schedule.consecutive_failures = (schedule.consecutive_failures or 0) + 1
    if schedule.consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
        schedule.enabled = False
        schedule.health = ScheduleHealth.paused
        logger.error(
            "Schedule %s auto-paused after %d consecutive failures; last error: %s",
            schedule.id, schedule.consecutive_failures, error,
        )
    else:
        schedule.health = ScheduleHealth.failing
    session.add(schedule)


def _noop_mark(_conversation_id: str) -> None:
    """Fallback when the in-flight-marking helpers aren't available.

    Marking a conversation in-flight is a best-effort hint that stops the
    renderer injecting a synthetic continuation prompt mid-generation; the
    streaming registry also tracks this itself, so a missing helper must not
    take the whole run down.
    """


async def execute_schedule(
    schedule_id: UUID,
    is_manual: bool = False,
    conversation_id: UUID | None = None,
    idempotency_key: str | None = None,
) -> None:
    try:
        from cowork.api.v1.endpoints.responses import (
            mark_stream_active,
            mark_stream_finished,
        )
    except ImportError:
        mark_stream_active = mark_stream_finished = _noop_mark
    from cowork.handlers.responses import ResponsesHandler
    from cowork.schemas.responses import ResponsesRequest

    session = get_open_session()
    run_service = ScheduleRunService(session)
    schedule_service = ScheduleService(session)

    # Idempotency guard: a scheduled occurrence carries an occurrence-stable
    # key; a manual run gets a fresh uuid so it never collides. If a run with
    # this key already exists, this occurrence already fired — don't double it.
    # Resolve the occurrence key, and decide whether this run is the *live*
    # cadence tick (vs. a manual run or an offline catch-up replay). Only the
    # live tick advances ``next_run_at`` — catch-up runs were already
    # fast-forwarded past in ``_handle_missed_runs``, so advancing again here
    # would skip a future occurrence.
    _sched_now = schedule_service.get_schedule(schedule_id)
    tick_instant = _as_utc(_sched_now.next_run_at)
    current_key = _occurrence_key(_sched_now)
    if idempotency_key is None:
        idempotency_key = uuid4().hex if is_manual else current_key
    is_live_tick = not is_manual and idempotency_key == current_key
    if run_service.run_exists(schedule_id, idempotency_key):
        logger.info(
            "Schedule %s occurrence %s already ran — skipping duplicate",
            schedule_id, idempotency_key,
        )
        session.close()
        return

    try:
        run = run_service.create_run(
            schedule_id, is_manual=is_manual, idempotency_key=idempotency_key
        )
    except Exception:
        # Lost a race to insert the same key (unique constraint) — another
        # task is already handling this occurrence. Bail without firing.
        session.rollback()
        logger.info(
            "Schedule %s occurrence %s claimed by a concurrent run — skipping",
            schedule_id, idempotency_key,
        )
        session.close()
        return

    error: str | None = None
    attempts = 1
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

        async def _do_work() -> None:
            request = ResponsesRequest(
                input=schedule.prompt,
                model=schedule.model,
                stream=False,
                conversation=str(conversation_id),
            )
            await ResponsesHandler(session).handle(request)

        attempts = await _attempt_with_retries(_do_work)

        # Refresh schedule in case it changed during execution
        schedule = schedule_service.get_schedule(schedule_id)
        _apply_success(schedule, conversation_id, session)

        # Reconcile the cadence. A run can outlive a poll interval, so a
        # concurrent missed-runs sweep may have fast-forwarded next_run_at
        # while we were in-flight. Compare against the *committed* value:
        #  - unchanged → we own this tick, advance to the next occurrence.
        #  - already moved → adopt that value so our commit (the schedule
        #    row is dirty from _apply_success) doesn't clobber it back to a
        #    stale instant and skip an occurrence.
        if is_live_tick:
            committed = _committed_next_run_at(session, schedule_id)
            if committed == tick_instant:
                _advance_next_run_at(schedule, session)
            elif committed is not None:
                schedule.next_run_at = committed
                session.add(schedule)

        session.commit()

    except Exception as exc:
        # Some exceptions (notably asyncio.TimeoutError) stringify to "" — fall
        # back to the type name so the run is recorded as failed, not silently
        # "succeeded" by an empty-but-falsy error.
        error = str(exc) or type(exc).__name__
        logger.exception(f"Schedule {schedule_id} run failed: {error}")
        try:
            schedule = schedule_service.get_schedule(schedule_id)
            _apply_failure(schedule, error, session)
            # A failed live tick still advances the cadence so we don't
            # busy-loop re-firing the same overdue occurrence every poll.
            # (Auto-pause aside — a disabled task won't be polled anyway.)
            # Same in-flight-sweep reconciliation as the success path.
            if is_live_tick and schedule.enabled:
                committed = _committed_next_run_at(session, schedule_id)
                if committed == tick_instant:
                    _advance_next_run_at(schedule, session)
                elif committed is not None:
                    schedule.next_run_at = committed
                    session.add(schedule)
            session.commit()
        except Exception:
            session.rollback()
    finally:
        if conversation_id:
            mark_stream_finished(str(conversation_id))
        try:
            run_service.finish_run(
                run.id, conversation_id=conversation_id, error=error, attempts=attempts
            )
        except Exception:
            logger.exception(f"Failed to finish run record for schedule {schedule_id}")
        session.close()


async def _scheduler_loop() -> None:
    logger.info("Scheduler loop started")
    while True:
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        session = get_open_session()
        catch_up: list[tuple[UUID, str]] = []
        due: list[tuple[UUID, str]] = []
        try:
            # Reconcile offline gaps first; this fast-forwards next_run_at and
            # hands back any catch-up occurrences to replay.
            catch_up = _handle_missed_runs(session)
            now = datetime.now(timezone.utc)
            schedules = ScheduleService(session).list_schedules()
            due = [
                (s.id, _occurrence_key(s))
                for s in schedules
                if s.enabled and _as_utc(s.next_run_at) <= now
            ]
        except Exception:
            logger.exception("Scheduler loop error during poll")
        finally:
            session.close()

        # Idempotency keys make these safe even if a catch-up occurrence and a
        # live tick name the same instant — the second create_run is rejected.
        for schedule_id, key in catch_up + due:
            asyncio.create_task(
                execute_schedule(schedule_id, is_manual=False, idempotency_key=key)
            )


def start_scheduler() -> None:
    global _scheduler_task
    if _scheduler_task is not None and not _scheduler_task.done():
        return
    _scheduler_task = asyncio.create_task(_scheduler_loop())
    logger.info("Scheduler background task created")
