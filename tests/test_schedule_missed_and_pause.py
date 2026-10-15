"""Missed runs, pause and resume, and the failure streak behind auto-pause.

A recurring schedule that fell behind runs its latest missed slot once, if
that slot is no older than seven days; a one-off runs up to an hour late and
is missed from then on. A pause stops everything that would start on its own
but lets a running attempt finish. Three cron occurrences in a row that fail
pause the schedule.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, select

from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
from cowork.models.project import Project
from cowork.models.schedule import Schedule, ScheduleOccurrence, ScheduleRun, ScheduleSkippedPeriod
from cowork.schemas.schedules import AttemptOutcome, OccurrenceStatus, ScheduleState, schedule_state
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.schedules import (
    MAX_FAILED_OCCURRENCES,
    ScheduleConflict,
    ScheduleOccurrenceService,
    ScheduleRunService,
    ScheduleService,
    Settlement,
)

POSTGRES_URI = os.environ.get("COWORK_TEST_POSTGRES_URI", "")


@pytest.fixture(params=["sqlite", "postgres"])
def engine(request, tmp_path):
    if request.param == "postgres":
        if not POSTGRES_URI:
            pytest.skip("set COWORK_TEST_POSTGRES_URI to a disposable database")
        engine = create_engine(POSTGRES_URI, pool_size=10)
        SQLModel.metadata.drop_all(engine)
    else:
        engine = create_engine(f"sqlite:///{tmp_path / 'missed.db'}", connect_args={"timeout": 30})
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Project(id=GENERAL_PROJECT_ID, name="general", path="/general"))
        session.commit()
    yield engine
    if request.param == "postgres":
        SQLModel.metadata.drop_all(engine)
    engine.dispose()


def _scoped(session: Session) -> ScopedSession:
    return ScopedSession(session, SYSTEM_SCOPE)


def _schedule(engine, *, cadence: str = "daily", next_run_at: datetime, tz: str = "UTC") -> UUID:
    """A schedule row as it stands, without the create path's normalizing,
    so a test can place its slot in the past."""
    with Session(engine) as session:
        schedule = Schedule(
            title="Report", prompt="Summarize", cadence=cadence, next_run_at=next_run_at,
            model="default", timezone=tz, project_id=GENERAL_PROJECT_ID,
        )
        session.add(schedule)
        session.commit()
        return schedule.id


def _row(engine, schedule_id: UUID) -> Schedule:
    with Session(engine) as session:
        return session.get(Schedule, schedule_id)


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _poll(engine, now: datetime | None = None, *, monkeypatch=None) -> list[UUID]:
    import cowork.scheduler as scheduler

    with Session(engine) as session:
        return scheduler._poll(_scoped(session))


def _skipped(engine, schedule_id: UUID) -> list[ScheduleSkippedPeriod]:
    with Session(engine) as session:
        return list(session.exec(
            select(ScheduleSkippedPeriod).where(ScheduleSkippedPeriod.schedule_id == schedule_id)
        ).all())


def _occurrences(engine, schedule_id: UUID) -> list[ScheduleOccurrence]:
    with Session(engine) as session:
        return list(session.exec(
            select(ScheduleOccurrence).where(ScheduleOccurrence.schedule_id == schedule_id)
        ).all())


def _attempt(engine, occurrence_id: UUID, outcome: AttemptOutcome, *, retry_delay: float | None = None):
    """Start an attempt at the occurrence and end it as the executor would."""
    with Session(engine) as session:
        service = ScheduleOccurrenceService(_scoped(session))
        started = service.start_attempt(occurrence_id, is_manual=False, lease_seconds=60)
        assert started is not None
        settlement = service.end_attempt(
            occurrence_id, started.token, outcome=outcome, error="boom", retry_delay=retry_delay,
        )
        session.commit()
        return settlement


def _claim(engine, schedule_id: UUID, key: str, *, manual: bool = False) -> UUID:
    with Session(engine) as session:
        occurrence = ScheduleOccurrenceService(_scoped(session)).claim(
            schedule_id, occurrence_key=key, trigger_source="manual" if manual else "cron",
            scheduled_for=datetime.now(timezone.utc), max_attempts=3, lease_seconds=60,
        )
        assert occurrence is not None
        return occurrence.id


# --- A recurring schedule that fell behind ----------------------------------


@pytest.mark.parametrize(
    ("cadence", "away", "step"),
    [
        ("hourly", timedelta(hours=5, minutes=10), timedelta(hours=1)),
        ("hourly", timedelta(days=40), timedelta(hours=1)),
        ("daily", timedelta(days=3, hours=2), timedelta(days=1)),
        ("daily", timedelta(days=30), timedelta(days=1)),
        ("weekly", timedelta(weeks=3, hours=1), timedelta(weeks=1)),
        ("weekly", timedelta(days=90), timedelta(weeks=1)),
    ],
)
def test_after_an_absence_the_latest_missed_slot_runs_once(engine, cadence, away, step):
    """However long the app was away, one occurrence is claimed, for the
    latest slot, and the rest are summarized in one record."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    first = now - away
    schedule_id = _schedule(engine, cadence=cadence, next_run_at=first)

    ready = _poll(engine)

    [claimed] = ready
    [occurrence] = _occurrences(engine, schedule_id)
    assert occurrence.id == claimed
    ran = _utc(occurrence.scheduled_for)
    assert now - step < ran <= now
    [period] = _skipped(engine, schedule_id)
    assert _utc(period.first_slot) == first
    assert _utc(period.last_slot) == ran - step
    assert period.slot_count == round((ran - first) / step)
    assert period.reason == "offline"
    assert _utc(_row(engine, schedule_id).next_run_at) == ran + step


def test_a_weekly_slot_almost_seven_days_late_still_runs(engine):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    late = now - timedelta(days=6, hours=23)
    schedule_id = _schedule(engine, cadence="weekly", next_run_at=late)

    [claimed] = _poll(engine)

    [occurrence] = _occurrences(engine, schedule_id)
    assert _utc(occurrence.scheduled_for) == late
    assert _skipped(engine, schedule_id) == []


@pytest.mark.parametrize(
    ("age", "runs"),
    [(timedelta(hours=1), True), (timedelta(hours=1, seconds=1), False)],
)
def test_the_lookback_keeps_a_slot_up_to_its_age_and_no_older(age, runs):
    """Every cadence has a slot at least weekly, so a schedule that is behind
    always has one inside the seven-day lookback. The cutoff itself is
    checked here with a lookback shorter than the cadence."""
    from cowork.schedule_timing import missed_slots

    now = datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc)
    missed = missed_slots("daily", now - age, "UTC", now=now, lookback=timedelta(hours=1))

    assert (missed.run is not None) is runs
    assert missed.skipped == (0 if runs else 1)


def test_a_slot_that_is_only_late_runs_without_a_skipped_record(engine):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    schedule_id = _schedule(engine, cadence="daily", next_run_at=now - timedelta(minutes=5))

    [claimed] = _poll(engine)

    assert _skipped(engine, schedule_id) == []
    assert _row(engine, schedule_id).missed_runs == 0


def test_polls_over_unchanged_state_record_nothing_twice(engine):
    import cowork.scheduler as scheduler

    now = datetime.now(timezone.utc)
    schedule_id = _schedule(engine, cadence="daily", next_run_at=now - timedelta(days=4, hours=1))
    for _ in range(3):
        with Session(engine) as session:
            scheduler._handle_missed_runs(_scoped(session), now)

    [period] = _skipped(engine, schedule_id)
    assert period.slot_count == 4
    assert _row(engine, schedule_id).missed_runs == 4


def test_two_pollers_catching_up_together_record_one_summary(engine):
    import cowork.scheduler as scheduler

    now = datetime.now(timezone.utc)
    schedule_id = _schedule(engine, cadence="hourly", next_run_at=now - timedelta(days=2))
    barrier = threading.Barrier(4)
    errors: list[BaseException] = []

    def poll():
        try:
            with Session(engine) as session:
                barrier.wait()
                scheduler._handle_missed_runs(_scoped(session), now)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=poll) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    [period] = _skipped(engine, schedule_id)
    assert _row(engine, schedule_id).missed_runs == period.slot_count


def test_a_daily_slot_keeps_its_wall_clock_time_across_a_dst_change(engine):
    """Slots step in the schedule's own time zone: a daily 9:00 in New York
    is still 9:00 after the clocks change, so the latest missed slot is."""
    import cowork.scheduler as scheduler

    first = datetime(2026, 10, 30, 13, 0, tzinfo=timezone.utc)  # 9:00 EDT
    now = datetime(2026, 11, 4, 15, 0, tzinfo=timezone.utc)  # after the switch to EST
    schedule_id = _schedule(engine, cadence="daily", next_run_at=first, tz="America/New_York")

    with Session(engine) as session:
        scheduler._handle_missed_runs(_scoped(session), now)

    assert _utc(_row(engine, schedule_id).next_run_at) == datetime(2026, 11, 4, 14, 0, tzinfo=timezone.utc)


# --- A one-off whose time passed --------------------------------------------


@pytest.mark.parametrize(
    ("late", "runs"),
    [(timedelta(minutes=59, seconds=59), True), (timedelta(hours=1), False), (timedelta(days=3), False)],
)
def test_a_one_off_runs_under_an_hour_late_and_is_missed_from_then_on(engine, late, runs):
    import cowork.scheduler as scheduler

    now = datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc)
    schedule_id = _schedule(engine, cadence="once", next_run_at=now - late)

    with Session(engine) as session:
        scheduler._handle_missed_runs(_scoped(session), now)
        due = scheduler._due_schedules(_scoped(session), now)

    row = _row(engine, schedule_id)
    if runs:
        assert [s.id for s in due] == [schedule_id]
        assert row.missed_at is None
        assert schedule_state(row) == ScheduleState.active
    else:
        assert due == []
        assert row.missed_at is not None
        assert schedule_state(row) == ScheduleState.missed
        [period] = _skipped(engine, schedule_id)
        assert period.reason == "one_off_missed"


def test_run_now_runs_a_missed_one_off_and_uses_it_up(engine):
    import cowork.scheduler as scheduler

    now = datetime.now(timezone.utc)
    schedule_id = _schedule(engine, cadence="once", next_run_at=now - timedelta(days=1))
    with Session(engine) as session:
        scheduler._handle_missed_runs(_scoped(session), now)
    assert schedule_state(_row(engine, schedule_id)) == ScheduleState.missed

    with Session(engine) as session:
        assert scheduler.claim_manual_run(_scoped(session), schedule_id) is not None

    row = _row(engine, schedule_id)
    assert row.missed_at is None
    assert schedule_state(row) == ScheduleState.done


def test_resume_refuses_a_one_off_whose_time_has_passed(engine):
    import cowork.scheduler as scheduler

    now = datetime.now(timezone.utc)
    schedule_id = _schedule(engine, cadence="once", next_run_at=now - timedelta(days=1))
    with Session(engine) as session:
        scheduler._handle_missed_runs(_scoped(session), now)

    with Session(engine) as session, pytest.raises(ScheduleConflict):
        ScheduleService(_scoped(session)).resume_schedule(schedule_id)


def test_giving_a_missed_one_off_a_new_time_schedules_it_again(engine):
    import cowork.scheduler as scheduler

    now = datetime.now(timezone.utc)
    schedule_id = _schedule(engine, cadence="once", next_run_at=now - timedelta(days=1))
    with Session(engine) as session:
        scheduler._handle_missed_runs(_scoped(session), now)
    later = now + timedelta(days=1)

    with Session(engine) as session:
        edited = ScheduleService(_scoped(session)).update_schedule(
            schedule_id, next_run_at=later, enabled=False,
        )
        assert schedule_state(edited) == ScheduleState.active
        assert _utc(edited.next_run_at) == later


def test_a_one_off_that_ran_is_done_not_paused(engine):
    import cowork.scheduler as scheduler

    now = datetime.now(timezone.utc)
    schedule_id = _schedule(engine, cadence="once", next_run_at=now - timedelta(minutes=1))
    [claimed] = _poll(engine)

    row = _row(engine, schedule_id)
    assert row.enabled is False
    assert schedule_state(row) == ScheduleState.done
    # A retry of a one-off still runs: its claim disabled it, nobody paused it.
    assert _attempt(engine, claimed, AttemptOutcome.failed_before_execution, retry_delay=0) == (
        Settlement.retry_scheduled
    )
    with Session(engine) as session:
        assert ScheduleOccurrenceService(_scoped(session)).start_attempt(
            claimed, is_manual=False, lease_seconds=60
        ) is not None


# --- Pause ------------------------------------------------------------------


def test_a_pause_during_backoff_cancels_the_retry(engine):
    now = datetime.now(timezone.utc)
    schedule_id = _schedule(engine, next_run_at=now + timedelta(days=1))
    occurrence_id = _claim(engine, schedule_id, "slot:a")
    assert _attempt(engine, occurrence_id, AttemptOutcome.failed_before_execution, retry_delay=0) == (
        Settlement.retry_scheduled
    )

    with Session(engine) as session:
        paused = ScheduleService(_scoped(session)).pause_schedule(schedule_id)

    assert schedule_state(paused) == ScheduleState.paused
    [occurrence] = _occurrences(engine, schedule_id)
    assert occurrence.status == OccurrenceStatus.cancelled
    assert _poll(engine) == []


def test_a_pause_cancels_a_claimed_slot_that_has_not_started(engine):
    now = datetime.now(timezone.utc)
    schedule_id = _schedule(engine, next_run_at=now + timedelta(days=1))
    occurrence_id = _claim(engine, schedule_id, "slot:a")

    with Session(engine) as session:
        ScheduleService(_scoped(session)).pause_schedule(schedule_id)
    with Session(engine) as session:
        assert ScheduleOccurrenceService(_scoped(session)).start_attempt(
            occurrence_id, is_manual=False, lease_seconds=60
        ) is None


def test_a_pause_lets_the_running_attempt_finish_and_nothing_follows(engine):
    """The attempt that was running when the pause landed ends normally. A
    failure it had before acting is not retried, and the occurrence settles."""
    now = datetime.now(timezone.utc)
    schedule_id = _schedule(engine, next_run_at=now + timedelta(days=1))
    occurrence_id = _claim(engine, schedule_id, "slot:a")
    with Session(engine) as session:
        started = ScheduleOccurrenceService(_scoped(session)).start_attempt(
            occurrence_id, is_manual=False, lease_seconds=60
        )

    with Session(engine) as session:
        ScheduleService(_scoped(session)).pause_schedule(schedule_id)
    with Session(engine) as session:
        service = ScheduleOccurrenceService(_scoped(session))
        assert service.renew(occurrence_id, started.token, lease_seconds=60)
        assert service.end_attempt(
            occurrence_id, started.token, outcome=AttemptOutcome.failed_before_execution,
            error="boom", retry_delay=0,
        ) == Settlement.settled
        session.commit()

    [occurrence] = _occurrences(engine, schedule_id)
    assert occurrence.status == OccurrenceStatus.failed


def test_a_retry_scheduled_as_the_pause_landed_never_starts_and_resume_drops_it(engine):
    """The attempt read the schedule unpaused and scheduled its retry just as
    the pause committed. The retry is refused at its start, and resuming
    clears it so the next slot can be claimed."""
    now = datetime.now(timezone.utc)
    schedule_id = _schedule(engine, next_run_at=now + timedelta(days=1))
    occurrence_id = _claim(engine, schedule_id, "slot:a")
    assert _attempt(engine, occurrence_id, AttemptOutcome.failed_before_execution, retry_delay=0) == (
        Settlement.retry_scheduled
    )
    with Session(engine) as session:  # the pause, without its cancellation
        row = session.get(Schedule, schedule_id)
        row.enabled, row.paused_at = False, now
        session.add(row)
        session.commit()

    with Session(engine) as session:
        assert ScheduleOccurrenceService(_scoped(session)).start_attempt(
            occurrence_id, is_manual=False, lease_seconds=60
        ) is None
    with Session(engine) as session:
        ScheduleService(_scoped(session)).resume_schedule(schedule_id)

    [occurrence] = _occurrences(engine, schedule_id)
    assert occurrence.status == OccurrenceStatus.cancelled


def test_a_pause_racing_a_claim_leaves_one_deterministic_outcome(engine):
    """Either the claim lands first and the pause cancels it before it
    starts, or the pause lands first and there is nothing to claim. Either
    way no attempt starts."""
    import cowork.scheduler as scheduler

    now = datetime.now(timezone.utc)
    schedule_id = _schedule(engine, next_run_at=now - timedelta(seconds=10))
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []
    claimed: list[UUID | None] = []

    def claim():
        try:
            with Session(engine) as session:
                schedule = session.get(Schedule, schedule_id)
                barrier.wait()
                claimed.append(scheduler._claim_slot(_scoped(session), schedule))
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    def pause():
        try:
            with Session(engine) as session:
                barrier.wait()
                ScheduleService(_scoped(session)).pause_schedule(schedule_id)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=claim), threading.Thread(target=pause)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    if claimed[0] is not None:
        with Session(engine) as session:
            assert ScheduleOccurrenceService(_scoped(session)).start_attempt(
                claimed[0], is_manual=False, lease_seconds=60
            ) is None
    assert all(o.status != OccurrenceStatus.active for o in _occurrences(engine, schedule_id))


def test_run_now_still_runs_a_paused_schedule(engine):
    import cowork.scheduler as scheduler

    schedule_id = _schedule(engine, next_run_at=datetime.now(timezone.utc) + timedelta(days=1))
    with Session(engine) as session:
        ScheduleService(_scoped(session)).pause_schedule(schedule_id)

    with Session(engine) as session:
        occurrence_id = scheduler.claim_manual_run(_scoped(session), schedule_id)
    with Session(engine) as session:
        assert ScheduleOccurrenceService(_scoped(session)).start_attempt(
            occurrence_id, is_manual=True, lease_seconds=60
        ) is not None


def test_saving_an_edit_with_the_current_enabled_value_changes_nothing(engine):
    schedule_id = _schedule(engine, next_run_at=datetime.now(timezone.utc) + timedelta(days=1))
    with Session(engine) as session:
        ScheduleService(_scoped(session)).pause_schedule(schedule_id)
    with Session(engine) as session:
        edited = ScheduleService(_scoped(session)).update_schedule(schedule_id, title="Renamed", enabled=False)
    assert schedule_state(edited) == ScheduleState.paused


# --- Auto-pause -------------------------------------------------------------


def _fail(engine, schedule_id: UUID, count: int, *, start: int = 0, manual: bool = False) -> None:
    for n in range(start, start + count):
        occurrence_id = _claim(engine, schedule_id, f"{'manual' if manual else 'slot'}:{n}", manual=manual)
        _attempt(engine, occurrence_id, AttemptOutcome.failed_during_execution)


def test_three_failed_occurrences_in_a_row_pause_the_schedule_once(engine):
    schedule_id = _schedule(engine, next_run_at=datetime.now(timezone.utc) + timedelta(days=1))

    _fail(engine, schedule_id, MAX_FAILED_OCCURRENCES - 1)
    assert schedule_state(_row(engine, schedule_id)) == ScheduleState.active

    _fail(engine, schedule_id, 1, start=MAX_FAILED_OCCURRENCES)
    row = _row(engine, schedule_id)
    assert schedule_state(row) == ScheduleState.auto_paused
    paused_at = row.auto_paused_at

    _fail(engine, schedule_id, 1, start=10, manual=True)
    assert _row(engine, schedule_id).auto_paused_at == paused_at


def test_retries_inside_one_occurrence_count_once(engine):
    schedule_id = _schedule(engine, next_run_at=datetime.now(timezone.utc) + timedelta(days=1))
    occurrence_id = _claim(engine, schedule_id, "slot:a")
    for _ in range(3):
        with Session(engine) as session:
            row = session.get(ScheduleOccurrence, occurrence_id)
            row.next_attempt_at = datetime.now(timezone.utc)
            session.add(row)
            session.commit()
        _attempt(engine, occurrence_id, AttemptOutcome.failed_before_execution, retry_delay=0)

    row = _row(engine, schedule_id)
    assert row.consecutive_failures == 1
    assert schedule_state(row) == ScheduleState.active


def test_a_success_resets_the_streak_and_other_endings_leave_it(engine):
    schedule_id = _schedule(engine, next_run_at=datetime.now(timezone.utc) + timedelta(days=1))
    _fail(engine, schedule_id, 2)
    _fail(engine, schedule_id, 5, start=10, manual=True)
    _attempt(engine, _claim(engine, schedule_id, "slot:cancelled"), AttemptOutcome.cancelled)
    assert _row(engine, schedule_id).consecutive_failures == 2

    _attempt(engine, _claim(engine, schedule_id, "slot:ok"), AttemptOutcome.completed)
    assert _row(engine, schedule_id).consecutive_failures == 0


def test_a_reaped_and_a_boot_reaped_failure_count_once_each(engine):
    schedule_id = _schedule(engine, next_run_at=datetime.now(timezone.utc) + timedelta(days=1))
    occurrence_id = _claim(engine, schedule_id, "slot:a")
    with Session(engine) as session:
        service = ScheduleOccurrenceService(_scoped(session))
        started = service.start_attempt(occurrence_id, is_manual=False, lease_seconds=-1)
        assert service.begin_execution(started, occurrence_id, lease_seconds=-1)
    for _ in range(2):  # a second reaper finds nothing left to reap
        with Session(engine) as session:
            service = ScheduleOccurrenceService(_scoped(session))
            for lease in service.expired_leases(datetime.now(timezone.utc)):
                service.reap(lease, now=datetime.now(timezone.utc), error="lapsed", retry_delay=lambda _n: 0)
    assert _row(engine, schedule_id).consecutive_failures == 1

    other = _claim(engine, schedule_id, "slot:b")
    with Session(engine) as session:
        ScheduleOccurrenceService(_scoped(session)).start_attempt(other, is_manual=False, lease_seconds=60)
    for _ in range(2):
        with Session(engine) as session:
            ScheduleRunService(_scoped(session)).reap_orphaned_runs()
    assert _row(engine, schedule_id).consecutive_failures == 2


def test_resume_clears_the_pause_and_streak_keeps_history_and_waits_for_the_next_slot(engine):
    now = datetime.now(timezone.utc)
    schedule_id = _schedule(engine, cadence="hourly", next_run_at=now - timedelta(hours=3))
    _fail(engine, schedule_id, MAX_FAILED_OCCURRENCES)

    with Session(engine) as session:
        resumed = ScheduleService(_scoped(session)).resume_schedule(schedule_id)

    assert schedule_state(resumed) == ScheduleState.active
    assert resumed.consecutive_failures == 0
    assert _utc(resumed.next_run_at) > now
    assert _poll(engine) == []
    assert len(_occurrences(engine, schedule_id)) == MAX_FAILED_OCCURRENCES
    with Session(engine) as session:
        assert len(session.exec(select(ScheduleRun).where(ScheduleRun.schedule_id == schedule_id)).all()) == (
            MAX_FAILED_OCCURRENCES
        )


# --- Edits ------------------------------------------------------------------


def test_editing_cadence_or_timezone_moves_the_next_slot_at_once(engine):
    now = datetime.now(timezone.utc)
    schedule_id = _schedule(engine, cadence="daily", next_run_at=now + timedelta(hours=1))
    saturday = datetime(2030, 1, 5, 9, 0, tzinfo=timezone.utc)
    assert saturday.weekday() == 5

    with Session(engine) as session:
        service = ScheduleService(_scoped(session))
        edited = service.update_schedule(schedule_id, cadence="weekdays", next_run_at=saturday)
        assert _utc(edited.next_run_at) == saturday + timedelta(days=2)
        # A start already in the past moves to the next slot instead of
        # firing as a missed run on the next poll.
        stale = service.update_schedule(schedule_id, cadence="daily", next_run_at=now - timedelta(days=3))
        assert _utc(stale.next_run_at) > now


def test_a_cron_success_does_not_suppress_the_next_slot(engine):
    """The freshness guard is for a manual run just before a slot. A cron run
    that just finished is its own slot and must not skip the next one."""
    import cowork.scheduler as scheduler

    now = datetime.now(timezone.utc)
    schedule_id = _schedule(engine, cadence="hourly", next_run_at=now - timedelta(seconds=30))
    with Session(engine) as session:
        runs = ScheduleRunService(_scoped(session))
        runs.finish_run(runs.create_run(schedule_id, is_manual=False).id)

    with Session(engine) as session:
        assert [s.id for s in scheduler._due_schedules(_scoped(session), now)] == [schedule_id]
