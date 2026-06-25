"""Timezone-correctness for scheduled-task next-run computation.

The scheduler stores ``next_run_at`` as a UTC instant but advances it in
the schedule's stored IANA zone so daily/weekly cadences keep their local
wall-clock time across DST boundaries. These tests pin the DST behaviour
with ``America/Los_Angeles`` (spring-forward + fall-back) and confirm the
zone is actually read end-to-end through the DB-backed advance path.

All DB access goes through the throwaway SQLite engine wired up in
conftest.py — no real ~/.cowork data is touched.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.schedule import Schedule
from cowork.schemas.schedules import Cadence
from cowork.scheduler import (
    _advance_next_run_at,
    _compute_next_run,
    _handle_missed_runs,
)
from cowork.services.projects import GENERAL_PROJECT_ID

LA = ZoneInfo("America/Los_Angeles")
UTC = timezone.utc

# DST 2025 transitions for America/Los_Angeles:
#   spring-forward: Sun Mar  9 02:00  PST(-8) -> PDT(-7)
#   fall-back:      Sun Nov  2 02:00  PDT(-7) -> PST(-8)


def _la_9am_utc(year: int, month: int, day: int) -> datetime:
    """9:00 AM local in Los Angeles on the given date, as a UTC instant."""
    return datetime(year, month, day, 9, 0, tzinfo=LA).astimezone(UTC)


# ── pure computation: daily across DST ──────────────────────────────


def test_daily_9am_spring_forward_no_drift():
    # Last run: Sat Mar 8 2025 09:00 PST. Next must be Sun Mar 9 09:00
    # PDT — 23 wall-clock-preserving hours later, NOT a naive +24h.
    last = _la_9am_utc(2025, 3, 8)
    after = last + timedelta(seconds=1)

    nxt = _compute_next_run(last, Cadence.daily, "America/Los_Angeles", after)

    assert nxt == _la_9am_utc(2025, 3, 9)
    # Still 9 AM local — the whole point.
    assert nxt.astimezone(LA).hour == 9
    # The bug we fixed: a naive timedelta would land an hour late.
    naive = last + timedelta(days=1)
    assert naive.astimezone(LA).hour == 10
    assert nxt != naive
    # The real gap across spring-forward is 23h, not 24h.
    assert nxt - last == timedelta(hours=23)


def test_daily_9am_fall_back_no_drift():
    # Last run: Sat Nov 1 2025 09:00 PDT. Next must be Sun Nov 2 09:00
    # PST — 25 hours later across the fall-back.
    last = _la_9am_utc(2025, 11, 1)
    after = last + timedelta(seconds=1)

    nxt = _compute_next_run(last, Cadence.daily, "America/Los_Angeles", after)

    assert nxt == _la_9am_utc(2025, 11, 2)
    assert nxt.astimezone(LA).hour == 9
    assert nxt - last == timedelta(hours=25)


def test_daily_utc_zone_is_plain_24h():
    # A UTC-zoned schedule has no DST, so daily stays exactly 24h.
    last = datetime(2025, 3, 8, 9, 0, tzinfo=UTC)
    nxt = _compute_next_run(last, Cadence.daily, "UTC", last + timedelta(seconds=1))
    assert nxt == datetime(2025, 3, 9, 9, 0, tzinfo=UTC)
    assert nxt - last == timedelta(hours=24)


# ── pure computation: weekly respects the zone ──────────────────────


def test_weekly_across_spring_forward_keeps_local_time():
    # Anchor Wed Mar 5 2025 09:00 PST; one week later is Wed Mar 12,
    # by which point the zone is in PDT. Local 9 AM must be preserved,
    # so the UTC instant shifts by an hour vs. a naive 7*24h.
    last = _la_9am_utc(2025, 3, 5)
    nxt = _compute_next_run(last, Cadence.weekly, "America/Los_Angeles", last + timedelta(seconds=1))

    assert nxt == _la_9am_utc(2025, 3, 12)
    assert nxt.astimezone(LA).hour == 9
    # 6 days + 23 hours of real elapsed time across the transition.
    assert nxt - last == timedelta(days=6, hours=23)
    assert nxt != last + timedelta(weeks=1)


def test_weekly_advances_a_full_week_no_dst():
    last = _la_9am_utc(2025, 6, 1)  # well clear of any transition
    nxt = _compute_next_run(last, Cadence.weekly, "America/Los_Angeles", last + timedelta(seconds=1))
    assert nxt == _la_9am_utc(2025, 6, 8)
    assert nxt - last == timedelta(weeks=1)


# ── pure computation: hourly is DST-agnostic ────────────────────────


def test_hourly_is_fixed_interval_across_dst():
    # 01:30 PST just before spring-forward. Hourly fires every 60 real
    # minutes — the next instant is a fixed +1h regardless of the gap.
    last = datetime(2025, 3, 9, 1, 30, tzinfo=LA).astimezone(UTC)
    nxt = _compute_next_run(last, Cadence.hourly, "America/Los_Angeles", last + timedelta(seconds=1))
    assert nxt - last == timedelta(hours=1)


def test_compute_skips_to_first_future_occurrence():
    # When several cycles are already in the past, the result is the
    # first occurrence strictly after `after`, not just one step.
    last = _la_9am_utc(2025, 6, 1)
    after = _la_9am_utc(2025, 6, 4) + timedelta(hours=2)  # past Jun 4's run
    nxt = _compute_next_run(last, Cadence.daily, "America/Los_Angeles", after)
    assert nxt == _la_9am_utc(2025, 6, 5)


def test_unknown_zone_falls_back_to_utc():
    last = datetime(2025, 3, 8, 9, 0, tzinfo=UTC)
    nxt = _compute_next_run(last, Cadence.daily, "Not/AZone", last + timedelta(seconds=1))
    assert nxt == datetime(2025, 3, 9, 9, 0, tzinfo=UTC)


# ── DB-backed: the stored timezone is actually read ─────────────────


@pytest.fixture()
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


def _make_schedule(session: Session, *, cadence: str, next_run_at: datetime, tz: str) -> Schedule:
    sched = Schedule(
        title="tz-test",
        prompt="do the thing",
        cadence=cadence,
        timezone=tz,
        next_run_at=next_run_at,
        model="default",
        project_id=GENERAL_PROJECT_ID,
        enabled=True,
    )
    session.add(sched)
    session.commit()
    session.refresh(sched)
    return sched


def test_advance_reads_stored_zone_for_daily(session, monkeypatch):
    import cowork.scheduler as scheduler

    # Freeze "now" to just after the Mar 8 09:00 PST run so advancing
    # crosses the spring-forward boundary.
    last = _la_9am_utc(2025, 3, 8)
    fixed_now = last + timedelta(minutes=1)

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.astimezone(tz) if tz else fixed_now.replace(tzinfo=None)

    monkeypatch.setattr(scheduler, "datetime", _FrozenDatetime)

    sched = _make_schedule(session, cadence=Cadence.daily, next_run_at=last, tz="America/Los_Angeles")
    _advance_next_run_at(sched, session)
    session.commit()
    session.refresh(sched)

    stored = scheduler._as_utc(sched.next_run_at)
    assert stored == _la_9am_utc(2025, 3, 9)
    assert stored.astimezone(LA).hour == 9


def test_handle_missed_runs_counts_and_advances_in_zone(session, monkeypatch):
    import cowork.scheduler as scheduler

    # next_run anchored at Jun 1 09:00 PT, "now" just past Jun 4's run.
    # The anchor itself is an occurrence that came due and never ran, so
    # Jun 1, 2, 3, 4 are all missed (4 total); next future run is Jun 5.
    anchor = _la_9am_utc(2025, 6, 1)
    fixed_now = _la_9am_utc(2025, 6, 4) + timedelta(hours=2)

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.astimezone(tz) if tz else fixed_now.replace(tzinfo=None)

    monkeypatch.setattr(scheduler, "datetime", _FrozenDatetime)

    sched = _make_schedule(session, cadence=Cadence.daily, next_run_at=anchor, tz="America/Los_Angeles")
    _handle_missed_runs(session)
    session.refresh(sched)

    assert sched.missed_runs == 4
    stored = scheduler._as_utc(sched.next_run_at)
    assert stored == _la_9am_utc(2025, 6, 5)
    assert stored.astimezone(LA).hour == 9
