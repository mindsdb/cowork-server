from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import cowork.schedule_timing as schedule_timing
from cowork.schedule_timing import (
    advance_occurrence,
    count_missed_occurrences,
    next_future_occurrence,
    resolve_timezone,
)
from cowork.schemas.schedules import Cadence


def test_resolve_timezone_local_uses_system_zone(monkeypatch):
    # "local" must resolve to the machine's real zone, not silently to UTC.
    monkeypatch.setattr(schedule_timing, "get_localzone", lambda: ZoneInfo("America/New_York"))
    assert resolve_timezone("local") == ZoneInfo("America/New_York")


def test_resolve_timezone_local_falls_back_to_utc_on_error(monkeypatch):
    def _boom():
        raise RuntimeError("no zone")

    monkeypatch.setattr(schedule_timing, "get_localzone", _boom)
    assert resolve_timezone("local") == ZoneInfo("UTC")


def test_resolve_timezone_empty_and_unknown_default_to_utc():
    assert resolve_timezone(None) == ZoneInfo("UTC")
    assert resolve_timezone("") == ZoneInfo("UTC")
    assert resolve_timezone("Not/AZone") == ZoneInfo("UTC")


def test_resolve_timezone_passes_through_valid_iana():
    assert resolve_timezone("Europe/London") == ZoneInfo("Europe/London")


def test_daily_advance_local_tracks_system_zone_dst(monkeypatch):
    # A schedule saved with "local" should keep local wall-clock across DST,
    # which the old UTC fallback broke.
    monkeypatch.setattr(schedule_timing, "get_localzone", lambda: ZoneInfo("America/New_York"))
    tz = ZoneInfo("America/New_York")
    before = datetime(2026, 3, 7, 14, 0, tzinfo=timezone.utc)  # 9:00 EST
    on_spring_day = advance_occurrence(Cadence.daily, before, "local")
    assert on_spring_day.astimezone(tz).hour == 9  # still 9:00, now EDT


def test_daily_advance_keeps_local_wall_clock():
    tz = ZoneInfo("America/New_York")
    first = datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc)
    second = advance_occurrence(Cadence.daily, first, "America/New_York")

    assert second.astimezone(tz).hour == 9
    assert second.astimezone(tz).minute == 0
    assert second.astimezone(tz).date().isoformat() == "2026-01-16"


def test_daily_advance_handles_dst_spring_forward_week():
    tz = ZoneInfo("America/New_York")
    before = datetime(2026, 3, 7, 14, 0, tzinfo=timezone.utc)
    on_spring_day = advance_occurrence(Cadence.daily, before, "America/New_York")
    after_spring = advance_occurrence(Cadence.daily, on_spring_day, "America/New_York")

    assert on_spring_day.astimezone(tz).date().isoformat() == "2026-03-08"
    assert on_spring_day.astimezone(tz).hour == 9
    assert after_spring.astimezone(tz).date().isoformat() == "2026-03-09"
    assert after_spring.astimezone(tz).hour == 9


def test_weekdays_advance_mid_week_is_next_day():
    tz = ZoneInfo("America/New_York")
    monday = datetime(2026, 1, 19, 14, 0, tzinfo=timezone.utc)  # 9:00 EST
    tuesday = advance_occurrence(Cadence.weekdays, monday, "America/New_York")

    assert tuesday.astimezone(tz).date().isoformat() == "2026-01-20"
    assert tuesday.astimezone(tz).hour == 9


def test_weekdays_advance_skips_weekend():
    tz = ZoneInfo("America/New_York")
    friday = datetime(2026, 1, 16, 14, 0, tzinfo=timezone.utc)  # 9:00 EST
    monday = advance_occurrence(Cadence.weekdays, friday, "America/New_York")

    assert monday.astimezone(tz).date().isoformat() == "2026-01-19"
    assert monday.astimezone(tz).hour == 9


def test_weekly_advance_keeps_local_wall_clock():
    tz = ZoneInfo("America/Los_Angeles")
    first = datetime(2026, 6, 3, 16, 30, tzinfo=timezone.utc)
    second = advance_occurrence(Cadence.weekly, first, "America/Los_Angeles")

    local = second.astimezone(tz)
    assert local.weekday() == 2
    assert local.hour == 9
    assert local.minute == 30


def test_hourly_advance_is_elapsed_time():
    first = datetime(2026, 6, 25, 10, 15, tzinfo=timezone.utc)
    second = advance_occurrence(Cadence.hourly, first, "America/New_York")
    assert second == datetime(2026, 6, 25, 11, 15, tzinfo=timezone.utc)


def test_next_future_occurrence_steps_until_after_now():
    anchor = datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc)
    now = datetime(2026, 6, 25, 10, 0, tzinfo=timezone.utc)
    nxt = next_future_occurrence(Cadence.daily, anchor, "UTC", now=now)

    assert nxt == datetime(2026, 6, 26, 9, 0, tzinfo=timezone.utc)


def test_count_missed_occurrences():
    anchor = datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc)
    now = datetime(2026, 6, 28, 10, 0, tzinfo=timezone.utc)
    missed, nxt = count_missed_occurrences(Cadence.daily, anchor, "UTC", now=now)

    assert missed == 4
    assert nxt == datetime(2026, 6, 29, 9, 0, tzinfo=timezone.utc)


def test_count_missed_single_overdue_occurrence():
    anchor = datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc)
    now = datetime(2026, 6, 25, 9, 3, tzinfo=timezone.utc)
    missed, nxt = count_missed_occurrences(Cadence.daily, anchor, "UTC", now=now)

    assert missed == 1
    assert nxt == datetime(2026, 6, 26, 9, 0, tzinfo=timezone.utc)


def test_count_missed_occurrences_weekdays_does_not_count_weekend():
    anchor = datetime(2026, 1, 16, 9, 0, tzinfo=timezone.utc)  # Friday
    now = datetime(2026, 1, 19, 10, 0, tzinfo=timezone.utc)  # Monday
    missed, nxt = count_missed_occurrences(Cadence.weekdays, anchor, "UTC", now=now)

    assert missed == 2  # Friday and Monday; the weekend has no occurrence to miss
    assert nxt == datetime(2026, 1, 20, 9, 0, tzinfo=timezone.utc)  # Tuesday
