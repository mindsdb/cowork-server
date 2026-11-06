from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tzlocal import get_localzone

from cowork.common.datetime_utils import ensure_utc
from cowork.common.logger import get_logger
from cowork.schemas.schedules import Cadence

logger = get_logger(__name__)

# Cadences a fixed elapsed time apart, whatever the zone or the date.
FIXED_INTERVALS = {
    Cadence.every_15_minutes: timedelta(minutes=15),
    Cadence.every_30_minutes: timedelta(minutes=30),
    Cadence.hourly: timedelta(hours=1),
}


def resolve_timezone(name: str | None) -> ZoneInfo:
    if not name:
        return ZoneInfo("UTC")
    if name == "local":
        # cowork-server runs on the user's own machine, so "local" means the
        # system's zone. Resolve it to a real (DST-aware) IANA zone instead of
        # silently falling back to UTC, which would fire schedules on the
        # wrong wall clock.
        try:
            local = get_localzone()
            if isinstance(local, ZoneInfo):
                return local
            # Older tzlocal returned a pytz zone; recover its IANA key.
            key = getattr(local, "key", None) or str(local)
            return ZoneInfo(key)
        except Exception:
            logger.warning("Could not resolve 'local' timezone; falling back to UTC")
            return ZoneInfo("UTC")
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC")


def advance_occurrence(cadence: str, at: datetime, timezone_name: str) -> datetime:
    """Return the next scheduled instant strictly after ``at``."""
    current = ensure_utc(at)
    if current is None:
        raise ValueError("at must be a datetime")

    interval = FIXED_INTERVALS.get(cadence)
    if interval is not None:
        return current + interval

    tz = resolve_timezone(timezone_name)
    local = current.astimezone(tz)
    if cadence == Cadence.daily:
        next_date = local.date() + timedelta(days=1)
    elif cadence == Cadence.weekdays:
        next_date = local.date() + timedelta(days=1)
        while next_date.weekday() >= 5:  # Saturday=5, Sunday=6
            next_date += timedelta(days=1)
    elif cadence == Cadence.weekly:
        next_date = local.date() + timedelta(weeks=1)
    else:
        raise ValueError(f"Unsupported cadence for recurrence: {cadence}")

    next_local = datetime.combine(next_date, local.time(), tzinfo=tz)
    return next_local.astimezone(timezone.utc)


def next_future_occurrence(
    cadence: str,
    next_run_at: datetime,
    timezone_name: str,
    *,
    now: datetime | None = None,
) -> datetime:
    """Fast-forward ``next_run_at`` to the first occurrence after ``now``."""
    reference = ensure_utc(now or datetime.now(timezone.utc))
    cursor = ensure_utc(next_run_at)
    if reference is None or cursor is None:
        raise ValueError("next_run_at and now must be datetimes")

    while cursor <= reference:
        cursor = advance_occurrence(cadence, cursor, timezone_name)
    return cursor


def count_missed_occurrences(
    cadence: str,
    next_run_at: datetime,
    timezone_name: str,
    *,
    now: datetime | None = None,
) -> tuple[int, datetime]:
    """Count overdue occurrences and return the first future ``next_run_at``."""
    reference = ensure_utc(now or datetime.now(timezone.utc))
    cursor = ensure_utc(next_run_at)
    if reference is None or cursor is None:
        raise ValueError("next_run_at and now must be datetimes")

    missed = 0
    while cursor <= reference:
        missed += 1
        cursor = advance_occurrence(cadence, cursor, timezone_name)
    return missed, cursor


@dataclass(frozen=True)
class MissedSlots:
    """A recurring schedule's slots from ``next_run_at`` up to now."""

    # The latest overdue slot no older than the lookback, which runs once;
    # None when every overdue slot is older than that.
    run: datetime | None
    # How many slots are passed over without running.
    skipped: int
    # The first slot after now.
    future: datetime


def missed_slots(
    cadence: str,
    next_run_at: datetime,
    timezone_name: str,
    *,
    now: datetime,
    lookback: timedelta,
) -> MissedSlots:
    """Which of the overdue slots runs: the latest one no older than
    ``lookback``, once. Every other overdue slot is passed over."""
    reference = ensure_utc(now)
    cursor = ensure_utc(next_run_at)
    if reference is None or cursor is None:
        raise ValueError("next_run_at and now must be datetimes")
    floor = reference - lookback
    latest: datetime | None = None
    count = 0
    interval = FIXED_INTERVALS.get(cadence)
    if interval is not None and cursor < floor - interval:
        # Slots well before the lookback can only be passed over, so they are
        # counted rather than walked: a schedule a year behind would otherwise
        # step through every one of them.
        jumped = int((floor - cursor) // interval) - 1
        count += jumped
        cursor += interval * jumped
    while cursor <= reference:
        count += 1
        latest = cursor
        cursor = advance_occurrence(cadence, cursor, timezone_name)
    if latest is not None and latest >= floor:
        return MissedSlots(run=latest, skipped=count - 1, future=cursor)
    return MissedSlots(run=None, skipped=count, future=cursor)


def normalize_next_run(
    cadence: str,
    next_run_at: datetime,
    timezone_name: str,
    *,
    now: datetime | None = None,
) -> datetime:
    """The first slot at or after ``next_run_at`` that is still ahead.

    A recurring start already in the past moves forward to the first slot
    after ``now`` instead of reading as a missed run. A one-off keeps the time
    it was given.
    """
    start = ensure_utc(next_run_at)
    if start is None:
        raise ValueError("next_run_at must be a datetime")
    if cadence == Cadence.once:
        return start
    reference = ensure_utc(now or datetime.now(timezone.utc))
    if start <= reference:
        start = next_future_occurrence(cadence, start, timezone_name, now=reference)
    return start
