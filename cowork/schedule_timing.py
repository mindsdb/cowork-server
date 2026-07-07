from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from cowork.common.datetime_utils import ensure_utc
from cowork.schemas.schedules import Cadence


def resolve_timezone(name: str | None) -> ZoneInfo:
    if not name or name == "local":
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

    if cadence == Cadence.hourly:
        return current + timedelta(hours=1)

    tz = resolve_timezone(timezone_name)
    local = current.astimezone(tz)
    if cadence == Cadence.daily:
        next_date = local.date() + timedelta(days=1)
    elif cadence == Cadence.weekly:
        next_date = local.date() + timedelta(weeks=1)
    else:
        raise ValueError(f"Unsupported cadence for recurrence: {cadence}")

    next_local = datetime.combine(next_date, local.time(), tzinfo=tz)
    return next_local.astimezone(UTC)


def next_future_occurrence(
    cadence: str,
    next_run_at: datetime,
    timezone_name: str,
    *,
    now: datetime | None = None,
) -> datetime:
    """Fast-forward ``next_run_at`` to the first occurrence after ``now``."""
    reference = ensure_utc(now or datetime.now(UTC))
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
    reference = ensure_utc(now or datetime.now(UTC))
    cursor = ensure_utc(next_run_at)
    if reference is None or cursor is None:
        raise ValueError("next_run_at and now must be datetimes")

    missed = 0
    while cursor <= reference:
        missed += 1
        cursor = advance_occurrence(cadence, cursor, timezone_name)
    return missed, cursor
