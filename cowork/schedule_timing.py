from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tzlocal import get_localzone

from cowork.common.datetime_utils import ensure_utc
from cowork.common.logger import get_logger
from cowork.schemas.schedules import Cadence

logger = get_logger(__name__)


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

    if cadence == Cadence.hourly:
        return current + timedelta(hours=1)

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
