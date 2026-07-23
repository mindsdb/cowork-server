"""The morning digest: canonical schedule template.

One place owns the digest's schedule shape so the agent, onboarding (O1),
and tests can never drift apart: requires_browser (it works live tabs), a
daily slot at the user's hour, and the prompt that drives the skill.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlmodel import Session

from cowork.models.schedule import Schedule
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.schedules import ScheduleService

DIGEST_TITLE = "Morning digest"

DIGEST_PROMPT = (
    "Run the morning-digest skill: gather overnight mail subjects, "
    "Linear/project changes, and today's calendar from my live tabs and "
    "connectors; produce one digest artifact; park exactly one approval if "
    "anything needs to go out. Read-mostly — sign-in walls are noted, never "
    "crossed."
)


def _next_slot(hour: int, tz_name: str, now: datetime | None = None) -> datetime:
    """The next occurrence of `hour` today-or-tomorrow, UTC-stamped.

    The scheduler's cadence handling owns real timezone math; the template
    just needs a sane first slot (next 9:00 local-ish, stored UTC).
    """
    now = now or datetime.now(timezone.utc)
    slot = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if slot <= now:
        slot += timedelta(days=1)
    return slot


def create_digest_schedule(
    session: Session,
    *,
    hour: int = 9,
    timezone_name: str = "UTC",
    project_id: UUID = GENERAL_PROJECT_ID,
) -> Schedule:
    """Create the morning digest schedule (requires_browser, daily at `hour`)."""
    if not 0 <= hour <= 23:
        raise ValueError("hour must be 0-23")
    return ScheduleService(session).create_schedule(
        title=DIGEST_TITLE,
        prompt=DIGEST_PROMPT,
        cadence="daily",
        next_run_at=_next_slot(hour, timezone_name),
        model="default",
        timezone=timezone_name,
        project_id=project_id,
        enabled=True,
        requires_browser=True,
    )
