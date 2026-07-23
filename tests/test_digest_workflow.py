from __future__ import annotations

from datetime import datetime, timedelta, timezone

from cowork.common.datetime_utils import ensure_utc
from cowork.services.digest import DIGEST_PROMPT, _next_slot, create_digest_schedule
from tests.test_schedule_runs import _session


def test_digest_schedule_shape():
    session = _session()
    schedule = create_digest_schedule(session, hour=9)
    assert schedule.requires_browser is True
    assert schedule.cadence == "daily"
    assert schedule.enabled is True
    assert schedule.title == "Morning digest"
    assert "morning-digest skill" in DIGEST_PROMPT
    assert "exactly one approval" in DIGEST_PROMPT
    # First slot is in the future, at 9:00 sharp.
    assert schedule.next_run_at.minute == 0
    assert schedule.next_run_at.hour == 9
    assert ensure_utc(schedule.next_run_at) > datetime.now(timezone.utc)
    session.close()


def test_next_slot_rolls_to_tomorrow_after_the_hour():
    now = datetime(2026, 7, 24, 10, 30, tzinfo=timezone.utc)
    assert _next_slot(9, "UTC", now=now) == datetime(2026, 7, 25, 9, 0, tzinfo=timezone.utc)
    early = datetime(2026, 7, 24, 6, 0, tzinfo=timezone.utc)
    assert _next_slot(9, "UTC", now=early) == datetime(2026, 7, 24, 9, 0, tzinfo=timezone.utc)


def test_hour_validation():
    import pytest

    with pytest.raises(ValueError, match="0-23"):
        create_digest_schedule(_session(), hour=25)


def test_skill_file_exists_and_carries_the_contract():
    from pathlib import Path

    text = Path("cowork/skills_builtin/morning-digest/SKILL.md").read_text()
    for phrase in ("Read-mostly", "One draft, one approval", "Sign-in walls are data", "See you at 9:00"):
        assert phrase in text
