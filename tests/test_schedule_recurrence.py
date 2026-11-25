"""Weekly tasks on several days, and monthly tasks on a day or a weekday of
the month: how their runs are stepped, and how a save stores the rule."""
from __future__ import annotations

import importlib
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
import sqlalchemy as sa
import test_turn_pool_connections as turn_pool
from alembic.migration import MigrationContext
from alembic.operations import Operations
from pydantic import ValidationError

from cowork import scheduler
from cowork.common.datetime_utils import ensure_utc
from cowork.db.scoped import SYSTEM_SCOPE
from cowork.db.units import unit_session
from cowork.harnesses.anton_harness.schedule_tools import describe_cadence
from cowork.models.schedule import Schedule
from cowork.schedule_timing import (
    advance_occurrence,
    align_to_rule,
    missed_slots,
    next_future_occurrence,
)
from cowork.schemas.schedules import Cadence, Recurrence, ScheduleCreateRequest
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.schedules import ScheduleService
from tests.integration.prereq import missing_prerequisite

NY = "America/New_York"


def _ny(*args) -> datetime:
    return datetime(*args, tzinfo=ZoneInfo(NY)).astimezone(timezone.utc)


def _runs(cadence, start, recurrence, count, tz=NY):
    runs, at = [], start
    for _ in range(count):
        at = advance_occurrence(cadence, at, tz, recurrence)
        runs.append(at.astimezone(ZoneInfo(tz)))
    return runs


# --- stepping -------------------------------------------------------------


def test_weekly_on_several_days_runs_on_each_of_them():
    # Mon 3 Nov 2025 09:00.
    runs = _runs(Cadence.weekly, _ny(2025, 11, 3, 9, 0), {"days": ["mon", "wed", "fri"]}, 4)
    assert [(r.strftime("%a"), r.day, r.hour) for r in runs] == [
        ("Wed", 5, 9), ("Fri", 7, 9), ("Mon", 10, 9), ("Wed", 12, 9),
    ]


def test_weekly_without_days_still_steps_a_week():
    runs = _runs(Cadence.weekly, _ny(2025, 11, 3, 9, 0), None, 2)
    assert [r.day for r in runs] == [10, 17]


def test_weekly_days_keep_the_wall_clock_across_a_dst_change():
    # New York leaves DST on Sun 2 Nov 2025.
    runs = _runs(Cadence.weekly, _ny(2025, 10, 31, 9, 0), {"days": ["fri", "mon"]}, 2)
    assert [(r.day, r.hour, r.utcoffset()) for r in runs] == [
        (3, 9, timedelta(hours=-5)), (7, 9, timedelta(hours=-5)),
    ]


def test_monthly_on_a_day_runs_on_a_short_months_last_day():
    runs = _runs(Cadence.monthly, _ny(2026, 1, 31, 8, 30), {"month_day": 31}, 4)
    assert [(r.month, r.day, r.hour, r.minute) for r in runs] == [
        (2, 28, 8, 30), (3, 31, 8, 30), (4, 30, 8, 30), (5, 31, 8, 30),
    ]


def test_monthly_on_the_first_monday():
    runs = _runs(Cadence.monthly, _ny(2026, 1, 5, 9, 0), {"week": 1, "weekday": "mon"}, 3)
    assert [(r.month, r.day, r.strftime("%a")) for r in runs] == [(2, 2, "Mon"), (3, 2, "Mon"), (4, 6, "Mon")]


def test_monthly_on_the_last_friday():
    runs = _runs(Cadence.monthly, _ny(2026, 1, 30, 17, 0), {"week": -1, "weekday": "fri"}, 3)
    assert [(r.month, r.day) for r in runs] == [(2, 27), (3, 27), (4, 24)]


def test_missed_monthly_slots_are_walked_by_the_rule():
    start = _ny(2026, 1, 1, 9, 0)
    now = _ny(2026, 4, 2, 9, 0)
    slots = missed_slots(
        Cadence.monthly, start, NY, now=now, lookback=timedelta(days=7), recurrence={"month_day": 1},
    )
    assert slots.run == _ny(2026, 4, 1, 9, 0)
    assert slots.skipped == 3
    assert slots.future == _ny(2026, 5, 1, 9, 0)


def test_next_future_occurrence_follows_the_rule():
    nxt = next_future_occurrence(
        Cadence.weekly, _ny(2025, 11, 3, 9, 0), NY, now=_ny(2025, 11, 4, 12, 0), recurrence={"days": ["mon", "thu"]},
    )
    assert nxt == _ny(2025, 11, 6, 9, 0)


def test_align_moves_a_start_to_the_first_day_the_rule_allows():
    # Tue 4 Nov: the first of Mon/Wed/Fri after it is Wed 5 Nov.
    assert align_to_rule(Cadence.weekly, _ny(2025, 11, 4, 9, 0), NY, {"days": ["mon", "wed", "fri"]}) == _ny(
        2025, 11, 5, 9, 0,
    )
    on_a_matching_day = _ny(2025, 11, 5, 9, 0)
    assert align_to_rule(Cadence.weekly, on_a_matching_day, NY, {"days": ["wed"]}) == on_a_matching_day
    assert align_to_rule(Cadence.daily, _ny(2025, 11, 4, 9, 0), NY) == _ny(2025, 11, 4, 9, 0)


# --- the rule's shape -----------------------------------------------------


def test_a_rule_takes_exactly_one_form():
    assert Recurrence.model_validate({"days": ["fri", "mon", "fri"]}).days == ["mon", "fri"]
    assert Recurrence.model_validate({"monthDay": 15}).stored() == {"month_day": 15}
    assert Recurrence.model_validate({"week": -1, "weekday": "fri"}).stored() == {"week": -1, "weekday": "fri"}
    for bad in ({}, {"days": []}, {"monthDay": 32}, {"week": 1}, {"week": 5, "weekday": "mon"},
                {"days": ["mon"], "monthDay": 1}, {"days": ["funday"]}):
        with pytest.raises(ValidationError):
            Recurrence.model_validate(bad)


def test_a_rule_must_fit_its_cadence():
    base = {"title": "t", "prompt": "p", "nextRunAt": "2030-01-01T09:00:00Z"}
    ScheduleCreateRequest.model_validate({**base, "cadence": "weekly", "recurrence": {"days": ["mon"]}})
    ScheduleCreateRequest.model_validate({**base, "cadence": "monthly", "recurrence": {"monthDay": 1}})
    for cadence, recurrence in (("daily", {"days": ["mon"]}), ("weekly", {"monthDay": 1}), ("monthly", {"days": ["mon"]})):
        with pytest.raises(ValidationError):
            ScheduleCreateRequest.model_validate({**base, "cadence": cadence, "recurrence": recurrence})


def test_cards_read_the_rule_back():
    at = _ny(2026, 1, 5, 9, 0)
    assert describe_cadence("weekly", at, NY, {"days": ["mon", "wed", "fri"]}) == (
        f"Every Monday, Wednesday and Friday at 09:00 ({NY})"
    )
    assert describe_cadence("monthly", at, NY, {"week": 1, "weekday": "mon"}) == (
        f"The first Monday of every month at 09:00 ({NY})"
    )
    assert describe_cadence("monthly", at, NY, {"month_day": 22}) == f"The 22nd of every month at 09:00 ({NY})"
    assert describe_cadence("monthly", at, NY, {"month_day": 31}) == (
        f"The 31st of every month, or the last day of a shorter month, at 09:00 ({NY})"
    )
    assert describe_cadence("weekly", at, NY) == f"Every Monday at 09:00 ({NY})"


# --- saving ---------------------------------------------------------------


def _create(**overrides):
    fields = dict(
        title="rule test", prompt="do the thing", cadence="daily", next_run_at=_ny(2030, 1, 1, 9, 0),
        model="default", timezone=NY, project_id=GENERAL_PROJECT_ID, enabled=True,
    )
    fields.update(overrides)
    with unit_session(scope=SYSTEM_SCOPE) as session:
        schedule = ScheduleService(session).create_schedule(**fields)
        return schedule.id, schedule.recurrence, schedule.next_run_at


def _delete(schedule_id):
    with unit_session(scope=SYSTEM_SCOPE) as session:
        ScheduleService(session).delete_schedule(schedule_id)


def test_a_monthly_task_saved_without_a_rule_keeps_its_first_runs_day():
    # Tue 1 Jan 2030.
    schedule_id, recurrence, first = _create(cadence="monthly", next_run_at=_ny(2030, 1, 31, 9, 0))
    try:
        assert recurrence == {"month_day": 31}
        assert first.astimezone(ZoneInfo(NY)).day == 31
    finally:
        _delete(schedule_id)


def test_a_save_moves_the_first_run_onto_the_rule():
    # Tue 1 Jan 2030; the first Friday after it is 4 Jan.
    schedule_id, recurrence, first = _create(cadence="weekly", recurrence={"days": ["fri"]})
    try:
        assert recurrence == {"days": ["fri"]}
        assert first.astimezone(ZoneInfo(NY)).date().isoformat() == "2030-01-04"
    finally:
        _delete(schedule_id)


async def test_the_api_saves_edits_and_clears_a_rule():
    due = datetime.now(timezone.utc) + timedelta(days=1)
    with turn_pool._a_daily_schedule(due=due) as schedule_id:
        url = f"/api/v1/schedules/{schedule_id}"
        async with turn_pool._client() as client:
            weekly = await client.patch(url, json={"cadence": "weekly", "recurrence": {"days": ["tue", "thu"]}})
            assert weekly.status_code == 200, weekly.text
            assert weekly.json()["recurrence"]["days"] == ["tue", "thu"]
            assert datetime.fromisoformat(weekly.json()["nextRunAt"]).weekday() in (1, 3)

            renamed = await client.patch(url, json={"title": "Renamed"})
            assert renamed.json()["recurrence"]["days"] == ["tue", "thu"]

            wrong = await client.patch(url, json={"recurrence": {"monthDay": 3}})
            assert wrong.status_code == 422, wrong.text

            monthly = await client.patch(url, json={"cadence": "monthly", "recurrence": {"week": -1, "weekday": "fri"}})
            assert monthly.status_code == 200, monthly.text
            assert monthly.json()["recurrence"]["week"] == -1
            assert monthly.json()["recurrence"]["weekday"] == "fri"

            incompatible = await client.patch(url, json={"cadence": "daily", "recurrence": {"days": ["mon"]}})
            assert incompatible.status_code == 422, incompatible.text
            assert "A daily task" in incompatible.json()["detail"]
            assert (await client.get(url)).json()["cadence"] == "monthly"

            daily = await client.patch(url, json={"cadence": "daily"})
            assert daily.status_code == 200, daily.text
            assert daily.json()["recurrence"] is None


@pytest.mark.parametrize("dialect", ["sqlite", pytest.param("postgresql", marks=pytest.mark.postgres_service)])
def test_downgrade_pauses_monthly_tasks_and_preserves_weekly_tasks(dialect, monkeypatch):
    uri = "sqlite://"
    if dialect == "postgresql":
        uri = os.environ.get("COWORK_TEST_POSTGRES_URI")
        if not uri:
            missing_prerequisite("set COWORK_TEST_POSTGRES_URI to a disposable database",
                                 required_by="COWORK_REQUIRE_SERVICES")
    migration = importlib.import_module("cowork.db.alembic.versions.a7c3e9f1b5d2_schedule_recurrence")
    engine = sa.create_engine(uri)
    schedules = sa.Table(
        "schedules", sa.MetaData(),
        sa.Column("cadence", sa.String, primary_key=True),
        sa.Column("recurrence", sa.JSON),
        sa.Column("enabled", sa.Boolean),
        sa.Column("paused_at", sa.DateTime(timezone=True)),
        prefixes=["TEMPORARY"],
    )
    try:
        with engine.begin() as connection:
            schedules.create(connection)
            connection.execute(schedules.insert(), [
                {"cadence": "monthly", "recurrence": {"month_day": 31}, "enabled": True},
                {"cadence": "weekly", "recurrence": {"days": ["mon", "fri"]}, "enabled": True},
            ])
            monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(connection)))
            migration.downgrade()

            assert "recurrence" not in {c["name"] for c in sa.inspect(connection).get_columns("schedules")}
            rows = {
                row.cadence: (row.enabled, row.paused_at is not None)
                for row in connection.execute(sa.select(schedules.c.cadence, schedules.c.enabled, schedules.c.paused_at))
            }
            assert rows == {"monthly": (False, True), "weekly": (True, False)}
    finally:
        engine.dispose()


# --- the scheduler, catch-up and resume -----------------------------------
#
# Each path below must pass the task's rule along. A rule for every day of the
# week steps a day at a time, so the expected slot does not depend on the day
# the test runs, while a weekly task stepped without its rule moves a week.

EVERY_DAY = {"days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]}


def _just_passed() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=1)


def test_after_a_run_the_scheduler_steps_to_the_rules_next_day():
    slot = _just_passed()
    schedule = Schedule(
        title="rule test", prompt="do the thing", cadence="weekly", recurrence=EVERY_DAY,
        next_run_at=slot, model="default", timezone="UTC",
    )

    scheduler._advance_next_run_at(schedule, SimpleNamespace(add=lambda _row: None))

    assert schedule.next_run_at == slot + timedelta(days=1)


def test_catch_up_walks_missed_slots_by_the_rule():
    slot = _just_passed()
    schedule_id, _, _ = _create(
        cadence="weekly", recurrence=EVERY_DAY, next_run_at=slot - timedelta(days=3), timezone="UTC",
    )
    try:
        with unit_session(scope=SYSTEM_SCOPE) as session:
            schedule = ScheduleService(session).get_schedule(schedule_id)
            scheduler._catch_up(session, schedule, slot + timedelta(seconds=30))
        with unit_session(scope=SYSTEM_SCOPE) as session:
            schedule = ScheduleService(session).get_schedule(schedule_id)
            # The latest of the four overdue daily slots runs; the three before it are passed over.
            assert ensure_utc(schedule.next_run_at) == slot
            assert schedule.missed_runs == 3
    finally:
        _delete(schedule_id)


def test_resume_waits_for_the_rules_next_day():
    slot = _just_passed()
    schedule_id, _, _ = _create(
        cadence="weekly", recurrence=EVERY_DAY, next_run_at=slot - timedelta(days=3), timezone="UTC",
    )
    try:
        with unit_session(scope=SYSTEM_SCOPE) as session:
            next_run, runs_now = ScheduleService(session).resume_plan(schedule_id)
        assert not runs_now
        assert next_run == slot + timedelta(days=1)
    finally:
        _delete(schedule_id)
