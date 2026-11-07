"""A trial run: Run now with ``trial=true``, which tries a task out.

The desktop's "Save and run once" starts one right after creating a task. It
must not cost the task its schedule: a one-off stays scheduled for its own
time, the slot after it still runs (the freshness guard ignores it), and the
failure streak neither grows nor resets. Without the flag, Run now behaves as
before.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest

from sqlalchemy import create_engine, inspect

from cowork import scheduler
from cowork.common.datetime_utils import ensure_utc
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.migrations import run_schema_migrations
from cowork.schemas.schedules import RunStatus
from cowork.services.schedules import ScheduleRunService, ScheduleService
from tests import test_schedule_run_failures as failures

from tests.test_schedule_run_failures import (  # noqa: F401  (fixtures)
    _client,
    _delete_schedule,
    _fresh_scheduler,
    _make_schedule as _original_make_schedule,
    _poll,
    _runs,
    _schedule,
    _scoped,
)


turns = failures.turns


@pytest.fixture(autouse=True)
def fresh_slot():
    global SLOT
    SLOT = (datetime.now(timezone.utc) - timedelta(minutes=1)).replace(microsecond=0)


def _make_schedule(**kwargs):
    kwargs.setdefault("next_run_at", SLOT)
    return _original_make_schedule(**kwargs)


@pytest.fixture
def schedule_id():
    created = _make_schedule()
    yield created
    _delete_schedule(created)


def _set_streak(schedule_id, count: int) -> None:
    session, scoped = _scoped()
    try:
        schedule = ScheduleService(scoped).get_schedule(schedule_id)
        schedule.consecutive_failures = count
        session.add(schedule)
        session.commit()
    finally:
        session.close()


# --- One-offs ------------------------------------------------------------------


async def test_a_trial_run_leaves_a_one_off_scheduled_for_its_own_time(turns):
    one_off = _make_schedule(cadence="once", next_run_at=SLOT)
    try:
        await scheduler.execute_schedule(one_off, is_manual=True, trial=True)

        schedule = _schedule(one_off)
        assert schedule.enabled
        assert ensure_utc(schedule.next_run_at) == SLOT
        assert schedule.missed_at is None and schedule.paused_at is None
        (run,) = _runs(one_off)
        assert run.trial and run.status == RunStatus.success
        assert [slot.slot_at for slot in _poll(one_off)] == [SLOT], "its own time still fires"
    finally:
        _delete_schedule(one_off)


async def test_a_plain_run_now_still_uses_a_one_off_up(turns):
    one_off = _make_schedule(cadence="once", next_run_at=SLOT)
    try:
        await scheduler.execute_schedule(one_off, is_manual=True)

        assert not _schedule(one_off).enabled
        assert _poll(one_off) == []
    finally:
        _delete_schedule(one_off)


# --- The freshness guard -------------------------------------------------------


async def test_a_trial_success_does_not_suppress_the_next_slot(schedule_id, turns):
    await scheduler.execute_schedule(schedule_id, is_manual=True, trial=True)

    assert [slot.slot_at for slot in _poll(schedule_id)] == [SLOT]


async def test_a_plain_run_now_success_still_suppresses_the_next_slot(schedule_id, turns):
    await scheduler.execute_schedule(schedule_id, is_manual=True)

    assert _poll(schedule_id) == []


# --- The failure streak --------------------------------------------------------


async def test_a_trial_success_does_not_end_a_failure_streak(schedule_id, turns):
    _set_streak(schedule_id, 2)
    await scheduler.execute_schedule(schedule_id, is_manual=True, trial=True)

    assert _schedule(schedule_id).consecutive_failures == 2


async def test_a_trial_failure_does_not_extend_a_failure_streak(schedule_id, turns):
    _set_streak(schedule_id, 2)
    turns.ending.reason, turns.ending.acted = "error", True
    await scheduler.execute_schedule(schedule_id, is_manual=True, trial=True)

    (run,) = _runs(schedule_id)
    assert run.status == RunStatus.failed
    schedule = _schedule(schedule_id)
    assert schedule.consecutive_failures == 2
    assert schedule.auto_paused_at is None


async def test_a_plain_run_now_success_still_ends_a_failure_streak(schedule_id, turns):
    _set_streak(schedule_id, 2)
    await scheduler.execute_schedule(schedule_id, is_manual=True)

    assert _schedule(schedule_id).consecutive_failures == 0


# --- The API -------------------------------------------------------------------


async def test_run_now_passes_trial_through_and_the_run_reports_it(schedule_id, turns, monkeypatch):
    started: list[dict] = []

    async def capture(schedule_id, **kwargs):
        started.append(kwargs)
        scheduler.release_run_place()

    monkeypatch.setattr(scheduler, "run_admitted", capture)

    def end_claimed_run() -> None:
        # The press claimed its run row before answering; end it so the next
        # press can claim the schedule.
        session, scoped = _scoped()
        try:
            ScheduleRunService(scoped).finish_run(started[-1]["run_id"])
        finally:
            session.close()

    async with _client() as client:
        assert (await client.post(f"/api/v1/schedules/{schedule_id}/run-now?trial=true")).status_code == 202
        end_claimed_run()
        assert (await client.post(f"/api/v1/schedules/{schedule_id}/run-now")).status_code == 202
        end_claimed_run()
        runs = (await client.get(f"/api/v1/schedules/{schedule_id}/runs")).json()["runs"]
    assert [kwargs["trial"] for kwargs in started] == [True, False]
    assert sorted(run["trial"] for run in runs) == [False, True], "the claimed run row carries the flag"
    assert {(run["trial"], run["trigger"]) for run in runs} == {(False, "manual"), (True, "trial")}


async def test_a_cron_run_is_never_a_trial(schedule_id, turns):
    (slot,) = _poll(schedule_id)
    with pytest.raises(ValueError, match="cron run cannot be a trial"):
        await scheduler.execute_schedule(slot.schedule_id, slot_at=slot.slot_at, trial=True)
    assert _runs(schedule_id) == []


# --- Migration -----------------------------------------------------------------


def test_the_trial_migration_upgrades_and_downgrades(tmp_path, monkeypatch):
    from alembic import command

    from cowork.db.migrations import _alembic_config

    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    uri = f"sqlite:///{tmp_path / 'trial.db'}"
    engine = create_engine(uri)
    run_schema_migrations(engine, uri)
    assert "trial" in {c["name"] for c in inspect(engine).get_columns("schedule_runs")}

    config = _alembic_config(uri)
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.downgrade(config, "8c3e1f6a2b47")
    assert "trial" not in {c["name"] for c in inspect(engine).get_columns("schedule_runs")}

    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
    assert "trial" in {c["name"] for c in inspect(engine).get_columns("schedule_runs")}
