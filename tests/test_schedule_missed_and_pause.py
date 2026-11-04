"""Missed runs, explicit pause, and auto-pause after repeated failures.

A recurring schedule that fell behind runs its latest missed slot once, if
that slot is no more than seven days old, and the slots it passed over are
counted. A one-off runs up to an hour late; from an hour on it is marked
missed and runs only on Run now, which uses a one-off up. Pause stops what
would start on its own, and resume clears the pause and the failure streak.
A resumed schedule waits for its next slot, unless a run gave its due slot
back before its agent acted and no later slot has come due: that slot runs
once more. Three cron slots in a row that settle failed pause the schedule;
a success resets the count.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import create_engine, inspect, text

from cowork import scheduler
from cowork.common.datetime_utils import ensure_utc
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.migrations import run_schema_migrations
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
from cowork.db.session import get_open_session
from cowork.models.schedule import Schedule, ScheduleRun
from cowork.schedule_timing import FIXED_INTERVALS, advance_occurrence, missed_slots
from cowork.schemas.schedules import Cadence, RunStatus, ScheduleState, schedule_state
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.schedules import MAX_FAILED_SLOTS, ScheduleConflict, ScheduleRunService, ScheduleService

NOW = datetime.now(timezone.utc).replace(microsecond=0)
# A slot that came due a minute ago, so the missed-run pass leaves it due.
SLOT = NOW - timedelta(minutes=1)


@pytest.fixture(autouse=True)
def _now_at_test_start():
    """Take NOW and SLOT when each test starts. The scheduler reads the real
    clock, so a NOW taken once at import drifts by however long the rest of
    the suite ran first: in a full run, a one-off "50 minutes late" became
    more than an hour late and counted as missed."""
    global NOW, SLOT
    NOW = datetime.now(timezone.utc).replace(microsecond=0)
    SLOT = NOW - timedelta(minutes=1)


@pytest.fixture(autouse=True)
def _fresh_scheduler(monkeypatch):
    monkeypatch.setattr(scheduler, "_scheduler_task", None)
    monkeypatch.setattr(scheduler, "_scheduled_runs", set())
    monkeypatch.setattr(scheduler, "_admission", scheduler._Admission(scheduler._MAX_CONCURRENT_RUNS))


def _make_schedule(*, cadence: str = "daily", next_run_at: datetime | None = None, enabled: bool = True) -> UUID:
    next_run_at = SLOT if next_run_at is None else next_run_at
    session = get_open_session()
    try:
        return ScheduleService(ScopedSession(session, SYSTEM_SCOPE)).create_schedule(
            title=f"missed and pause {uuid4()}", prompt="do the thing", cadence=cadence,
            next_run_at=next_run_at, model="default", timezone="UTC", project_id=GENERAL_PROJECT_ID,
            enabled=enabled,
        ).id
    finally:
        session.close()


def _delete_schedule(schedule_id: UUID) -> None:
    from sqlmodel import select

    from cowork.models.conversation import Conversation
    from cowork.services.conversations import ConversationService

    session = get_open_session()
    try:
        scoped = ScopedSession(session, SYSTEM_SCOPE)
        row = session.get(Schedule, schedule_id)
        if row is None:
            return
        for conversation_id in session.exec(select(Conversation.id).where(Conversation.topic == row.title)).all():
            ConversationService(scoped).delete_conversation(conversation_id)
        ScheduleService(scoped).delete_schedule(schedule_id)
    finally:
        session.close()


@pytest.fixture
def made():
    """Schedules a test makes, removed afterwards with their runs."""
    ids: list[UUID] = []

    def make(**kwargs) -> UUID:
        ids.append(_make_schedule(**kwargs))
        return ids[-1]

    yield make
    for schedule_id in ids:
        _delete_schedule(schedule_id)


@pytest.fixture
def turns(monkeypatch):
    """A turn handler that counts the turns it starts and ends each at once,
    and a switch for what the finished turn's records show."""
    import cowork.handlers.responses as responses_mod

    started: list[str] = []
    ending = SimpleNamespace(reason="completed", acted=True)
    during: list = []

    class CountingHandler:
        def __init__(self, *, principal=None, interactive=True):
            pass

        async def handle(self, request):
            started.append(request.conversation)
            for step in during:
                step()

            async def stream():
                if False:
                    yield

            return stream()

    async def evidence(_conversation_id):
        return scheduler._TurnEvidence(reason=ending.reason, acted=ending.acted)

    monkeypatch.setattr(responses_mod, "ResponsesHandler", CountingHandler)
    monkeypatch.setattr(scheduler, "_turn_evidence", evidence)
    return SimpleNamespace(started=started, ending=ending, during=during)


def _schedule(schedule_id: UUID) -> Schedule:
    session = get_open_session()
    try:
        return ScheduleService(ScopedSession(session, SYSTEM_SCOPE)).get_schedule(schedule_id)
    finally:
        session.close()


def _runs(schedule_id: UUID) -> list[ScheduleRun]:
    session = get_open_session()
    try:
        return list(reversed(ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE)).list_runs(schedule_id)))
    finally:
        session.close()


def _set(schedule_id: UUID, **values) -> None:
    session = get_open_session()
    try:
        row = session.get(Schedule, schedule_id)
        for key, value in values.items():
            setattr(row, key, value)
        session.add(row)
        session.commit()
    finally:
        session.close()


def _poll(schedule_id: UUID) -> list[scheduler.DueSlot]:
    """One poll's due slots for this schedule; other tests share the database."""
    session = get_open_session()
    try:
        return [s for s in scheduler._poll(ScopedSession(session, SYSTEM_SCOPE)) if s.schedule_id == schedule_id]
    finally:
        session.close()


def _service(call):
    session = get_open_session()
    try:
        return call(ScheduleService(ScopedSession(session, SYSTEM_SCOPE)))
    finally:
        session.close()


async def _run(slot: scheduler.DueSlot) -> None:
    await scheduler.execute_schedule(slot.schedule_id, slot_at=slot.slot_at)


def _client() -> httpx.AsyncClient:
    from cowork.server import create_app

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test")


# --- Which missed slot runs -------------------------------------------------


def _walked(cadence: str, next_run_at: datetime, *, now: datetime, lookback: timedelta):
    """missed_slots worked out by visiting every slot, the reference the
    counted fast path must agree with."""
    slots = []
    cursor = next_run_at
    while cursor <= now:
        slots.append(cursor)
        cursor = advance_occurrence(cadence, cursor, "UTC")
    if slots and slots[-1] >= now - lookback:
        return slots[-1], len(slots) - 1, cursor
    return None, len(slots), cursor


def test_the_latest_missed_slot_inside_the_lookback_runs_and_the_rest_are_passed_over():
    start = NOW - timedelta(days=3, hours=2)
    missed = missed_slots(Cadence.daily, start, "UTC", now=NOW, lookback=timedelta(days=7))
    assert missed.run == start + timedelta(days=3)
    assert missed.skipped == 3
    assert missed.future == start + timedelta(days=4)


def test_no_missed_slot_runs_when_every_one_is_older_than_the_lookback():
    start = NOW - timedelta(days=30)
    missed = missed_slots(Cadence.weekly, start, "UTC", now=NOW - timedelta(days=10), lookback=timedelta(days=5))
    assert missed.run is None
    assert missed.skipped == 3


@pytest.mark.parametrize("cadence", sorted(FIXED_INTERVALS, key=str))
def test_counting_fixed_interval_slots_agrees_with_walking_them(cadence):
    rng = random.Random(7)
    lookback = timedelta(days=7)
    for _ in range(40):
        behind = timedelta(minutes=rng.randint(1, 60 * 24 * 40))
        start = (NOW - behind).replace(second=0)
        missed = missed_slots(cadence, start, "UTC", now=NOW, lookback=lookback)
        assert (missed.run, missed.skipped, missed.future) == _walked(
            cadence, start, now=NOW, lookback=lookback,
        ), behind


# --- Catching up after the app was not running --------------------------------


def test_a_daily_schedule_days_behind_runs_its_latest_slot_once_and_counts_the_rest(made, turns):
    schedule_id = made(next_run_at=SLOT - timedelta(days=3))

    (due,) = _poll(schedule_id)

    assert due.slot_at == SLOT
    schedule = _schedule(schedule_id)
    assert ensure_utc(schedule.next_run_at) == SLOT
    assert schedule.missed_runs == 3


def test_an_hourly_schedule_a_month_behind_runs_its_latest_slot_once(made, turns):
    start = SLOT - timedelta(days=30)
    schedule_id = made(cadence="hourly", next_run_at=start)

    (due,) = _poll(schedule_id)

    assert due.slot_at == SLOT
    assert _schedule(schedule_id).missed_runs == 30 * 24


def test_a_single_recent_missed_slot_just_runs(made, turns):
    schedule_id = made(next_run_at=SLOT)

    (due,) = _poll(schedule_id)

    assert due.slot_at == SLOT
    assert _schedule(schedule_id).missed_runs == 0


def test_catching_up_twice_counts_the_missed_slots_once(made, turns):
    schedule_id = made(next_run_at=SLOT - timedelta(days=2))
    _poll(schedule_id)
    _poll(schedule_id)

    assert _schedule(schedule_id).missed_runs == 2


def test_a_schedule_with_a_run_going_is_not_caught_up(made, turns):
    schedule_id = made(next_run_at=SLOT - timedelta(days=2))
    session = get_open_session()
    try:
        ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE)).create_run(schedule_id, lease_seconds=60)
    finally:
        session.close()

    assert _poll(schedule_id) == []
    assert ensure_utc(_schedule(schedule_id).next_run_at) == SLOT - timedelta(days=2)


# --- One-offs ---------------------------------------------------------------


def test_a_one_off_less_than_an_hour_late_runs(made, turns):
    schedule_id = made(cadence="once", next_run_at=NOW - timedelta(minutes=50))
    (due,) = _poll(schedule_id)
    assert due.slot_at == NOW - timedelta(minutes=50)


def test_a_one_off_an_hour_late_or_more_is_missed_and_does_not_run(made, turns):
    schedule_id = made(cadence="once", next_run_at=NOW - timedelta(minutes=61))

    assert _poll(schedule_id) == []

    schedule = _schedule(schedule_id)
    assert schedule_state(schedule) == ScheduleState.missed
    assert schedule.enabled is False and schedule.missed_at is not None
    assert schedule.missed_runs == 1


async def test_run_now_uses_up_a_missed_one_off(made, turns):
    schedule_id = made(cadence="once", next_run_at=NOW - timedelta(hours=2))
    _poll(schedule_id)
    assert schedule_state(_schedule(schedule_id)) == ScheduleState.missed

    await scheduler.execute_schedule(schedule_id, is_manual=True)

    schedule = _schedule(schedule_id)
    assert schedule_state(schedule) == ScheduleState.done
    assert schedule.missed_at is None
    assert len(turns.started) == 1


async def test_run_now_uses_up_a_one_off_that_has_not_come_due(made, turns):
    schedule_id = made(cadence="once", next_run_at=NOW + timedelta(days=1))

    await scheduler.execute_schedule(schedule_id, is_manual=True)

    assert schedule_state(_schedule(schedule_id)) == ScheduleState.done


def test_a_missed_one_off_given_a_new_time_is_scheduled_again(made, turns):
    schedule_id = made(cadence="once", next_run_at=NOW - timedelta(hours=2))
    _poll(schedule_id)

    _service(lambda s: s.update_schedule(schedule_id, next_run_at=NOW + timedelta(hours=1), enabled=False))

    schedule = _schedule(schedule_id)
    assert schedule_state(schedule) == ScheduleState.active
    assert schedule.enabled is True and schedule.missed_at is None


# --- Pause and resume -------------------------------------------------------


def test_pause_records_when_and_resume_waits_for_the_next_slot(made, turns):
    schedule_id = made(next_run_at=SLOT)

    _service(lambda s: s.pause_schedule(schedule_id))
    paused = _schedule(schedule_id)
    assert schedule_state(paused) == ScheduleState.paused
    assert paused.enabled is False and paused.paused_at is not None
    assert _poll(schedule_id) == []

    _service(lambda s: s.resume_schedule(schedule_id))
    resumed = _schedule(schedule_id)
    assert schedule_state(resumed) == ScheduleState.active
    assert resumed.paused_at is None
    assert ensure_utc(resumed.next_run_at) == SLOT + timedelta(days=1), "nothing runs at once"
    assert _poll(schedule_id) == []
    assert turns.started == []


async def test_a_run_going_when_paused_finishes_and_nothing_follows_it_until_resumed(made, turns):
    turns.ending.reason, turns.ending.acted = "error", False
    schedule_id = made(next_run_at=SLOT)
    turns.during.append(lambda: _service(lambda s: s.pause_schedule(schedule_id)))
    (first,) = _poll(schedule_id)

    await _run(first)

    (run,) = _runs(schedule_id)
    assert run.status == RunStatus.failed
    assert _poll(schedule_id) == [], "nothing follows the run while paused"

    turns.during.clear()
    _service(lambda s: s.resume_schedule(schedule_id))
    assert ensure_utc(_schedule(schedule_id).next_run_at) == SLOT + timedelta(days=1), "its turn used the slot up"
    assert len(turns.started) == 1


async def _fail_before_the_turn_while_paused(schedule_id: UUID, monkeypatch) -> None:
    """Run the due slot; the schedule is paused, and the run then fails
    before its turn starts, so it gives the slot back."""
    import cowork.services.product_permissions as permissions

    async def paused_then_refused(*_args, **_kwargs):
        _service(lambda s: s.pause_schedule(schedule_id))
        raise RuntimeError("could not read the task's settings")

    monkeypatch.setattr(permissions, "require_product_permission", paused_then_refused)
    (first,) = _poll(schedule_id)
    await _run(first)


async def test_a_slot_given_back_before_acting_while_paused_runs_once_resumed(made, turns, monkeypatch):
    schedule_id = made(next_run_at=SLOT)
    await _fail_before_the_turn_while_paused(schedule_id, monkeypatch)

    (run,) = _runs(schedule_id)
    assert run.status == RunStatus.failed and run.slot_at is None
    assert _poll(schedule_id) == [], "nothing follows the run while paused"

    _service(lambda s: s.resume_schedule(schedule_id))
    assert _poll(schedule_id) == [scheduler.DueSlot(schedule_id=schedule_id, slot_at=SLOT)], (
        "the slot it left due runs once resumed"
    )
    assert turns.started == []


async def test_a_slot_left_due_is_not_run_once_a_later_slot_came_due_during_the_pause(made, turns, monkeypatch):
    schedule_id = made(next_run_at=SLOT)
    await _fail_before_the_turn_while_paused(schedule_id, monkeypatch)
    # The pause then lasts two days: move the slot it left due back so that
    # two later slots have come due since.
    _set(schedule_id, next_run_at=SLOT - timedelta(days=2))

    _service(lambda s: s.resume_schedule(schedule_id))

    assert ensure_utc(_schedule(schedule_id).next_run_at) == SLOT + timedelta(days=1), (
        "slots that came due while paused are not run, and the stale one goes with them"
    )
    assert _poll(schedule_id) == []


def test_resume_refuses_a_one_off_whose_time_has_passed(made, turns):
    schedule_id = made(cadence="once", next_run_at=NOW + timedelta(hours=1))
    _service(lambda s: s.pause_schedule(schedule_id))
    _set(schedule_id, next_run_at=NOW - timedelta(minutes=5))

    with pytest.raises(ScheduleConflict):
        _service(lambda s: s.resume_schedule(schedule_id))
    assert schedule_state(_schedule(schedule_id)) == ScheduleState.paused


def test_an_edit_that_sends_enabled_pauses_and_resumes_like_the_actions(made, turns):
    schedule_id = made(next_run_at=NOW + timedelta(hours=3))

    _service(lambda s: s.update_schedule(schedule_id, enabled=False, title="renamed"))
    assert schedule_state(_schedule(schedule_id)) == ScheduleState.paused

    _service(lambda s: s.update_schedule(schedule_id, enabled=False, title="renamed again"))
    assert _schedule(schedule_id).title == "renamed again"
    assert schedule_state(_schedule(schedule_id)) == ScheduleState.paused

    _set(schedule_id, consecutive_failures=2)
    _service(lambda s: s.update_schedule(schedule_id, enabled=True))
    resumed = _schedule(schedule_id)
    assert schedule_state(resumed) == ScheduleState.active
    assert resumed.consecutive_failures == 0


def test_a_schedule_created_off_is_paused_not_done(made, turns):
    schedule_id = made(cadence="once", next_run_at=NOW + timedelta(days=1), enabled=False)
    assert schedule_state(_schedule(schedule_id)) == ScheduleState.paused


# --- The failure streak -----------------------------------------------------


async def _fail_a_slot(schedule_id: UUID, turns, *, acted: bool = True) -> None:
    """Move the schedule onto a new slot due now and run it to a failed end."""
    _set(schedule_id, next_run_at=SLOT - timedelta(seconds=len(turns.started) + 1))
    turns.ending.reason, turns.ending.acted = "error", acted
    (due,) = _poll(schedule_id)
    await _run(due)


async def test_three_slots_in_a_row_that_fail_pause_the_schedule(made, turns):
    schedule_id = made(next_run_at=SLOT)

    for n in range(1, MAX_FAILED_SLOTS):
        await _fail_a_slot(schedule_id, turns)
        schedule = _schedule(schedule_id)
        assert schedule.consecutive_failures == n
        assert schedule.auto_paused_at is None

    await _fail_a_slot(schedule_id, turns)

    schedule = _schedule(schedule_id)
    assert schedule.consecutive_failures == MAX_FAILED_SLOTS
    assert schedule_state(schedule) == ScheduleState.auto_paused
    assert schedule.enabled is False
    _set(schedule_id, next_run_at=SLOT - timedelta(seconds=30))
    assert _poll(schedule_id) == []

    _service(lambda s: s.resume_schedule(schedule_id))
    resumed = _schedule(schedule_id)
    assert schedule_state(resumed) == ScheduleState.active
    assert resumed.consecutive_failures == 0 and resumed.auto_paused_at is None


async def test_a_success_resets_the_streak(made, turns):
    schedule_id = made(next_run_at=SLOT)
    await _fail_a_slot(schedule_id, turns)
    await _fail_a_slot(schedule_id, turns)

    _set(schedule_id, next_run_at=SLOT - timedelta(seconds=30))
    turns.ending.reason, turns.ending.acted = "completed", True
    (due,) = _poll(schedule_id)
    await _run(due)

    assert _schedule(schedule_id).consecutive_failures == 0


async def test_a_manual_failure_does_not_count(made, turns):
    schedule_id = made(next_run_at=NOW + timedelta(days=1))
    turns.ending.reason, turns.ending.acted = "error", True

    await scheduler.execute_schedule(schedule_id, is_manual=True)

    assert _schedule(schedule_id).consecutive_failures == 0


def _left_running(schedule_id: UUID, *, linked_conversation: bool) -> None:
    """A run another replica started for SLOT and then died during."""
    from cowork.services.conversations import ConversationService

    session = get_open_session()
    try:
        scoped = ScopedSession(session, SYSTEM_SCOPE)
        run = ScheduleRunService(scoped).create_run(schedule_id, slot_at=SLOT, lease_seconds=-60)
        if linked_conversation:
            conversation = ConversationService(scoped).create_conversation(
                topic=_schedule(schedule_id).title, project_id=GENERAL_PROJECT_ID,
            )
            ScheduleRunService(scoped).set_run_conversation(run.id, conversation.id)
    finally:
        session.close()


def test_a_reaped_run_whose_turn_may_have_acted_counts_as_a_failed_slot(made, turns):
    schedule_id = made(next_run_at=SLOT)
    _left_running(schedule_id, linked_conversation=True)

    _poll(schedule_id)

    assert _schedule(schedule_id).consecutive_failures == 1


def test_a_reaped_run_that_never_started_does_not_count(made, turns):
    schedule_id = made(next_run_at=SLOT)
    _left_running(schedule_id, linked_conversation=False)

    _poll(schedule_id)

    assert _schedule(schedule_id).consecutive_failures == 0


async def test_a_shutdown_before_the_turn_does_not_count(made, turns, monkeypatch):
    from cowork.streaming.registry import ServerShuttingDown
    import cowork.handlers.responses as responses_mod

    class RefusingHandler:
        def __init__(self, *, principal=None, interactive=True):
            pass

        async def handle(self, request):
            raise ServerShuttingDown()

    monkeypatch.setattr(responses_mod, "ResponsesHandler", RefusingHandler)
    schedule_id = made(next_run_at=SLOT)
    (due,) = _poll(schedule_id)

    await _run(due)

    assert _schedule(schedule_id).consecutive_failures == 0


# --- The API ----------------------------------------------------------------


async def test_the_api_reports_state_and_the_pause_actions_change_it(made, turns):
    schedule_id = made(next_run_at=NOW + timedelta(hours=3))

    async with _client() as client:
        shown = (await client.get(f"/api/v1/schedules/{schedule_id}")).json()
        assert (shown["state"], shown["pausedAt"], shown["consecutiveFailures"]) == ("active", None, 0)
        paused = (await client.post(f"/api/v1/schedules/{schedule_id}/pause")).json()
        assert paused["state"] == "paused" and paused["pausedAt"] is not None
        resumed = (await client.post(f"/api/v1/schedules/{schedule_id}/resume")).json()
        assert resumed["state"] == "active"


async def test_the_api_reports_an_auto_pause(made, turns):
    schedule_id = made(next_run_at=NOW + timedelta(hours=3))
    _set(schedule_id, enabled=False, auto_paused_at=NOW, consecutive_failures=3)

    async with _client() as client:
        shown = (await client.get(f"/api/v1/schedules/{schedule_id}")).json()

    assert shown["state"] == "auto_paused"
    assert shown["autoPausedAt"] is not None
    assert shown["consecutiveFailures"] == 3


async def test_resuming_a_one_off_whose_time_passed_answers_409(made, turns):
    schedule_id = made(cadence="once", next_run_at=NOW + timedelta(hours=1))
    _service(lambda s: s.pause_schedule(schedule_id))
    _set(schedule_id, next_run_at=NOW - timedelta(minutes=5))

    async with _client() as client:
        answer = await client.post(f"/api/v1/schedules/{schedule_id}/resume")

    assert answer.status_code == 409
    assert "time has passed" in answer.json()["detail"]


# --- Migration --------------------------------------------------------------


def test_the_pause_migration_upgrades_sorts_disabled_schedules_and_downgrades(tmp_path, monkeypatch):
    from alembic import command

    from cowork.db.migrations import _alembic_config

    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    uri = f"sqlite:///{tmp_path / 'pause.db'}"
    engine = create_engine(uri)
    run_schema_migrations(engine, uri)
    config = _alembic_config(uri)

    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.downgrade(config, "5b7c2d9e4f10")
    columns = {c["name"] for c in inspect(engine).get_columns("schedules")}
    assert not {"consecutive_failures", "auto_paused_at", "paused_at", "missed_at"} & columns

    project_id = GENERAL_PROJECT_ID.hex
    rows = {
        "recurring_off": ("daily", 0, 0),
        "one_off_missed": ("once", 0, 1),
        "one_off_paused": ("once", 0, 0),
        "one_off_on": ("once", 1, 0),
    }
    ids = {}
    with engine.begin() as connection:
        connection.execute(
            text("INSERT OR IGNORE INTO projects (id, name, path) VALUES (:id, 'general', '/general')"),
            {"id": project_id},
        )
        for name, (cadence, enabled, missed) in rows.items():
            ids[name] = uuid4().hex
            connection.execute(
                text(
                    "INSERT INTO schedules (id, title, prompt, cadence, timezone, next_run_at, enabled, "
                    "project_id, model, missed_runs) VALUES (:id, :t, 'p', :c, 'UTC', '2026-01-01 00:00:00', "
                    ":e, :p, 'default', :m)"
                ),
                {"id": ids[name], "t": name, "c": cadence, "e": enabled, "p": project_id, "m": missed},
            )

    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")

    with engine.connect() as connection:
        got = {
            row.title: (row.paused_at is not None, row.missed_at is not None, row.consecutive_failures)
            for row in connection.execute(
                text("SELECT title, paused_at, missed_at, consecutive_failures FROM schedules")
            )
        }
    assert got["recurring_off"] == (True, False, 0)
    assert got["one_off_missed"] == (False, True, 0)
    assert got["one_off_paused"] == (True, False, 0)
    assert got["one_off_on"] == (False, False, 0)
