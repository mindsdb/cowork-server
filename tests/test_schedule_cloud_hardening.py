"""Scheduled runs on a hosted deployment, where several replicas share one
database and each turn runs in a scratchpad pod.

A replica that reaps a run whose process died also asks that run's pod to
stop, and a run that finds another replica reaped it while its turn went on
stops the turn and leaves the run as the reaper recorded it. One
organization has at most three scheduled runs going at once across every
replica. The run's time limit is a deployment setting.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import ValidationError

from cowork import scheduler
from cowork.common.datetime_utils import ensure_utc
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
from cowork.db.session import get_open_session
from cowork.models.schedule import Schedule, ScheduleRun
from cowork.schemas.schedules import RunStatus
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.schedules import AbandonedTurn, OrgRunsAtLimit, ScheduleRunService, ScheduleService

# A slot that came due a minute ago, so the poll's missed-run pass leaves it due.
SLOT = (datetime.now(timezone.utc) - timedelta(minutes=1)).replace(microsecond=0)


@pytest.fixture(autouse=True)
def _fresh_scheduler(monkeypatch):
    from cowork.common.settings.app_settings import get_app_settings
    monkeypatch.setenv("COWORK_SCHEDULED_RUNS_PER_ORG", "3")
    get_app_settings.cache_clear()
    monkeypatch.setattr(scheduler, "_active_run_ids", set())
    monkeypatch.setattr(scheduler, "_scheduler_task", None)
    monkeypatch.setattr(scheduler, "_scheduled_runs", set())
    monkeypatch.setattr(scheduler, "_admission", scheduler._Admission(scheduler._runs_per_process))
    yield
    get_app_settings.cache_clear()


def _make_schedule(*, org_id: str | None = None) -> UUID:
    session = get_open_session()
    try:
        schedule = ScheduleService(ScopedSession(session, SYSTEM_SCOPE)).create_schedule(
            title=f"cloud {uuid4()}", prompt="do the thing", cadence="daily", next_run_at=SLOT,
            model="default", timezone="UTC", project_id=GENERAL_PROJECT_ID, enabled=True,
        )
        if org_id is not None:
            row = session.get(Schedule, schedule.id)
            row.org_id = org_id
            session.add(row)
            session.commit()
        return schedule.id
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
def schedules():
    made: list[UUID] = []

    def make(org_id: str | None = None) -> UUID:
        made.append(_make_schedule(org_id=org_id))
        return made[-1]

    yield make
    for schedule_id in made:
        _delete_schedule(schedule_id)


def _runs(schedule_id: UUID) -> list[ScheduleRun]:
    session = get_open_session()
    try:
        return list(reversed(ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE)).list_runs(schedule_id)))
    finally:
        session.close()


def _schedule(schedule_id: UUID) -> Schedule:
    session = get_open_session()
    try:
        return ScheduleService(ScopedSession(session, SYSTEM_SCOPE)).get_schedule(schedule_id)
    finally:
        session.close()


def _running(schedule_id: UUID, **kwargs) -> ScheduleRun:
    """A run going on some replica, with a lease well in the future."""
    session = get_open_session()
    try:
        return ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE)).create_run(
            schedule_id, lease_seconds=3600, **kwargs,
        )
    finally:
        session.close()


def _poll(abandoned: list | None = None) -> list[scheduler.DueSlot]:
    session = get_open_session()
    try:
        return scheduler._poll(ScopedSession(session, SYSTEM_SCOPE), abandoned)
    finally:
        session.close()


@pytest.fixture
def turns(monkeypatch):
    """A turn handler whose turns go on until released, and that end
    completed."""
    import cowork.handlers.responses as responses_mod

    started: list[str] = []
    release = asyncio.Event()

    class HeldHandler:
        def __init__(self, *, principal=None, interactive=True):
            pass

        async def handle(self, request):
            started.append(request.conversation)

            async def stream():
                await release.wait()
                if False:
                    yield

            return stream()

    async def evidence(_conversation_id):
        return scheduler._TurnEvidence(reason="completed", acted=True)

    monkeypatch.setattr(responses_mod, "ResponsesHandler", HeldHandler)
    monkeypatch.setattr(scheduler, "_turn_evidence", evidence)
    return SimpleNamespace(started=started, release=release)


# --- Abandoned turns -----------------------------------------------------------


def _left_running_with_a_turn(schedule_id: UUID) -> UUID:
    """A run whose process died after its turn started: linked to a
    conversation, its lease long past."""
    from cowork.services.conversations import ConversationService

    session = get_open_session()
    try:
        scoped = ScopedSession(session, SYSTEM_SCOPE)
        conversation = ConversationService(scoped).create_conversation(
            topic=session.get(Schedule, schedule_id).title, project_id=GENERAL_PROJECT_ID,
        )
        run = ScheduleRunService(scoped).create_run(schedule_id, slot_at=SLOT, lease_seconds=3600)
        row = session.get(ScheduleRun, run.id)
        row.conversation_id = conversation.id
        row.execution_claimed_at = row.started_at
        row.lease_expires_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        session.add(row)
        session.commit()
        return conversation.id
    finally:
        session.close()


def test_a_reaped_run_whose_turn_may_have_started_is_named_for_its_pod_to_stop(schedules):
    schedule_id = schedules()
    conversation_id = _left_running_with_a_turn(schedule_id)

    abandoned: list = []
    _poll(abandoned)

    assert [turn.conversation_id for turn in abandoned] == [str(conversation_id)]
    (reaped,) = _runs(schedule_id)
    assert reaped.status == RunStatus.failed


def _turn_index(monkeypatch, turn: dict | None) -> None:
    """get_turn answering with the entry ``turn``'s fields make, or None."""
    import cowork.streaming.turn_index as turn_index

    entry = None if turn is None else turn_index.TurnIndexEntry.from_index({"turn_id": "1", **turn})

    async def get_turn(_conversation_id):
        return entry

    monkeypatch.setattr(turn_index, "get_turn", get_turn)


def _cancels(monkeypatch, *, fails: bool = False) -> list[str]:
    asked: list[str] = []

    async def request_cancel(correlation_id, **_):
        asked.append(correlation_id)
        if fails:
            raise ConnectionError("redis is down")

    monkeypatch.setattr(scheduler, "request_cancel", request_cancel)
    return asked


def _abandoned(*, run_started_ago: float = 300, lease_expired_ago: float = 10) -> AbandonedTurn:
    """A run reaped just now: it started ``run_started_ago`` seconds ago and
    its lease ran out ``lease_expired_ago`` seconds ago, so its process last
    renewed it a lease before that."""
    now = datetime.now(timezone.utc)
    return AbandonedTurn(
        conversation_id="conv-1",
        run_started_at=now - timedelta(seconds=run_started_ago),
        lease_expired_at=now - timedelta(seconds=lease_expired_ago),
    )


def _started_ago(seconds: float) -> str:
    return str(time.time() - seconds)


async def test_an_abandoned_turn_in_a_pod_is_flagged_to_stop(monkeypatch):
    monkeypatch.setattr(scheduler, "get_backend", lambda: "redis")
    # The run's turn began soon after the run, while its process was alive.
    _turn_index(monkeypatch, {"correlation_id": "corr-1", "started_at": _started_ago(290)})
    asked = _cancels(monkeypatch)

    await scheduler._cancel_abandoned_turn(_abandoned())

    assert asked == ["corr-1"]


async def test_an_abandoned_turn_is_left_alone_on_the_in_process_backend(monkeypatch):
    monkeypatch.setattr(scheduler, "get_backend", lambda: "memory")
    _turn_index(monkeypatch, {"correlation_id": "corr-1", "started_at": _started_ago(290)})
    asked = _cancels(monkeypatch)

    await scheduler._cancel_abandoned_turn(_abandoned())

    assert asked == []


async def test_an_older_turn_on_the_conversation_is_not_flagged(monkeypatch):
    """The turn index names the conversation's current turn. One that began
    well before the run did is not the run's turn, and is already over."""
    monkeypatch.setattr(scheduler, "get_backend", lambda: "redis")
    _turn_index(monkeypatch, {"correlation_id": "old", "started_at": _started_ago(3600)})
    asked = _cancels(monkeypatch)

    await scheduler._cancel_abandoned_turn(_abandoned())

    assert asked == []


async def test_a_turn_begun_after_the_runs_process_stopped_is_not_flagged(monkeypatch):
    """A person can chat in a scheduled run's conversation. A turn that began
    after the run's process stopped renewing its lease is theirs, not the
    run's, so the reap leaves it running."""
    monkeypatch.setattr(scheduler, "get_backend", lambda: "redis")
    # The lease ran out 10s ago, so the process last renewed it ~130s ago and
    # was gone by ~100s ago; this turn began 60s ago.
    _turn_index(monkeypatch, {"correlation_id": "theirs", "started_at": _started_ago(60)})
    asked = _cancels(monkeypatch)

    await scheduler._cancel_abandoned_turn(_abandoned())

    assert asked == []


async def test_a_flag_that_cannot_be_written_is_logged_not_raised(monkeypatch):
    monkeypatch.setattr(scheduler, "get_backend", lambda: "redis")
    _turn_index(monkeypatch, {"correlation_id": "corr-1", "started_at": _started_ago(290)})
    asked = _cancels(monkeypatch, fails=True)

    await scheduler._cancel_abandoned_turn(_abandoned())

    assert asked == ["corr-1"]


async def test_a_run_reaped_while_its_turn_goes_on_stops_the_turn_and_keeps_the_reap(schedules, turns, monkeypatch):
    schedule_id = schedules()
    monkeypatch.setattr(scheduler, "_LEASE_RENEW_SECONDS", 0.05)
    stopped: list[str] = []

    async def stop(conversation_id):
        stopped.append(conversation_id)
        turns.release.set()

    monkeypatch.setattr(scheduler, "_stop_reaped_turn", stop)

    run = asyncio.create_task(scheduler.execute_schedule(schedule_id, slot_at=SLOT))
    async with asyncio.timeout(10):
        while not turns.started:
            await asyncio.sleep(0.01)
    # Another replica's poll, an hour on, finds the lease lapsed and reaps it.
    session = get_open_session()
    try:
        with monkeypatch.context() as replica:
            replica.setattr(scheduler, "_active_run_ids", set())
            scheduler._reap_lapsed_runs(ScopedSession(session, SYSTEM_SCOPE), datetime.now(timezone.utc) + timedelta(hours=1))
    finally:
        session.close()
    await asyncio.wait_for(run, timeout=10)

    assert stopped == turns.started
    (reaped,) = _runs(schedule_id)
    assert reaped.status == RunStatus.failed
    assert reaped.error == scheduler._LAPSED_RUN_ERROR
    assert ensure_utc(_schedule(schedule_id).next_run_at) == SLOT + timedelta(days=1), (
        "the reaper settled the slot once; the stopped run does not settle it again"
    )


async def test_a_reaped_turn_is_stopped_here_and_in_its_pod(monkeypatch):
    handle = SimpleNamespace(
        is_running=True,
        lifecycle=SimpleNamespace(timed_out=False, identity=SimpleNamespace(remote=True, correlation_id="corr-9")),
        cancelled=False,
    )

    async def cancel():
        handle.cancelled = True

    handle.cancel = cancel
    monkeypatch.setattr(scheduler.registry, "get", lambda _conversation_id: handle)
    monkeypatch.setattr(scheduler, "get_backend", lambda: "memory")
    asked = _cancels(monkeypatch)

    await scheduler._stop_reaped_turn("conv-9")

    assert asked == ["corr-9"]
    assert handle.cancelled
    assert handle.lifecycle.timed_out, "saved as interrupted, not as the user's Stop"


# --- One organization's runs ---------------------------------------------------


async def test_a_slot_over_its_organizations_limit_stays_due(schedules, turns):
    turns.release.set()
    busy = [schedules(org_id="org-a") for _ in range(scheduler._org_run_limit())]
    for schedule_id in busy:
        _running(schedule_id)
    waiting = schedules(org_id="org-a")

    await scheduler.execute_schedule(waiting, slot_at=SLOT)

    assert _runs(waiting) == [], "nothing was claimed"
    assert ensure_utc(_schedule(waiting).next_run_at) == SLOT
    assert waiting not in [slot.schedule_id for slot in _poll()], "full orgs do not reserve process capacity"


async def test_other_organizations_and_desktop_schedules_are_not_held_back(schedules, turns):
    turns.release.set()
    for _ in range(scheduler._org_run_limit()):
        _running(schedules(org_id="org-a"))
    other_org = schedules(org_id="org-b")
    desktop = schedules()

    await scheduler.execute_schedule(other_org, slot_at=SLOT)
    await scheduler.execute_schedule(desktop, slot_at=SLOT)

    assert [run.status for run in _runs(other_org)] == [RunStatus.success]
    assert [run.status for run in _runs(desktop)] == [RunStatus.success]


def test_two_replicas_claiming_for_one_organization_count_one_after_the_other(schedules):
    """Each claim counts its organization's runs in the insert's transaction,
    so the second replica sees the first replica's run."""
    for _ in range(scheduler._org_run_limit() - 1):
        _running(schedules(org_id="org-a"))
    first, second = schedules(org_id="org-a"), schedules(org_id="org-a")

    replica_a, replica_b = get_open_session(), get_open_session()
    try:
        ScheduleRunService(ScopedSession(replica_a, SYSTEM_SCOPE)).create_run(
            first, slot_at=SLOT, lease_seconds=60, org_limit=scheduler._org_run_limit(),
        )
        with pytest.raises(OrgRunsAtLimit):
            ScheduleRunService(ScopedSession(replica_b, SYSTEM_SCOPE)).create_run(
                second, slot_at=SLOT, lease_seconds=60, org_limit=scheduler._org_run_limit(),
            )
    finally:
        replica_a.close()
        replica_b.close()

    assert len(_runs(first)) == 1
    assert _runs(second) == []


def test_org_runs_full_counts_only_runs_going(schedules):
    schedule_id = schedules(org_id="org-a")
    session = get_open_session()
    try:
        scoped = ScopedSession(session, SYSTEM_SCOPE)
        schedule = ScheduleService(scoped).get_schedule(schedule_id)
        for _ in range(5):
            finished = _running(schedules(org_id="org-a"))
            ScheduleRunService(scoped).finish_run(finished.id, status=RunStatus.success)
        assert not scheduler.org_runs_full(scoped, schedule)
        for _ in range(scheduler._org_run_limit()):
            _running(schedules(org_id="org-a"))
        assert scheduler.org_runs_full(scoped, schedule)
        desktop = ScheduleService(scoped).get_schedule(schedules())
        assert not scheduler.org_runs_full(scoped, desktop)
    finally:
        session.close()


async def test_run_now_is_refused_when_the_organization_is_at_its_limit(schedules, monkeypatch):
    from cowork.server import create_app

    schedule_id = schedules()
    monkeypatch.setattr(scheduler, "org_runs_full", lambda _session, _schedule: True)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test") as client:
        answer = await client.post(f"/api/v1/schedules/{schedule_id}/run-now")

    assert answer.status_code == 409
    assert "account" in answer.json()["detail"]
    assert scheduler._admission.held == 0


async def test_run_now_taking_the_organizations_last_place_is_counted_in_its_claim(schedules, monkeypatch):
    """The early check can pass for two presses at once; the claim counts the
    organization's runs again in its own transaction, so a press over the
    limit is refused and nothing is left running for it."""
    from cowork.server import create_app

    for _ in range(scheduler._org_run_limit()):
        _running(schedules(org_id="org-a"))
    schedule_id = schedules(org_id="org-a")
    monkeypatch.setattr(scheduler, "org_runs_full", lambda _session, _schedule: False)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test") as client:
        answer = await client.post(f"/api/v1/schedules/{schedule_id}/run-now")

    assert answer.status_code == 409
    assert "account" in answer.json()["detail"]
    assert _runs(schedule_id) == []
    assert scheduler._admission.held == 0


# --- The run's time limit -------------------------------------------------------


def test_the_time_limit_defaults_to_30_minutes_within_its_bounds(monkeypatch):
    from cowork.common.settings.app_settings import AppSettings

    monkeypatch.delenv("COWORK_SCHEDULED_RUN_MAX_SECONDS", raising=False)
    assert AppSettings().scheduled_run_max_seconds == 1800
    monkeypatch.setenv("COWORK_SCHEDULED_RUN_MAX_SECONDS", "7200")
    assert AppSettings().scheduled_run_max_seconds == 7200
    for out_of_bounds in ("299", "86401"):
        monkeypatch.setenv("COWORK_SCHEDULED_RUN_MAX_SECONDS", out_of_bounds)
        with pytest.raises(ValidationError):
            AppSettings()


def test_the_run_limits_default_by_where_turns_run_within_their_bounds(monkeypatch):
    from cowork.common.settings.app_settings import AppSettings

    monkeypatch.delenv("COWORK_SCHEDULED_RUNS_PER_PROCESS", raising=False)
    monkeypatch.delenv("COWORK_SCHEDULED_RUNS_PER_ORG", raising=False)
    settings = AppSettings()
    assert settings.scheduled_runs_per_process is None
    assert settings.scheduled_runs_per_org is None
    monkeypatch.setenv("COWORK_SCHEDULED_RUNS_PER_PROCESS", "24")
    monkeypatch.setenv("COWORK_SCHEDULED_RUNS_PER_ORG", "12")
    settings = AppSettings()
    assert (settings.scheduled_runs_per_process, settings.scheduled_runs_per_org) == (24, 12)
    for name, out_of_bounds in (
        ("COWORK_SCHEDULED_RUNS_PER_PROCESS", "0"),
        ("COWORK_SCHEDULED_RUNS_PER_PROCESS", "65"),
        ("COWORK_SCHEDULED_RUNS_PER_ORG", "0"),
        ("COWORK_SCHEDULED_RUNS_PER_ORG", "257"),
    ):
        monkeypatch.setenv(name, out_of_bounds)
        with pytest.raises(ValidationError):
            AppSettings()
        monkeypatch.setenv(name, "4")


@pytest.mark.parametrize(
    ("backend", "configured", "expected"),
    [("memory", None, 3), ("redis", None, 16), ("memory", 5, 5), ("redis", 40, 40)],
)
def test_the_process_limit_follows_the_backend_unless_set(monkeypatch, backend, configured, expected):
    monkeypatch.setattr(
        "cowork.common.settings.app_settings.get_app_settings",
        lambda: SimpleNamespace(scheduled_runs_per_process=configured, scheduled_runs_per_org=8),
    )
    monkeypatch.setattr(scheduler, "get_backend", lambda: backend)

    assert scheduler._runs_per_process() == expected


def test_admission_reads_its_limit_at_each_reservation():
    limit = {"value": 1}
    admission = scheduler._Admission(lambda: limit["value"])

    assert admission.reserve()
    assert not admission.reserve()
    limit["value"] = 2
    assert admission.reserve()


def test_the_organization_limit_comes_from_its_setting(monkeypatch):
    monkeypatch.setattr(
        "cowork.common.settings.app_settings.get_app_settings",
        lambda: SimpleNamespace(scheduled_runs_per_process=None, scheduled_runs_per_org=11),
    )

    assert scheduler._org_run_limit() == 11


def _limit(monkeypatch, seconds: int, backend: str = "memory") -> None:
    monkeypatch.setattr(
        "cowork.common.settings.app_settings.get_app_settings",
        lambda: SimpleNamespace(scheduled_run_max_seconds=seconds),
    )
    monkeypatch.setattr(scheduler, "get_backend", lambda: backend)


def test_a_run_is_stopped_at_the_configured_limit(monkeypatch):
    _limit(monkeypatch, 3600)
    assert scheduler._run_duration_limit() == 3600


def test_a_pod_turn_gets_at_least_the_controllers_own_bound(monkeypatch):
    _limit(monkeypatch, 300, backend="redis")
    assert scheduler._run_duration_limit() == (
        scheduler._REMOTE_TURN_BUDGET_SECONDS + scheduler._REMOTE_SETUP_MARGIN_SECONDS
    )
    _limit(monkeypatch, 3600, backend="redis")
    assert scheduler._run_duration_limit() == 3600


def test_the_timeout_message_names_the_limit():
    assert scheduler._duration_error(1800) == "Run exceeded the 30-minute limit and was stopped."
    assert scheduler._duration_error(5400) == "Run exceeded the 90-minute limit and was stopped."


async def test_waking_process_does_not_reap_its_own_live_run(schedules, turns):
    schedule_id = schedules()
    task = asyncio.create_task(scheduler.execute_schedule(schedule_id, slot_at=SLOT))
    try:
        async with asyncio.timeout(5):
            while not turns.started:
                await asyncio.sleep(0)
        with get_open_session() as session:
            scheduler._reap_lapsed_runs(ScopedSession(session, SYSTEM_SCOPE),
                                      datetime.now(timezone.utc) + timedelta(hours=1))
        assert _runs(schedule_id)[0].status == RunStatus.running
    finally:
        turns.release.set()
        await task
    assert _runs(schedule_id)[0].status == RunStatus.success
    assert not scheduler._active_run_ids


async def test_deleting_task_stops_its_turn_and_logs_deletion(schedules, turns, monkeypatch, caplog):
    import logging
    schedule_id = schedules()
    monkeypatch.setattr(scheduler, "_LEASE_RENEW_SECONDS", 0.01)
    monkeypatch.setattr(scheduler.logger, "disabled", False)
    caplog.set_level(logging.WARNING, logger=scheduler.logger.name)
    stopped = []

    async def stop(conversation_id):
        stopped.append(conversation_id)
        turns.release.set()

    monkeypatch.setattr(scheduler, "_stop_reaped_turn", stop)
    task = asyncio.create_task(scheduler.execute_schedule(schedule_id, slot_at=SLOT))
    async with asyncio.timeout(5):
        while not turns.started:
            await asyncio.sleep(0)
    with get_open_session() as session:
        ScheduleService(ScopedSession(session, SYSTEM_SCOPE)).delete_schedule(schedule_id)
    await asyncio.wait_for(task, 5)
    assert stopped == turns.started
    assert any("task was deleted" in r.getMessage() for r in caplog.records)
    assert not scheduler._active_run_ids
    # Deleting a task intentionally leaves its conversation; clean it up here.
    from cowork.services.conversations import ConversationService
    with get_open_session() as session:
        ConversationService(ScopedSession(session, SYSTEM_SCOPE)).delete_conversation(UUID(stopped[0]))


def test_poll_reserves_only_available_org_capacity_before_process_admission(schedules, monkeypatch):
    monkeypatch.setattr(scheduler, "_org_run_limit", lambda: 2)
    _running(schedules(org_id="a"))
    a_due = [schedules(org_id="a") for _ in range(20)]
    b_due = [schedules(org_id="b") for _ in range(3)]
    with get_open_session() as session:
        scoped = ScopedSession(session, SYSTEM_SCOPE)
        ordered = [session.get(Schedule, id) for id in a_due + b_due]
        monkeypatch.setattr(scheduler, "_due_schedules", lambda *_: ordered)
        ids = [slot.schedule_id for slot in scheduler._poll(scoped)]
    assert ids == [a_due[0], *b_due[:2]]


def test_unset_org_limit_does_not_count_or_refuse_runs(schedules, monkeypatch):
    monkeypatch.setattr(scheduler, "_org_run_limit", lambda: None)

    def forbidden(*args, **kwargs):
        pytest.fail("unset org limit must not count running rows")

    monkeypatch.setattr(ScheduleRunService, "org_runs_in_flight", forbidden)
    ids = [schedules(org_id="a") for _ in range(4)]
    assert set(ids) <= {slot.schedule_id for slot in _poll()}
    with get_open_session() as session:
        scoped = ScopedSession(session, SYSTEM_SCOPE)
        assert not scheduler.org_runs_full(scoped, session.get(Schedule, ids[0]))
        scheduler.claim_manual_run(scoped, ids[0])


@pytest.mark.parametrize("write", ["outcome", "error", "finish"])
def test_late_writes_leave_reaped_result_and_streak_untouched(schedules, write):
    from cowork.schemas.schedules import AttemptOutcome
    schedule_id = schedules()
    _left_running_with_a_turn(schedule_id)
    _poll()
    before = _runs(schedule_id)[0]
    assert before.status == RunStatus.failed
    streak = _schedule(schedule_id).consecutive_failures
    next_run = _schedule(schedule_id).next_run_at
    with get_open_session() as session:
        scoped = ScopedSession(session, SYSTEM_SCOPE)
        if write == "finish":
            scheduler._finish_run(scoped, run_id=before.id, conversation_id=None, error=None, status=RunStatus.success)
        else:
            with pytest.raises(scheduler._LeaseLost):
                if write == "outcome":
                    scheduler._record_turn_outcome(
                        scoped, schedule_id=schedule_id, run_id=before.id,
                        end=scheduler._AttemptEnd(AttemptOutcome.failed_during_execution, "late failure"),
                        is_manual=False, conversation_id=None,
                    )
                else:
                    scheduler._record_schedule_error(scoped, schedule_id=schedule_id, run_id=before.id,
                                                     error="late failure", consume_slot=True)
    after = _runs(schedule_id)[0]
    assert (after.status, after.error, after.finished_at) == (before.status, before.error, before.finished_at)
    assert _schedule(schedule_id).consecutive_failures == streak == 1
    assert _schedule(schedule_id).next_run_at == next_run


def test_outcome_settlement_and_finish_are_one_commit(schedules):
    from cowork.schemas.schedules import AttemptOutcome
    schedule_id = schedules()
    _left_running_with_a_turn(schedule_id)
    run = _runs(schedule_id)[0]
    with get_open_session() as session:
        scheduler._record_turn_outcome(
            ScopedSession(session, SYSTEM_SCOPE), schedule_id=schedule_id, run_id=run.id,
            end=scheduler._AttemptEnd(AttemptOutcome.failed_during_execution, "failed"),
            is_manual=False, conversation_id=run.conversation_id,
        )
    assert _runs(schedule_id)[0].status == RunStatus.failed
    _poll()  # Its expired lease must not settle the already finalized failure again.
    assert _schedule(schedule_id).consecutive_failures == 1
    assert _runs(schedule_id)[0].error == "failed"


@pytest.mark.parametrize("cancelled", [False, True])
async def test_committed_reaps_are_cancelled_even_if_poll_does_not_return(schedules, monkeypatch, cancelled):
    schedule_id = schedules()
    conversation_id = _left_running_with_a_turn(schedule_id)
    called = asyncio.Event()
    seen = []

    def fail(*_):
        raise asyncio.CancelledError() if cancelled else RuntimeError("later poll read failed")

    async def cancel(turn):
        seen.append(turn.conversation_id)
        called.set()

    monkeypatch.setattr(scheduler, "_handle_missed_runs", fail)
    monkeypatch.setattr(scheduler, "_cancel_abandoned_turn", cancel)
    monkeypatch.setattr(scheduler, "_POLL_INTERVAL_SECONDS", 0)
    loop = asyncio.create_task(scheduler._scheduler_loop())
    try:
        await asyncio.wait_for(called.wait(), 5)
    finally:
        loop.cancel()
        await asyncio.gather(loop, return_exceptions=True)
    assert seen == [str(conversation_id)]
    assert _runs(schedule_id)[0].status == RunStatus.failed
