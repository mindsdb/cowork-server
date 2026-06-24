"""Failure policy + missed-run policy + idempotency for scheduled tasks.

These exercise the reliability layer added on top of the timezone slice:

  * transient run failures retry in-process with backoff; permanent ones don't
  * a hung run is bounded by a per-run timeout
  * repeated consecutive failures auto-pause the task and record health
  * each missed-run policy (skip / run_once / catch_up) reconciles correctly,
    including a ``once`` task whose time has passed
  * idempotency keys stop a manual + scheduled (or double-tick) double-fire

All DB access goes through the throwaway SQLite engine from conftest.py — no
real ~/.cowork data is touched. The LLM call and conversation creation are
stubbed so we test the scheduler's control flow, not the model.

``pytest-asyncio`` isn't a dependency here, so coroutines are driven directly
with ``asyncio.run`` from synchronous test functions.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlmodel import Session

import cowork.scheduler as scheduler
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.schedule import Schedule
from cowork.schemas.schedules import Cadence, MissedRunPolicy, ScheduleHealth
from cowork.scheduler import (
    _apply_failure,
    _attempt_with_retries,
    _handle_missed_runs,
    execute_schedule,
)
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.schedules import ScheduleRunService, ScheduleService

UTC = timezone.utc


@pytest.fixture()
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


@pytest.fixture(autouse=True)
def _clean_schedules():
    """Wipe schedules/runs before each test.

    The DB engine is session-scoped (see conftest), and ``_handle_missed_runs``
    / the scheduler operate over *all* schedules — so leftover rows from a
    prior test would pollute reconciliation counts. Start every test clean.
    """
    from sqlmodel import delete

    from cowork.models.schedule import ScheduleRun

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        s.exec(delete(ScheduleRun))
        s.exec(delete(Schedule))
        s.commit()
    yield


def _make_schedule(
    session: Session,
    *,
    cadence: str = Cadence.daily,
    next_run_at: datetime | None = None,
    tz: str = "UTC",
    enabled: bool = True,
    missed_run_policy: str = MissedRunPolicy.skip,
) -> Schedule:
    sched = Schedule(
        title="reliability-test",
        prompt="do the thing",
        cadence=cadence,
        timezone=tz,
        next_run_at=next_run_at or datetime(2025, 6, 1, 9, 0, tzinfo=UTC),
        model="default",
        project_id=GENERAL_PROJECT_ID,
        enabled=enabled,
        missed_run_policy=missed_run_policy,
    )
    session.add(sched)
    session.commit()
    session.refresh(sched)
    return sched


def _freeze_now(monkeypatch, fixed_now: datetime) -> None:
    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.astimezone(tz) if tz else fixed_now.replace(tzinfo=None)

    monkeypatch.setattr(scheduler, "datetime", _FrozenDatetime)


class _FakeHandler:
    """Stand-in for ResponsesHandler whose handle() does what the test wants."""

    behaviour = None  # set per-test: an async callable

    def __init__(self, session):
        self.session = session

    async def handle(self, request):
        return await type(self).behaviour()


def _install_fake_llm(monkeypatch, behaviour) -> list[int]:
    """Patch the LLM handler + conversation creation; return a call counter.

    ``behaviour`` is an async callable invoked on every handle(); ``calls[0]``
    counts how many times it ran (i.e. attempts across retries).
    """
    import cowork.handlers.responses as responses_mod
    import cowork.services.conversations as conversations_mod

    calls = [0]

    async def _counted():
        calls[0] += 1
        return await behaviour()

    _FakeHandler.behaviour = staticmethod(_counted)
    monkeypatch.setattr(responses_mod, "ResponsesHandler", _FakeHandler)

    class _FakeConversation:
        def __init__(self):
            self.id = uuid4()

    class _FakeConversationService:
        def __init__(self, session):
            pass

        def create_conversation(self, topic, project_id=None, conversation_id=None):
            return _FakeConversation()

    monkeypatch.setattr(conversations_mod, "ConversationService", _FakeConversationService)
    return calls


def _instant_backoff(monkeypatch) -> None:
    """Make retry backoff sleeps return immediately."""
    async def _no_sleep(*_a, **_k):
        return None

    monkeypatch.setattr(scheduler.asyncio, "sleep", _no_sleep)


# ── retry / backoff ─────────────────────────────────────────────────


def test_retries_transient_failure_then_succeeds(session, monkeypatch):
    # Fail twice (transient) then succeed; the run should report 3 attempts
    # and end healthy. Backoff sleeps are stubbed so the test is instant.
    _instant_backoff(monkeypatch)
    state = {"n": 0}

    async def flaky():
        state["n"] += 1
        if state["n"] < 3:
            raise ConnectionError("temporary network blip")
        return None

    calls = _install_fake_llm(monkeypatch, flaky)
    sched = _make_schedule(session, cadence=Cadence.once)

    asyncio.run(execute_schedule(sched.id, is_manual=True))

    assert calls[0] == 3
    runs = ScheduleRunService(session).list_runs(sched.id)
    assert len(runs) == 1
    assert runs[0].status == "success"
    assert runs[0].attempts == 3
    session.refresh(sched)
    assert sched.last_error is None
    assert sched.consecutive_failures == 0
    assert sched.health == ScheduleHealth.ok


def test_backoff_grows_and_is_capped(monkeypatch):
    # Pure retry helper: capture the sleep delays and assert exponential
    # growth, capped at _RETRY_MAX_DELAY_SECONDS.
    monkeypatch.setattr(scheduler, "_MAX_ATTEMPTS", 5)
    monkeypatch.setattr(scheduler, "_RETRY_BASE_DELAY_SECONDS", 1.0)
    monkeypatch.setattr(scheduler, "_RETRY_MAX_DELAY_SECONDS", 4.0)
    delays: list[float] = []

    async def _fake_sleep(d):
        delays.append(d)

    monkeypatch.setattr(scheduler.asyncio, "sleep", _fake_sleep)

    async def always_fail():
        raise ConnectionError("nope")

    with pytest.raises(ConnectionError):
        asyncio.run(_attempt_with_retries(always_fail))

    # 5 attempts -> 4 backoffs: 1, 2, 4, 4 (capped).
    assert delays == [1.0, 2.0, 4.0, 4.0]


def test_permanent_failure_is_not_retried(session, monkeypatch):
    _instant_backoff(monkeypatch)

    async def permanent():
        raise ValueError("schedule not found")  # matches a permanent marker

    calls = _install_fake_llm(monkeypatch, permanent)
    sched = _make_schedule(session, cadence=Cadence.once)

    asyncio.run(execute_schedule(sched.id, is_manual=True))

    assert calls[0] == 1  # no retries
    runs = ScheduleRunService(session).list_runs(sched.id)
    assert runs[0].status == "failed"
    assert runs[0].attempts == 1


# ── per-run timeout ─────────────────────────────────────────────────


def test_per_run_timeout_bounds_a_hung_run(session, monkeypatch):
    # A hung attempt must be cut off by the per-run timeout and counted as a
    # failure. Shrink the timeout and cap attempts at 1 so it's fast. The
    # hang awaits an Event that never fires (real await, so wait_for trips).
    monkeypatch.setattr(scheduler, "_RUN_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(scheduler, "_MAX_ATTEMPTS", 1)

    async def hang():
        await asyncio.Event().wait()  # never completes -> times out

    _install_fake_llm(monkeypatch, hang)
    sched = _make_schedule(session, cadence=Cadence.once)

    asyncio.run(execute_schedule(sched.id, is_manual=True))

    runs = ScheduleRunService(session).list_runs(sched.id)
    assert runs[0].status == "failed"
    assert runs[0].error  # a timeout error string was recorded


def test_timeout_is_treated_as_transient(monkeypatch):
    # Direct check that the retry helper retries a timed-out attempt. Backoff
    # sleeps are stubbed; the hang awaits a never-firing Event so it actually
    # hits the wait_for timeout rather than the stubbed sleep.
    monkeypatch.setattr(scheduler, "_RUN_TIMEOUT_SECONDS", 0.02)
    monkeypatch.setattr(scheduler, "_MAX_ATTEMPTS", 2)
    _instant_backoff(monkeypatch)
    attempts = {"n": 0}

    async def hang_once_then_ok():
        attempts["n"] += 1
        if attempts["n"] == 1:
            await asyncio.Event().wait()  # will time out
        return None

    n = asyncio.run(_attempt_with_retries(hang_once_then_ok))
    assert n == 2


# ── auto-pause after N consecutive failures ─────────────────────────


def test_apply_failure_auto_pauses_after_threshold(session, monkeypatch):
    monkeypatch.setattr(scheduler, "_MAX_CONSECUTIVE_FAILURES", 3)
    sched = _make_schedule(session, cadence=Cadence.daily)

    for i in range(1, 4):
        _apply_failure(sched, f"boom {i}", session)
        session.commit()
        session.refresh(sched)
        if i < 3:
            assert sched.enabled is True
            assert sched.health == ScheduleHealth.failing
            assert sched.consecutive_failures == i

    # Third consecutive failure trips the auto-pause.
    assert sched.enabled is False
    assert sched.health == ScheduleHealth.paused
    assert sched.consecutive_failures == 3
    assert sched.last_error == "boom 3"


def test_repeated_failures_auto_pause_through_execute(session, monkeypatch):
    monkeypatch.setattr(scheduler, "_MAX_CONSECUTIVE_FAILURES", 2)
    monkeypatch.setattr(scheduler, "_MAX_ATTEMPTS", 1)
    _instant_backoff(monkeypatch)

    async def boom():
        raise ConnectionError("still broken")

    _install_fake_llm(monkeypatch, boom)
    # Hourly so a failed live tick advances rather than disabling-as-once.
    base = datetime(2025, 6, 1, 9, 0, tzinfo=UTC)
    sched = _make_schedule(session, cadence=Cadence.hourly, next_run_at=base)

    # Two scheduled (live-tick) failures -> auto-paused.
    asyncio.run(execute_schedule(sched.id, is_manual=False))
    session.refresh(sched)
    assert sched.enabled is True
    assert sched.consecutive_failures == 1

    asyncio.run(execute_schedule(sched.id, is_manual=False))
    session.refresh(sched)
    assert sched.enabled is False
    assert sched.health == ScheduleHealth.paused
    assert sched.consecutive_failures == 2


def test_resume_clears_failure_streak(session):
    sched = _make_schedule(session, cadence=Cadence.daily, enabled=False)
    sched.consecutive_failures = 9
    sched.health = ScheduleHealth.paused
    sched.last_error = "old boom"
    session.add(sched)
    session.commit()

    resumed = ScheduleService(session).resume_schedule(sched.id)
    assert resumed.enabled is True
    assert resumed.consecutive_failures == 0
    assert resumed.last_error is None
    assert resumed.health == ScheduleHealth.ok


# ── missed-run policy ───────────────────────────────────────────────


def test_missed_policy_skip_runs_current_only(session, monkeypatch):
    # skip drops the offline backlog but still runs the occurrence that's
    # due right now (Jun 4) — otherwise a normal cadence tick would never
    # fire. next_run fast-forwards past everything due.
    base = datetime(2025, 6, 1, 9, 0, tzinfo=UTC)
    _freeze_now(monkeypatch, datetime(2025, 6, 4, 11, 0, tzinfo=UTC))
    sched = _make_schedule(
        session, cadence=Cadence.daily, next_run_at=base, missed_run_policy=MissedRunPolicy.skip
    )

    to_run = _handle_missed_runs(session)
    session.refresh(sched)

    assert len(to_run) == 1  # only the current (most-recent due) occurrence
    _sid, key = to_run[0]
    assert key.endswith(datetime(2025, 6, 4, 9, 0, tzinfo=UTC).isoformat())
    assert sched.missed_runs == 4  # Jun 1,2,3,4 all counted as elapsed
    assert scheduler._as_utc(sched.next_run_at) == datetime(2025, 6, 5, 9, 0, tzinfo=UTC)


def test_single_due_tick_runs_under_skip(session, monkeypatch):
    # Normal operation: only one occurrence is due (no backlog). All policies
    # run it; nothing is "missed".
    base = datetime(2025, 6, 1, 9, 0, tzinfo=UTC)
    _freeze_now(monkeypatch, datetime(2025, 6, 1, 9, 0, 30, tzinfo=UTC))  # 30s after due
    sched = _make_schedule(
        session, cadence=Cadence.daily, next_run_at=base, missed_run_policy=MissedRunPolicy.skip
    )

    to_run = _handle_missed_runs(session)
    session.refresh(sched)

    assert len(to_run) == 1
    assert sched.missed_runs == 1
    assert scheduler._as_utc(sched.next_run_at) == datetime(2025, 6, 2, 9, 0, tzinfo=UTC)


def test_missed_policy_run_once_replays_one_backlog_plus_current(session, monkeypatch):
    base = datetime(2025, 6, 1, 9, 0, tzinfo=UTC)
    _freeze_now(monkeypatch, datetime(2025, 6, 4, 11, 0, tzinfo=UTC))
    sched = _make_schedule(
        session, cadence=Cadence.daily, next_run_at=base, missed_run_policy=MissedRunPolicy.run_once
    )

    to_run = _handle_missed_runs(session)
    session.refresh(sched)

    # One backlog catch-up (Jun 3, the newest missed) + the current run (Jun 4).
    assert len(to_run) == 2
    keys = [k for _sid, k in to_run]
    assert keys == [
        f"{sched.id}@{datetime(2025, 6, 3, 9, 0, tzinfo=UTC).isoformat()}",
        f"{sched.id}@{datetime(2025, 6, 4, 9, 0, tzinfo=UTC).isoformat()}",
    ]
    assert scheduler._as_utc(sched.next_run_at) == datetime(2025, 6, 5, 9, 0, tzinfo=UTC)


def test_missed_policy_catch_up_replays_all_missed(session, monkeypatch):
    base = datetime(2025, 6, 1, 9, 0, tzinfo=UTC)
    _freeze_now(monkeypatch, datetime(2025, 6, 4, 11, 0, tzinfo=UTC))
    sched = _make_schedule(
        session, cadence=Cadence.daily, next_run_at=base, missed_run_policy=MissedRunPolicy.catch_up
    )

    to_run = _handle_missed_runs(session)
    session.refresh(sched)

    assert len(to_run) == 4  # Jun 1,2,3 backlog + Jun 4 current, oldest-first
    keys = [k for _sid, k in to_run]
    expected = [
        f"{sched.id}@{datetime(2025, 6, d, 9, 0, tzinfo=UTC).isoformat()}"
        for d in (1, 2, 3, 4)
    ]
    assert keys == expected
    assert scheduler._as_utc(sched.next_run_at) == datetime(2025, 6, 5, 9, 0, tzinfo=UTC)


def test_catch_up_backlog_is_capped(session, monkeypatch):
    monkeypatch.setattr(scheduler, "_MAX_CATCH_UP", 3)
    base = datetime(2025, 6, 1, 9, 0, tzinfo=UTC)
    _freeze_now(monkeypatch, datetime(2025, 6, 10, 11, 0, tzinfo=UTC))  # ~10 due
    sched = _make_schedule(
        session, cadence=Cadence.daily, next_run_at=base, missed_run_policy=MissedRunPolicy.catch_up
    )

    to_run = _handle_missed_runs(session)
    session.refresh(sched)

    # Backlog capped at _MAX_CATCH_UP (3) + the current occurrence = 4.
    assert len(to_run) == 4
    assert sched.missed_runs == 10  # but the counter still reflects reality


def test_once_task_runs_under_run_once_policy(session, monkeypatch):
    # A one-off whose time passed: under run_once it still gets dispatched
    # exactly once, then is disabled.
    past = datetime(2025, 6, 1, 9, 0, tzinfo=UTC)
    _freeze_now(monkeypatch, datetime(2025, 6, 2, 9, 0, tzinfo=UTC))
    sched = _make_schedule(
        session, cadence=Cadence.once, next_run_at=past, missed_run_policy=MissedRunPolicy.run_once
    )

    to_run = _handle_missed_runs(session)
    session.refresh(sched)

    assert len(to_run) == 1
    assert to_run[0][0] == sched.id
    assert sched.enabled is False  # spent


def test_once_task_skipped_under_skip_policy(session, monkeypatch):
    past = datetime(2025, 6, 1, 9, 0, tzinfo=UTC)
    _freeze_now(monkeypatch, datetime(2025, 6, 2, 9, 0, tzinfo=UTC))
    sched = _make_schedule(
        session, cadence=Cadence.once, next_run_at=past, missed_run_policy=MissedRunPolicy.skip
    )

    to_run = _handle_missed_runs(session)
    session.refresh(sched)

    assert to_run == []
    assert sched.enabled is False


# ── idempotency ─────────────────────────────────────────────────────


def test_idempotency_key_prevents_double_fire(session, monkeypatch):
    # Running the same occurrence key twice must execute the work only once.
    monkeypatch.setattr(scheduler, "_MAX_ATTEMPTS", 1)

    async def ok():
        return None

    calls = _install_fake_llm(monkeypatch, ok)
    sched = _make_schedule(session, cadence=Cadence.daily)
    key = scheduler._occurrence_key(sched)

    asyncio.run(execute_schedule(sched.id, is_manual=False, idempotency_key=key))
    asyncio.run(execute_schedule(sched.id, is_manual=False, idempotency_key=key))

    assert calls[0] == 1  # second call short-circuited
    runs = ScheduleRunService(session).list_runs(sched.id)
    assert len(runs) == 1


def test_catch_up_dispatch_executes_each_once(session, monkeypatch):
    # End-to-end: the keys _handle_missed_runs hands back drive one run each,
    # and re-dispatching the same keys (e.g. a second poll) double-fires none.
    monkeypatch.setattr(scheduler, "_MAX_ATTEMPTS", 1)
    base = datetime(2025, 6, 1, 9, 0, tzinfo=UTC)
    _freeze_now(monkeypatch, datetime(2025, 6, 4, 11, 0, tzinfo=UTC))

    async def ok():
        return None

    calls = _install_fake_llm(monkeypatch, ok)
    sched = _make_schedule(
        session, cadence=Cadence.daily, next_run_at=base, missed_run_policy=MissedRunPolicy.catch_up
    )

    to_run = _handle_missed_runs(session)
    assert len(to_run) == 4
    for sid, key in to_run:
        asyncio.run(execute_schedule(sid, is_manual=False, idempotency_key=key))
    # Re-dispatch the identical keys — every one is now a no-op.
    for sid, key in to_run:
        asyncio.run(execute_schedule(sid, is_manual=False, idempotency_key=key))

    assert calls[0] == 4  # four distinct occurrences ran exactly once each
    runs = ScheduleRunService(session).list_runs(sched.id)
    assert len(runs) == 4
    assert all(r.status == "success" for r in runs)


def test_slow_run_does_not_double_advance(session, monkeypatch):
    # Regression: a live tick whose work outlives a poll interval must not
    # advance the cadence twice. Simulate an in-flight missed-runs sweep by
    # moving next_run_at out from under the run while its work executes.
    monkeypatch.setattr(scheduler, "_MAX_ATTEMPTS", 1)
    base = datetime(2025, 6, 1, 9, 0, tzinfo=UTC)
    sched = _make_schedule(session, cadence=Cadence.daily, next_run_at=base)
    sid = sched.id
    tick_key = scheduler._occurrence_key(sched)

    async def slow():
        # Mid-run, a concurrent sweep fast-forwards the schedule to Jun 3.
        engine = get_engine(get_app_settings().database.uri)
        with Session(engine) as s2:
            other = s2.get(Schedule, sid)
            other.next_run_at = datetime(2025, 6, 3, 9, 0, tzinfo=UTC)
            s2.add(other)
            s2.commit()
        return None

    _install_fake_llm(monkeypatch, slow)

    asyncio.run(execute_schedule(sid, is_manual=False, idempotency_key=tick_key))

    session.expire_all()
    refreshed = session.get(Schedule, sid)
    # The run's own advance must be a no-op (its tick instant no longer
    # matches), leaving the sweep's Jun 3 intact — NOT advanced to Jun 4.
    assert scheduler._as_utc(refreshed.next_run_at) == datetime(2025, 6, 3, 9, 0, tzinfo=UTC)


def test_manual_run_does_not_collide_with_scheduled(session, monkeypatch):
    # A manual run uses a fresh uuid key, so it always fires even if the
    # current occurrence already ran on schedule.
    monkeypatch.setattr(scheduler, "_MAX_ATTEMPTS", 1)

    async def ok():
        return None

    calls = _install_fake_llm(monkeypatch, ok)
    sched = _make_schedule(session, cadence=Cadence.daily)
    key = scheduler._occurrence_key(sched)

    asyncio.run(execute_schedule(sched.id, is_manual=False, idempotency_key=key))  # scheduled
    asyncio.run(execute_schedule(sched.id, is_manual=True))  # manual run-now

    assert calls[0] == 2
    runs = ScheduleRunService(session).list_runs(sched.id)
    assert len(runs) == 2
    assert sum(1 for r in runs if r.is_manual) == 1
