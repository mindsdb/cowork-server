"""A scheduled run tells its schedule's owner how it ended (cowork.scheduler).

Finished and didn't-finish both push; a cancelled run, a trial (its owner is
watching it in the editor) and a run another replica reaped do not push from
the run itself. The reaper pushes the reaped run's owner instead.
"""
from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

import pytest

from cowork import scheduler
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
from cowork.db.session import get_open_session
from cowork.principal import Principal
from cowork.services.conversations import ConversationService
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.schedules import ScheduleRunService
from tests.test_schedule_run_outcomes import SLOT, _delete_schedule, _make_schedule, _poll, _run, _schedule

ORG = "11111111-1111-1111-1111-111111111111"
OWNER = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


@pytest.fixture(autouse=True)
def _fresh_scheduler(monkeypatch):
    monkeypatch.setattr(scheduler, "_scheduler_task", None)
    monkeypatch.setattr(scheduler, "_scheduled_runs", set())
    monkeypatch.setattr(scheduler, "_admission", scheduler._Admission(scheduler._runs_per_process))


@pytest.fixture
def schedule_id():
    created = _make_schedule()
    yield created
    _delete_schedule(created)


@pytest.fixture
def pushes(monkeypatch):
    """Every scheduled-run push, and the run's owner as an org principal."""
    import cowork.services.push.hooks as hooks

    sent: list[dict] = []
    monkeypatch.setattr(hooks, "scheduled_run_ended", lambda **kwargs: sent.append(kwargs))
    monkeypatch.setattr(
        scheduler, "_principal_for_schedule", lambda schedule: Principal(user_id=OWNER, org_id=ORG),
    )
    return sent


@pytest.fixture
def turns(monkeypatch):
    import cowork.handlers.responses as responses_mod

    ending = SimpleNamespace(reason="completed", acted=True)

    class Handler:
        def __init__(self, *, principal=None, interactive=True, schedule_id=None):
            assert interactive is False, "a scheduled turn is never interactive"

        async def handle(self, request):
            async def stream():
                if False:
                    yield

            return stream()

    async def evidence(_conversation_id):
        return scheduler._TurnEvidence(reason=ending.reason, acted=ending.acted)

    monkeypatch.setattr(responses_mod, "ResponsesHandler", Handler)
    monkeypatch.setattr(scheduler, "_turn_evidence", evidence)
    return ending


def _conversation_of_last_run(schedule_id: UUID) -> str | None:
    session = get_open_session()
    try:
        (run, *_) = ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE)).list_runs(schedule_id)
        return str(run.conversation_id) if run.conversation_id else None
    finally:
        session.close()


@pytest.mark.parametrize(("reason", "succeeded"), [("completed", True), ("error", False), ("interrupted", False)])
async def test_a_run_pushes_how_it_ended(schedule_id, turns, pushes, reason, succeeded):
    turns.reason = reason
    (slot,) = _poll(schedule_id)
    await _run(slot)
    assert pushes == [{
        "org_id": ORG, "user_id": OWNER, "schedule_name": _schedule(schedule_id).title,
        "conversation_id": _conversation_of_last_run(schedule_id), "succeeded": succeeded,
    }]
    assert pushes[0]["conversation_id"] is not None


async def test_a_cancelled_run_pushes_nothing(schedule_id, turns, pushes):
    turns.reason = "cancelled"
    (slot,) = _poll(schedule_id)
    await _run(slot)
    assert pushes == []


async def test_a_trial_run_pushes_nothing(schedule_id, turns, pushes):
    await scheduler.execute_schedule(schedule_id, is_manual=True, trial=True)
    assert pushes == []


async def test_a_manual_run_pushes(schedule_id, turns, pushes):
    await scheduler.execute_schedule(schedule_id, is_manual=True)
    assert [push["succeeded"] for push in pushes] == [True]


async def test_a_run_that_fails_before_its_turn_pushes_didnt_finish(schedule_id, turns, pushes, monkeypatch):
    import cowork.services.product_permissions as permissions

    async def refused(*_args, **_kwargs):
        raise RuntimeError("could not read the task's settings")

    monkeypatch.setattr(permissions, "require_product_permission", refused)
    (slot,) = _poll(schedule_id)
    await _run(slot)
    assert [push["succeeded"] for push in pushes] == [False]


async def test_a_passing_outage_that_leaves_the_slot_due_pushes_nothing(schedule_id, turns, pushes, monkeypatch):
    import cowork.services.product_permissions as permissions

    async def unavailable(*_args, **_kwargs):
        raise permissions.ProductPermissionUnavailable()

    monkeypatch.setattr(permissions, "require_product_permission", unavailable)
    (slot,) = _poll(schedule_id)
    await _run(slot)
    assert pushes == []


async def test_a_failing_push_never_fails_the_run(schedule_id, turns, monkeypatch):
    import cowork.services.push.hooks as hooks

    def broken(**_kwargs):
        raise RuntimeError("push down")

    monkeypatch.setattr(hooks, "scheduled_run_ended", broken)
    monkeypatch.setattr(
        scheduler, "_principal_for_schedule", lambda schedule: Principal(user_id=OWNER, org_id=ORG),
    )
    (slot,) = _poll(schedule_id)
    await _run(slot)
    from tests.test_schedule_run_outcomes import _runs

    (run,) = _runs(schedule_id)
    assert run.status == "success"


def _reap(schedule_id: UUID, *, claimed: bool, trial: bool = False) -> list[scheduler.ReapedRunPush]:
    with get_open_session() as session:
        scoped = ScopedSession(session, SYSTEM_SCOPE)
        runs = ScheduleRunService(scoped)
        run = runs.create_run(schedule_id, slot_at=None if trial else SLOT, lease_seconds=-60,
                              is_manual=trial, trial=trial)
        if claimed:
            conversation = ConversationService(scoped).create_conversation(
                topic=_schedule(schedule_id).title, project_id=GENERAL_PROJECT_ID,
            )
            assert runs.set_run_conversation(run.id, conversation.id)
    pushes: list[scheduler.ReapedRunPush] = []
    with get_open_session() as session:
        scheduler._poll(ScopedSession(session, SYSTEM_SCOPE), pushes=pushes)
    return [push for push in pushes if push.schedule_name == _schedule(schedule_id).title]


async def test_a_reaped_run_that_may_have_run_is_pushed_by_the_reaper(schedule_id):
    (push,) = _reap(schedule_id, claimed=True)
    assert push.schedule_name == _schedule(schedule_id).title
    assert push.conversation_id is not None


async def test_a_reaped_run_that_never_started_pushes_nothing(schedule_id):
    assert _reap(schedule_id, claimed=False) == []


async def test_a_reaped_trial_pushes_nothing(schedule_id):
    assert _reap(schedule_id, claimed=True, trial=True) == []


async def test_the_poll_loop_sends_the_reapers_pushes(monkeypatch):
    import cowork.services.push.hooks as hooks

    sent: list[dict] = []
    monkeypatch.setattr(hooks, "scheduled_run_ended", lambda **kwargs: sent.append(kwargs))
    scheduler._push_reaped_run(scheduler.ReapedRunPush(
        org_id=ORG, user_id=OWNER, schedule_name="Daily", conversation_id="c1",
    ))
    assert sent == [{
        "org_id": ORG, "user_id": OWNER, "schedule_name": "Daily", "conversation_id": "c1", "succeeded": False,
    }]
