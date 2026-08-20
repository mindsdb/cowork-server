from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel

from cowork.models.project import Project
from cowork.models.schedule import Schedule
from cowork.schemas.schedules import RunStatus
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
from cowork.services.schedules import ScheduleRunService, ScheduleService


def _session():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    session = Session(engine)
    session.add(Project(id=GENERAL_PROJECT_ID, name="general", path="/general"))
    session.commit()
    return session


def _schedule(session: Session) -> Schedule:
    schedule = Schedule(
        title="Daily report",
        prompt="Summarize",
        cadence="daily",
        next_run_at=datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc),
        model="default",
        project_id=GENERAL_PROJECT_ID,
    )
    session.add(schedule)
    session.commit()
    session.refresh(schedule)
    return schedule


def test_has_running_run_false_when_no_runs():
    session = _session()
    schedule = _schedule(session)
    assert ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE)).has_running_run(schedule.id) is False


def test_has_running_run_true_for_scheduled_run_in_progress():
    session = _session()
    schedule = _schedule(session)
    run_service = ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE))
    run = run_service.create_run(schedule.id, is_manual=False)

    assert run_service.has_running_run(schedule.id) is True

    run_service.finish_run(run.id)
    assert run_service.has_running_run(schedule.id) is False


def test_has_running_run_ignores_manual_run_in_progress():
    session = _session()
    schedule = _schedule(session)
    run_service = ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE))
    run = run_service.create_run(schedule.id, is_manual=True)

    assert run_service.has_running_run(schedule.id) is False

    run_service.finish_run(run.id)
    assert run_service.has_running_run(schedule.id) is False


# --- ENG-688: freshness guard — a due cron slot is skipped when a successful
# run (typically a manual "run now") finished within the cadence window.

def _finish_at(session: Session, run_id, when: datetime) -> None:
    from cowork.models.schedule import ScheduleRun

    run = session.get(ScheduleRun, run_id)
    run.finished_at = when
    session.add(run)
    session.commit()


def test_last_successful_finish_ignores_failures_and_running():
    session = _session()
    schedule = _schedule(session)
    run_service = ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE))

    assert run_service.last_successful_finish(schedule.id) is None

    failed = run_service.create_run(schedule.id, is_manual=True)
    run_service.finish_run(failed.id, error="boom")
    run_service.create_run(schedule.id, is_manual=False)  # still running
    assert run_service.last_successful_finish(schedule.id) is None

    ok = run_service.create_run(schedule.id, is_manual=True)
    run_service.finish_run(ok.id)
    finished = run_service.last_successful_finish(schedule.id)
    assert finished is not None and finished.tzinfo is not None


def test_due_slot_skipped_and_advanced_after_recent_manual_success():
    from cowork.scheduler import _due_schedules

    session = _session()
    schedule = _schedule(session)  # daily, due at 2026-06-25 09:00 UTC
    run_service = ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE))

    now = datetime(2026, 6, 25, 9, 0, 30, tzinfo=timezone.utc)
    run = run_service.create_run(schedule.id, is_manual=True)
    run_service.finish_run(run.id)
    _finish_at(session, run.id, now - timedelta(minutes=50))

    assert _due_schedules(ScopedSession(session, SYSTEM_SCOPE), now) == []
    session.refresh(schedule)
    # Slot consumed: advanced past the skipped occurrence to the next day.
    assert schedule.next_run_at.replace(tzinfo=timezone.utc) > now


def test_due_slot_runs_when_last_success_is_old():
    from cowork.scheduler import _due_schedules

    session = _session()
    schedule = _schedule(session)
    run_service = ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE))

    now = datetime(2026, 6, 25, 9, 0, 30, tzinfo=timezone.utc)
    run = run_service.create_run(schedule.id, is_manual=True)
    run_service.finish_run(run.id)
    _finish_at(session, run.id, now - timedelta(hours=2))

    assert [s.id for s in _due_schedules(ScopedSession(session, SYSTEM_SCOPE), now)] == [schedule.id]


def test_due_slot_deferred_while_manual_run_in_flight():
    """PR #181 review issue 3: a manual run still executing when the cron
    slot comes due must block the slot — otherwise both publish the same
    output. Deferred, not consumed: the slot stays due, and once the manual
    run finishes the freshness guard decides whether it still fires."""
    from cowork.scheduler import _due_schedules

    session = _session()
    schedule = _schedule(session)  # daily, due at 2026-06-25 09:00 UTC
    run_service = ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE))
    run_service.create_run(schedule.id, is_manual=True)  # still running

    now = datetime(2026, 6, 25, 9, 0, 30, tzinfo=timezone.utc)
    assert _due_schedules(ScopedSession(session, SYSTEM_SCOPE), now) == []
    session.refresh(schedule)
    assert schedule.next_run_at.replace(tzinfo=timezone.utc) == datetime(
        2026, 6, 25, 9, 0, tzinfo=timezone.utc
    )


def test_due_slot_runs_when_recent_run_failed():
    from cowork.scheduler import _due_schedules

    session = _session()
    schedule = _schedule(session)
    run_service = ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE))

    now = datetime(2026, 6, 25, 9, 0, 30, tzinfo=timezone.utc)
    run = run_service.create_run(schedule.id, is_manual=True)
    run_service.finish_run(run.id, error="boom")
    _finish_at(session, run.id, now - timedelta(minutes=10))

    assert [s.id for s in _due_schedules(ScopedSession(session, SYSTEM_SCOPE), now)] == [schedule.id]


# --- ENG-688: cancelled-run status + the UI-facing "running" flag.

def test_finish_run_status_override_records_cancelled():
    from cowork.schemas.schedules import RunStatus

    session = _session()
    schedule = _schedule(session)
    run_service = ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE))

    run = run_service.create_run(schedule.id, is_manual=False)
    finished = run_service.finish_run(run.id, status=RunStatus.cancelled)
    assert finished.status == RunStatus.cancelled
    assert finished.error is None
    # A cancelled run is not a success: it neither blocks via the freshness
    # guard nor counts as the last successful finish.
    assert run_service.last_successful_finish(schedule.id) is None


def test_has_active_run_counts_manual_runs():
    session = _session()
    schedule = _schedule(session)
    run_service = ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE))

    assert run_service.has_active_run(schedule.id) is False
    run = run_service.create_run(schedule.id, is_manual=True)
    # Manual runs are invisible to the cron-overlap guard but visible here.
    assert run_service.has_running_run(schedule.id) is False
    assert run_service.has_active_run(schedule.id) is True

    run_service.finish_run(run.id)
    assert run_service.has_active_run(schedule.id) is False


# --- ENG-688: schedule/run identity stamped on the turn's trace.

def test_execute_schedule_stamps_trace_identity(monkeypatch):
    import asyncio

    import cowork.handlers.responses as responses_mod
    from cowork.db.session import get_open_session
    from cowork.scheduler import execute_schedule
    from cowork.services.schedules import ScheduleService

    captured: list = []

    class FakeHandler:
        def __init__(self, session, principal=None):
            pass

        async def handle(self, request):
            captured.append(request)

            async def _gen():
                if False:
                    yield

            return _gen()

    monkeypatch.setattr(responses_mod, "ResponsesHandler", FakeHandler)

    session = get_open_session()
    schedule = ScheduleService(ScopedSession(session, SYSTEM_SCOPE)).create_schedule(
        title="trace stamp test",
        prompt="do the thing",
        cadence="daily",
        next_run_at=datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc),
        model="default",
        timezone="UTC",
        project_id=GENERAL_PROJECT_ID,
        enabled=True,
    )
    schedule_id = schedule.id
    session.close()

    try:
        asyncio.run(execute_schedule(schedule_id, is_manual=False))
        asyncio.run(execute_schedule(schedule_id, is_manual=True))

        cron_req, manual_req = captured
        assert cron_req.trace_tags == ["scheduled_task", "trigger:cron"]
        assert manual_req.trace_tags == ["scheduled_task", "trigger:manual"]
        for req, trigger in ((cron_req, "cron"), (manual_req, "manual")):
            assert req.trace_metadata["schedule_id"] == str(schedule_id)
            assert req.trace_metadata["trigger_type"] == trigger
            assert req.trace_metadata["schedule_run_id"]
    finally:
        s = get_open_session()
        ScheduleService(ScopedSession(s, SYSTEM_SCOPE)).delete_schedule(schedule_id)
        s.close()


# A scheduled run in org mode has no request, so it derives a service principal
# from the schedule row: the conversation is created under the owning org and
# the turn receives that principal so the remote backend can mint the org's key
# headlessly.

def test_execute_schedule_uses_service_principal_in_org_mode(monkeypatch):
    import asyncio

    import cowork.handlers.responses as responses_mod
    from cowork.common.settings.app_settings import get_app_settings
    from cowork.db.scoped import ScopedSession, TenantScope
    from cowork.db.session import get_open_session
    from cowork.models.conversation import Conversation
    from cowork.scheduler import execute_schedule
    from cowork.services.schedules import ScheduleService

    org_id = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    user_id = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()

    captured: dict = {}

    class FakeHandler:
        def __init__(self, session, principal=None):
            captured["principal"] = principal

        async def handle(self, request):
            async def _gen():
                if False:
                    yield

            return _gen()

    monkeypatch.setattr(responses_mod, "ResponsesHandler", FakeHandler)

    org_scope = TenantScope(org_mode=True, org_id=org_id, user_id=user_id)
    session = get_open_session()
    scoped = ScopedSession(session, org_scope)
    project = Project(name="p-org", path="/tmp/p-org")
    scoped.add(project)
    scoped.commit()
    scoped.refresh(project)
    schedule = ScheduleService(scoped).create_schedule(
        title="org daily",
        prompt="do it",
        cadence="daily",
        next_run_at=datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc),
        model="default",
        project_id=project.id,
    )
    schedule_id = schedule.id
    # The request that created the schedule stamped its identity on the row.
    assert schedule.org_id == org_id and schedule.created_by == user_id
    session.close()

    try:
        # Cron path (conversation_id=None).
        asyncio.run(execute_schedule(schedule_id, is_manual=False))

        principal = captured["principal"]
        assert principal is not None, "the turn must receive a service principal"
        assert principal.org_id == org_id
        assert principal.user_id == user_id

        # The conversation was created under the owning org, not as an invisible
        # NULL-org row.
        s = get_open_session()
        convs = ScopedSession(s, org_scope)
        rows = convs.exec(convs.select(Conversation)).all()
        assert any(c.org_id == org_id for c in rows)
        s.close()
    finally:
        s = get_open_session()
        ScheduleService(ScopedSession(s, org_scope)).delete_schedule(schedule_id)
        s.close()
        get_app_settings.cache_clear()


# A corrupt org-mode row (NULL org_id) can never resolve an identity, so it can
# never run. It must be disabled, not left due and re-fired every poll.

def test_execute_schedule_disables_corrupt_org_row_instead_of_looping(monkeypatch):
    import asyncio

    from sqlmodel import select

    from cowork.common.datetime_utils import ensure_utc
    from cowork.common.settings.app_settings import get_app_settings
    from cowork.db.scoped import ScopedSession, TenantScope
    from cowork.db.session import get_open_session
    from cowork.models.schedule import ScheduleRun
    from cowork.scheduler import execute_schedule
    from cowork.services.schedules import ScheduleService

    org_id = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    user_id = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"
    original_next_run = datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc)

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()

    org_scope = TenantScope(org_mode=True, org_id=org_id, user_id=user_id)
    session = get_open_session()
    scoped = ScopedSession(session, org_scope)
    project = Project(name="p-corrupt", path="/tmp/p-corrupt")
    scoped.add(project)
    scoped.commit()
    scoped.refresh(project)
    schedule = ScheduleService(scoped).create_schedule(
        title="corrupt daily",
        prompt="do it",
        cadence="daily",
        next_run_at=original_next_run,
        model="default",
        project_id=project.id,
    )
    schedule_id = schedule.id
    session.close()

    # Corrupt the row: NULL the org on a raw session so the scoped before-flush
    # listener doesn't re-stamp it.
    raw = get_open_session()
    row = raw.get(Schedule, schedule_id)
    row.org_id = None
    raw.add(row)
    raw.commit()
    raw.close()

    try:
        # One call is enough to observe the disable and the recorded failed run.
        asyncio.run(execute_schedule(schedule_id, is_manual=False))

        s = get_open_session()
        after = s.get(Schedule, schedule_id)
        assert after.enabled is False, "corrupt row must be disabled, not left due"
        # Not advanced — disabling is what stops the re-fire, not a moved slot.
        assert ensure_utc(after.next_run_at) == original_next_run
        runs = s.exec(
            select(ScheduleRun).where(ScheduleRun.schedule_id == schedule_id)
        ).all()
        assert runs, "the attempt must still record a run"
        assert all(r.status == RunStatus.failed for r in runs)
        s.close()
    finally:
        s = get_open_session()
        # org_id is NULL now, so clean up on a raw session.
        for r in s.exec(
            select(ScheduleRun).where(ScheduleRun.schedule_id == schedule_id)
        ).all():
            s.delete(r)
        dead = s.get(Schedule, schedule_id)
        if dead is not None:
            s.delete(dead)
        s.commit()
        s.close()
        get_app_settings.cache_clear()


# --- ENG-688: how the run actually ended comes from the stream buffer's
# terminal record. The producer runs detached and swallows its own
# cancellation (task.cancelled() stays False), so the terminal record is the
# only truthful signal — without it a cancelled or failed run is recorded as
# success.

def test_turn_terminal_reason_reads_the_terminal_record(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace

    import cowork.scheduler as scheduler_mod
    from cowork.streaming.buffer import FileStreamBuffer

    buf = FileStreamBuffer(tmp_path / "turn.jsonl")

    async def _fill():
        await buf.append("sse", {"sse": "event: response.created"})
        await buf.close("cancelled")

    asyncio.run(_fill())
    monkeypatch.setattr(
        scheduler_mod, "registry",
        SimpleNamespace(get=lambda cid: SimpleNamespace(buffer=buf)),
    )
    assert asyncio.run(scheduler_mod._turn_terminal_reason("c1")) == "cancelled"


def test_turn_terminal_reason_with_real_registry_cancel(tmp_path):
    """End-to-end through the real registry: cancel a producer that swallows
    its CancelledError the way handlers/responses._produce does. The task
    ends NOT-cancelled — which is exactly why task state can't be the
    signal — while the buffer terminal record says "cancelled"."""
    import asyncio

    import cowork.scheduler as scheduler_mod
    from cowork.streaming.buffer import FileStreamBuffer
    from cowork.streaming.registry import registry

    buf = FileStreamBuffer(tmp_path / "turn.jsonl")
    conversation_id = "eng688-real-cancel-test"

    async def main():
        started = asyncio.Event()

        async def producer():
            try:
                await buf.append("sse", {"sse": "event: response.created"})
                started.set()
                await asyncio.sleep(30)
                await buf.close("completed")
            except asyncio.CancelledError:
                await buf.close("cancelled")
                return

        handle = await registry.start(
            conversation_id=conversation_id,
            turn_id=0,
            buffer=buf,
            producer_coro=producer(),
        )
        await started.wait()
        await registry.cancel(conversation_id)
        assert handle.task.done()
        assert handle.task.cancelled() is False  # the swallowed cancel
        return await scheduler_mod._turn_terminal_reason(conversation_id)

    assert asyncio.run(main()) == "cancelled"


def test_turn_terminal_reason_none_without_handle(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    import cowork.scheduler as scheduler_mod

    monkeypatch.setattr(scheduler_mod, "registry", SimpleNamespace(get=lambda cid: None))
    assert asyncio.run(scheduler_mod._turn_terminal_reason("c1")) is None


def _execute_with_terminal(monkeypatch, reason, *, is_manual=False):
    """Run execute_schedule with a no-op turn and a forced terminal reason;
    return the resulting run/schedule state as plain values."""
    import asyncio

    import cowork.handlers.responses as responses_mod
    import cowork.scheduler as scheduler_mod
    from cowork.common.datetime_utils import ensure_utc
    from cowork.db.session import get_open_session
    from cowork.scheduler import execute_schedule
    from cowork.services.schedules import ScheduleService

    class FakeHandler:
        def __init__(self, session, principal=None):
            pass

        async def handle(self, request):
            async def _gen():
                if False:
                    yield

            return _gen()

    monkeypatch.setattr(responses_mod, "ResponsesHandler", FakeHandler)

    async def _terminal(_conversation_id):
        return reason

    monkeypatch.setattr(scheduler_mod, "_turn_terminal_reason", _terminal)

    session = get_open_session()
    schedule = ScheduleService(ScopedSession(session, SYSTEM_SCOPE)).create_schedule(
        title="terminal mapping test",
        prompt="do the thing",
        cadence="daily",
        next_run_at=datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc),
        model="default",
        timezone="UTC",
        project_id=GENERAL_PROJECT_ID,
        enabled=True,
    )
    schedule_id = schedule.id
    original_next = ensure_utc(schedule.next_run_at)
    session.close()

    try:
        asyncio.run(execute_schedule(schedule_id, is_manual=is_manual))
        check = get_open_session()
        fresh = ScheduleService(ScopedSession(check, SYSTEM_SCOPE)).get_schedule(schedule_id)
        run = ScheduleRunService(ScopedSession(check, SYSTEM_SCOPE)).list_runs(schedule_id)[0]
        state = {
            "run_status": run.status,
            "run_error": run.error,
            "run_conversation_id": run.conversation_id,
            "last_error": fresh.last_error,
            "last_run_at": fresh.last_run_at,
            "next_advanced": ensure_utc(fresh.next_run_at) > original_next,
        }
        check.close()
        return state
    finally:
        s = get_open_session()
        ScheduleService(ScopedSession(s, SYSTEM_SCOPE)).delete_schedule(schedule_id)
        s.close()


def test_execute_schedule_records_cancelled_and_consumes_slot(monkeypatch):
    from cowork.schemas.schedules import RunStatus

    state = _execute_with_terminal(monkeypatch, "cancelled")
    assert state["run_status"] == RunStatus.cancelled
    assert state["run_error"] is None
    assert state["last_error"] is None
    assert state["last_run_at"] is None
    # The slot is consumed — otherwise the next tick restarts the run the
    # user just killed (a cancelled run isn't a success, so the freshness
    # guard wouldn't block it).
    assert state["next_advanced"] is True


def test_execute_schedule_records_producer_error_as_failed(monkeypatch):
    from cowork.schemas.schedules import RunStatus

    state = _execute_with_terminal(monkeypatch, "error")
    assert state["run_status"] == RunStatus.failed
    assert state["run_error"]
    assert state["last_error"]
    assert state["next_advanced"] is True


def test_execute_schedule_completed_is_success(monkeypatch):
    from cowork.schemas.schedules import RunStatus

    state = _execute_with_terminal(monkeypatch, "completed")
    assert state["run_status"] == RunStatus.success
    assert state["run_error"] is None
    assert state["last_error"] is None
    assert state["last_run_at"] is not None
    assert state["next_advanced"] is True


def test_execute_schedule_links_conversation_before_turn_starts(monkeypatch):
    """The run's conversation is recorded as soon as it exists — not at
    finish — so the runs list can open a run that is still executing."""
    import asyncio

    import cowork.handlers.responses as responses_mod
    from cowork.db.session import get_open_session
    from cowork.scheduler import execute_schedule
    from cowork.services.schedules import ScheduleService

    session = get_open_session()
    schedule = ScheduleService(ScopedSession(session, SYSTEM_SCOPE)).create_schedule(
        title="early link test",
        prompt="do the thing",
        cadence="daily",
        next_run_at=datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc),
        model="default",
        timezone="UTC",
        project_id=GENERAL_PROJECT_ID,
        enabled=True,
    )
    schedule_id = schedule.id
    session.close()

    seen: dict = {}

    class FakeHandler:
        def __init__(self, session, principal=None):
            pass

        async def handle(self, request):
            check = get_open_session()
            run = ScheduleRunService(ScopedSession(check, SYSTEM_SCOPE)).list_runs(schedule_id)[0]
            seen["conversation_id_during_turn"] = (
                str(run.conversation_id) if run.conversation_id else None
            )
            seen["request_conversation"] = request.conversation
            check.close()

            async def _gen():
                if False:
                    yield

            return _gen()

    monkeypatch.setattr(responses_mod, "ResponsesHandler", FakeHandler)

    try:
        asyncio.run(execute_schedule(schedule_id, is_manual=False))
        assert seen["conversation_id_during_turn"] == seen["request_conversation"]
    finally:
        s = get_open_session()
        ScheduleService(ScopedSession(s, SYSTEM_SCOPE)).delete_schedule(schedule_id)
        s.close()


# --- ENG-769: reap orphaned `running` runs left by a crash/restart.

def test_reap_orphaned_runs_marks_running_as_failed():
    session = _session()
    schedule = _schedule(session)
    run_service = ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE))
    run = run_service.create_run(schedule.id, is_manual=False)

    # The stale `running` row would otherwise wedge the schedule forever.
    assert run_service.has_running_run(schedule.id) is True

    reaped = run_service.reap_orphaned_runs()

    assert reaped == 1
    assert run_service.has_running_run(schedule.id) is False

    session.refresh(run)
    assert run.status == RunStatus.failed
    assert run.error is not None
    assert run.finished_at is not None
    assert run.duration_ms is not None


def test_reap_orphaned_runs_reaps_manual_runs_too():
    session = _session()
    schedule = _schedule(session)
    run_service = ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE))
    manual = run_service.create_run(schedule.id, is_manual=True)

    assert run_service.reap_orphaned_runs() == 1

    session.refresh(manual)
    assert manual.status == RunStatus.failed


def test_reap_orphaned_runs_leaves_finished_runs_untouched():
    session = _session()
    schedule = _schedule(session)
    run_service = ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE))
    run = run_service.create_run(schedule.id, is_manual=False)
    run_service.finish_run(run.id)

    assert run_service.reap_orphaned_runs() == 0

    session.refresh(run)
    assert run.status == RunStatus.success
    assert run.error is None


def test_same_org_users_cannot_see_each_others_schedules(tmp_path, monkeypatch):
    """Staging audit P0: schedules are personal (created_by) but CRUD/list
    filtered by org only, so coworkers saw/paused/deleted each other's."""
    import pytest
    from sqlalchemy.pool import StaticPool
    from sqlmodel import Session, SQLModel, create_engine, select
    from cowork.db.scoped import ScopedSession, TenantScope
    from cowork.models.project import Project
    from cowork.services.schedules import ScheduleService
    from datetime import datetime, timezone

    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "p"))
    from cowork.common.settings.app_settings import get_app_settings
    get_app_settings.cache_clear()
    ORG = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(Project(name="proj", path="/tmp/p", org_id=ORG)); s.commit()
        pid = s.exec(select(Project).where(Project.name == "proj")).one().id

    def svc(user):
        return ScheduleService(ScopedSession(Session(eng), TenantScope(org_mode=True, org_id=ORG, user_id=user)))

    when = datetime(2030, 1, 1, tzinfo=timezone.utc)
    a = svc("alice").create_schedule(title="a", prompt="secret", cadence="daily",
                                     next_run_at=when, model="m", project_id=pid)
    assert a.id not in {x.id for x in svc("bob").list_schedules()}
    with pytest.raises(ValueError):
        svc("bob").get_schedule(a.id)
    assert svc("bob").delete_schedule(a.id) is False
    assert svc("alice").get_schedule(a.id).id == a.id
    get_app_settings.cache_clear()
