"""The notification inbox (cowork.services.notices), and the notices a
scheduled run posts when it ends.

A failed run, or one that stopped before finishing, posts a notice by
default; a successful run posts one only when its owner opted in. A trial
and a cancelled run post nothing. Runs a reaper fails post one too. Each run
posts at most one notice.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
from sqlmodel import select

from cowork import scheduler
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession, TenantScope
from cowork.db.session import get_open_session
from cowork.models.notice import Notice
from cowork.services import notices
from cowork.services.conversations import ConversationService
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.schedules import ScheduleRunService
from cowork.services.settings import SettingService
from tests.test_notice_dedup_migration import notice_postgres  # noqa: F401
from tests.test_schedule_run_outcomes import SLOT, _delete_schedule, _make_schedule, _poll, _run, _schedule

ORG = "11111111-1111-1111-1111-111111111111"
ALICE = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
BOB = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


@pytest.fixture(autouse=True)
def _fresh_scheduler(monkeypatch):
    monkeypatch.setattr(scheduler, "_scheduler_task", None)
    monkeypatch.setattr(scheduler, "_scheduled_runs", set())
    monkeypatch.setattr(scheduler, "_admission", scheduler._Admission(scheduler._runs_per_process))


@pytest.fixture
def schedule_id():
    created = _make_schedule()
    yield created
    _delete_notices(subject=_schedule(created).title)
    _delete_schedule(created)


@pytest.fixture
def turns(monkeypatch):
    import cowork.handlers.responses as responses_mod

    ending = SimpleNamespace(reason="completed", acted=True)

    class Handler:
        def __init__(self, *, principal=None, interactive=True, schedule_id=None):
            pass

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


@pytest.fixture
def notify_succeeded():
    """Turn successful-run notices on for the desktop's (unscoped) settings."""
    with get_open_session() as session:
        SettingService(session).upsert_setting("schedule_notify_succeeded", "true")
    yield
    with get_open_session() as session:
        SettingService(session).delete_setting("schedule_notify_succeeded")


def _delete_notices(**where) -> None:
    with get_open_session() as session:
        stmt = select(Notice)
        for column, value in where.items():
            stmt = stmt.where(getattr(Notice, column) == value)
        for notice in session.exec(stmt).all():
            session.delete(notice)
        session.commit()


def _notices_for(schedule_id: UUID) -> list[Notice]:
    with get_open_session() as session:
        return list(session.exec(
            select(Notice).where(Notice.subject == _schedule(schedule_id).title).order_by(Notice.created_at)
        ).all())


# --- the inbox --------------------------------------------------------------


def _org_scoped(session, user_id: str) -> ScopedSession:
    return ScopedSession(session, TenantScope(org_mode=True, org_id=ORG, user_id=user_id))


def _post(session, *, user_id: str | None = ALICE, org_id: str | None = ORG, subject: str = "inbox test", **kwargs):
    return notices.post_notice(
        session, org_id=org_id, user_id=user_id, kind="test.kind", subject=subject, target={"x": 1}, **kwargs,
    )


def test_a_person_sees_only_their_own_notices():
    subject = f"own {uuid4()}"
    try:
        with get_open_session() as session:
            mine = _post(session, user_id=ALICE, subject=subject)
            _post(session, user_id=BOB, subject=subject)
            _post(session, user_id=ALICE, org_id=str(uuid4()), subject=subject)
        with get_open_session() as session:
            alice = notices.NoticeService(_org_scoped(session, ALICE))
            listed = [n for n in alice.list_notices() if n.subject == subject]
            assert [n.id for n in listed] == [mine.id]
        with get_open_session() as session:
            bob = notices.NoticeService(_org_scoped(session, BOB))
            assert [n.user_id for n in bob.list_notices() if n.subject == subject] == [BOB]
            # Alice's notice can't be marked read by anyone else.
            assert bob.mark_read(mine.id) is False
    finally:
        _delete_notices(subject=subject)


def test_read_state_is_the_same_from_every_client():
    subject = f"read {uuid4()}"
    try:
        with get_open_session() as session:
            first = _post(session, subject=subject)
            _post(session, subject=subject)
        with get_open_session() as session:
            client_a = notices.NoticeService(_org_scoped(session, ALICE))
            before = client_a.unread_count()
            assert client_a.mark_read(first.id) is True
        with get_open_session() as session:
            other_client = notices.NoticeService(_org_scoped(session, ALICE))
            assert other_client.unread_count() == before - 1
            assert other_client.mark_all_read() >= 1
            assert other_client.unread_count() == 0
    finally:
        _delete_notices(subject=subject)


def test_old_notices_and_those_past_the_newest_are_pruned(monkeypatch):
    monkeypatch.setattr(notices, "KEEP_PER_PERSON", 3)
    person = str(uuid4())
    try:
        with get_open_session() as session:
            stale = _post(session, user_id=person, now=datetime.now(timezone.utc) - timedelta(days=31))
            for _ in range(3):
                _post(session, user_id=person)
            survivors = session.exec(select(Notice).where(Notice.user_id == person)).all()
            assert len(survivors) == 3
            assert stale.id not in {n.id for n in survivors}
            newest = _post(session, user_id=person)
            survivors = session.exec(select(Notice).where(Notice.user_id == person)).all()
            assert len(survivors) == 3
            assert newest.id in {n.id for n in survivors}
    finally:
        _delete_notices(user_id=person)


def _notices_client() -> httpx.AsyncClient:
    """The notices routes alone, as the app mounts them."""
    from fastapi import FastAPI

    from cowork.api.v1.endpoints.notices import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/notices")
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def test_the_app_mounts_the_notices_routes():
    from cowork.api.v1.router import api_router

    paths = {route.path for route in api_router.routes}
    assert {
        "/api/v1/notices/", "/api/v1/notices/unread-count", "/api/v1/notices/read-all",
        "/api/v1/notices/{notice_id}/read",
    } <= paths


async def test_the_api_lists_counts_and_marks_notices_read():
    subject = f"api {uuid4()}"
    try:
        with get_open_session() as session:
            posted = notices.post_notice(
                session, org_id=None, user_id=None, kind="test.kind", subject=subject, target={"runId": "r1"},
            )
        async with _notices_client() as client:
            listed = (await client.get("/api/v1/notices/")).json()
            (shown,) = [n for n in listed["notices"] if n["subject"] == subject]
            assert shown["kind"] == "test.kind"
            assert shown["target"] == {"runId": "r1"}
            assert shown["readAt"] is None
            unread = (await client.get("/api/v1/notices/unread-count")).json()["unreadCount"]
            assert unread == listed["unreadCount"] >= 1

            assert (await client.post(f"/api/v1/notices/{posted.id}/read")).status_code == 204
            assert (await client.get("/api/v1/notices/unread-count")).json()["unreadCount"] == unread - 1
            assert (await client.post(f"/api/v1/notices/{uuid4()}/read")).status_code == 404

            await client.post("/api/v1/notices/read-all")
            assert (await client.get("/api/v1/notices/unread-count")).json()["unreadCount"] == 0
    finally:
        _delete_notices(subject=subject)


# --- scheduled runs ---------------------------------------------------------


@pytest.mark.parametrize("reason", ["error", "interrupted"])
async def test_a_failed_run_posts_one_notice_naming_the_task(schedule_id, turns, reason):
    turns.reason = reason
    (slot,) = _poll(schedule_id)
    await _run(slot)
    (notice,) = _notices_for(schedule_id)
    assert notice.kind == notices.SCHEDULED_RUN_FAILED
    assert notice.subject == _schedule(schedule_id).title
    assert notice.target["scheduleId"] == str(schedule_id)
    assert notice.target["runId"] == notice.source_id
    assert notice.target["conversationId"] is not None
    assert notice.read_at is None


async def test_a_successful_run_posts_nothing_by_default(schedule_id, turns):
    (slot,) = _poll(schedule_id)
    await _run(slot)
    assert _notices_for(schedule_id) == []


async def test_a_successful_run_posts_a_notice_when_its_owner_opted_in(schedule_id, turns, notify_succeeded):
    (slot,) = _poll(schedule_id)
    await _run(slot)
    (notice,) = _notices_for(schedule_id)
    assert notice.kind == notices.SCHEDULED_RUN_SUCCEEDED


async def test_failures_post_nothing_when_turned_off(schedule_id, turns):
    with get_open_session() as session:
        SettingService(session).upsert_setting("schedule_notify_failed", "false")
    try:
        turns.reason = "error"
        (slot,) = _poll(schedule_id)
        await _run(slot)
        assert _notices_for(schedule_id) == []
    finally:
        with get_open_session() as session:
            SettingService(session).delete_setting("schedule_notify_failed")


async def test_a_cancelled_run_posts_nothing(schedule_id, turns):
    turns.reason = "cancelled"
    (slot,) = _poll(schedule_id)
    await _run(slot)
    assert _notices_for(schedule_id) == []


async def test_a_trial_run_posts_nothing(schedule_id, turns):
    turns.reason = "error"
    await scheduler.execute_schedule(schedule_id, is_manual=True, trial=True)
    assert _notices_for(schedule_id) == []


async def test_a_run_stopped_at_a_limit_posts_a_stopped_notice(schedule_id):
    with get_open_session() as session:
        run = ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE)).create_run(schedule_id, slot_at=SLOT)
        run.status, run.outcome = "failed", "stopped_at_limit"
        session.add(run)
        session.commit()
        run_id = run.id
    with get_open_session() as session:
        notices.post_scheduled_run_notice(ScopedSession(session, SYSTEM_SCOPE), run_id=run_id)
    (notice,) = _notices_for(schedule_id)
    assert notice.kind == notices.SCHEDULED_RUN_STOPPED


async def test_a_run_posts_at_most_one_notice(schedule_id, turns):
    turns.reason = "error"
    (slot,) = _poll(schedule_id)
    await _run(slot)
    (notice,) = _notices_for(schedule_id)
    with get_open_session() as session:
        assert notices.post_scheduled_run_notice(
            ScopedSession(session, SYSTEM_SCOPE), run_id=UUID(notice.source_id),
        ) is None
    assert len(_notices_for(schedule_id)) == 1


async def test_a_failing_notice_never_fails_the_run(schedule_id, turns, monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("inbox down")

    monkeypatch.setattr(notices, "post_scheduled_run_notice", broken)
    turns.reason = "error"
    (slot,) = _poll(schedule_id)
    await _run(slot)
    from tests.test_schedule_run_outcomes import _runs

    (run,) = _runs(schedule_id)
    assert run.status == "failed"


def _claimed_run(schedule_id: UUID, *, lease_seconds: int, trial: bool = False) -> UUID:
    with get_open_session() as session:
        scoped = ScopedSession(session, SYSTEM_SCOPE)
        runs = ScheduleRunService(scoped)
        run = runs.create_run(schedule_id, slot_at=None if trial else SLOT, lease_seconds=lease_seconds,
                              is_manual=trial, trial=trial)
        conversation = ConversationService(scoped).create_conversation(
            topic=_schedule(schedule_id).title, project_id=GENERAL_PROJECT_ID,
        )
        assert runs.set_run_conversation(run.id, conversation.id)
        return run.id


async def test_a_run_the_lease_reaper_fails_posts_a_notice(schedule_id):
    run_id = _claimed_run(schedule_id, lease_seconds=-60)
    with get_open_session() as session:
        scheduler._poll(ScopedSession(session, SYSTEM_SCOPE))
    (notice,) = _notices_for(schedule_id)
    assert notice.kind == notices.SCHEDULED_RUN_FAILED
    assert notice.source_id == str(run_id)


async def test_a_reaped_trial_posts_nothing(schedule_id):
    _claimed_run(schedule_id, lease_seconds=-60, trial=True)
    with get_open_session() as session:
        scheduler._poll(ScopedSession(session, SYSTEM_SCOPE))
    assert _notices_for(schedule_id) == []


async def test_a_run_the_boot_reaper_fails_posts_a_notice(schedule_id):
    run_id = _claimed_run(schedule_id, lease_seconds=600)
    executed: list = []
    with get_open_session() as session:
        scoped = ScopedSession(session, SYSTEM_SCOPE)
        ScheduleRunService(scoped).reap_orphaned_runs(executed=executed)
        scheduler.post_reaped_run_notices(scoped, [run.id for run in executed])
    assert run_id in {run.id for run in executed}
    (notice,) = _notices_for(schedule_id)
    assert notice.source_id == str(run_id)


def _race_notice_writers(session_factory, schedule_id, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    with session_factory() as session:
        run = ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE)).create_run(schedule_id, slot_at=SLOT)
        run.status, run.outcome = "failed", "interrupted"
        session.add(run)
        session.commit()
        run_id = run.id
    barrier = Barrier(2)

    def wants(*args):
        barrier.wait(timeout=10)
        return True

    monkeypatch.setattr(notices, "_wants", wants)

    def post(_):
        with session_factory() as session:
            result = notices.post_scheduled_run_notice(ScopedSession(session, SYSTEM_SCOPE), run_id=run_id)
            # The losing session must still be usable after the conflict.
            session.exec(select(Notice.id)).all()
            return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(post, range(2)))
    assert sum(result is not None for result in results) == 1
    with session_factory() as session:
        assert len(session.exec(select(Notice).where(Notice.source_id == str(run_id))).all()) == 1


def test_concurrent_reapers_post_only_once(schedule_id, monkeypatch):
    _race_notice_writers(get_open_session, schedule_id, monkeypatch)


@pytest.mark.postgres_service
def test_concurrent_reapers_post_only_once_postgres(notice_postgres, monkeypatch):  # noqa: F811
    from sqlmodel import Session
    from cowork.models.project import Project
    from cowork.services.schedules import ScheduleService

    with Session(notice_postgres) as session:
        project = Project(name="notice-race", path="/notice-race")
        session.add(project)
        session.commit()
        schedule = ScheduleService(ScopedSession(session, SYSTEM_SCOPE)).create_schedule(
            title="notice race", prompt="p", cadence="daily", next_run_at=SLOT,
            model="default", timezone="UTC", project_id=project.id, enabled=True,
        )
        schedule_id = schedule.id
    _race_notice_writers(lambda: Session(notice_postgres), schedule_id, monkeypatch)
