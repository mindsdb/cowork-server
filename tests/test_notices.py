"""The in-app notification inbox (cowork.services.notices).

A person sees and reads only their own notices, the oldest go past the limits,
and a scheduled run that didn't finish posts one to its schedule's owner on
every deployment, the desktop included.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlmodel import select

from cowork import scheduler
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession, TenantScope
from cowork.db.session import get_open_session
from cowork.models.notice import Notice
from cowork.services import notices
from cowork.services.notices import SCHEDULED_RUN_FAILED, NoticeService
from tests.test_push_scheduler import _fresh_scheduler, schedule_id, turns  # noqa: F401
from tests.test_schedule_run_outcomes import _client, _poll, _run, _schedule


def _org_scope(org_id: str, user_id: str) -> TenantScope:
    return TenantScope(org_mode=True, org_id=org_id, user_id=user_id)


def _with(scope: TenantScope, fn):
    session = get_open_session()
    try:
        return fn(NoticeService(ScopedSession(session, scope)))
    finally:
        session.close()


def _post(scope: TenantScope, title: str = "Weekly report") -> Notice:
    return _with(scope, lambda s: s.post(kind=SCHEDULED_RUN_FAILED, title=title, data={"run_id": "r"}))


def test_a_person_sees_and_reads_only_their_own():
    org, other_org = str(uuid4()), str(uuid4())
    me, colleague = _org_scope(org, str(uuid4())), _org_scope(org, str(uuid4()))
    stranger = _org_scope(other_org, str(uuid4()))
    mine = _post(me, "Mine")
    theirs = _post(colleague, "Theirs")
    _post(stranger, "Elsewhere")

    assert [n.title for n in _with(me, lambda s: s.list_notices())] == ["Mine"]
    assert _with(me, lambda s: s.unread_count()) == 1
    # Someone else's notice can't be marked read, even inside the same org.
    assert _with(me, lambda s: s.mark_read(theirs.id)) is False
    assert _with(colleague, lambda s: s.unread_count()) == 1

    assert _with(me, lambda s: s.mark_read(mine.id)) is True
    assert _with(me, lambda s: s.unread_count()) == 0
    assert _with(me, lambda s: s.list_notices())[0].read_at is not None


def test_mark_all_read_clears_only_the_callers():
    org = str(uuid4())
    me, colleague = _org_scope(org, str(uuid4())), _org_scope(org, str(uuid4()))
    _post(me)
    _post(me)
    _post(colleague)
    assert _with(me, lambda s: s.mark_all_read()) == 2
    assert _with(me, lambda s: s.unread_count()) == 0
    assert _with(colleague, lambda s: s.unread_count()) == 1


def test_the_oldest_go_past_the_limits(monkeypatch):
    monkeypatch.setattr(notices, "MAX_PER_PERSON", 3)
    me = _org_scope(str(uuid4()), str(uuid4()))
    stale = _post(me, "Stale")
    session = get_open_session()
    try:
        row = session.get(Notice, stale.id)
        row.created_at = datetime.now(timezone.utc) - timedelta(days=31)
        session.add(row)
        session.commit()
    finally:
        session.close()

    for title in ("One", "Two", "Three", "Four"):
        _post(me, title)
    assert [n.title for n in _with(me, lambda s: s.list_notices())] == ["Four", "Three", "Two"]


def test_a_long_title_is_cut_to_fit():
    me = _org_scope(str(uuid4()), str(uuid4()))
    assert len(_post(me, "x" * 400).title) == 255


# ── Scheduled runs ──


async def _settle() -> None:
    """Let the detached notice writes finish."""
    from cowork.services.push import dispatch

    while dispatch._background_tasks:
        await asyncio.gather(*list(dispatch._background_tasks), return_exceptions=True)


def _notices_for(schedule_id) -> list[Notice]:
    session = get_open_session()
    try:
        rows = session.exec(select(Notice).where(Notice.kind == SCHEDULED_RUN_FAILED)).all()
        return [n for n in rows if n.data.get("schedule_id") == str(schedule_id)]
    finally:
        session.close()


async def test_a_run_that_didnt_finish_posts_a_notice(schedule_id, turns):  # noqa: F811
    turns.reason = "error"
    (slot,) = _poll(schedule_id)
    await _run(slot)
    await _settle()
    (notice,) = _notices_for(schedule_id)
    assert notice.title == _schedule(schedule_id).title
    assert notice.read_at is None
    assert notice.data["outcome"] == "failed_during_execution"
    assert notice.data["run_id"] and notice.data["conversation_id"]
    # Never the failure's text: a notice can show on a locked screen.
    assert "error" not in notice.title.lower()


@pytest.mark.parametrize("reason", ["completed", "cancelled"])
async def test_a_finished_or_cancelled_run_posts_nothing(schedule_id, turns, reason):  # noqa: F811
    turns.reason = reason
    (slot,) = _poll(schedule_id)
    await _run(slot)
    await _settle()
    assert _notices_for(schedule_id) == []


async def test_a_trial_run_posts_nothing(schedule_id, turns):  # noqa: F811
    turns.reason = "error"
    await scheduler.execute_schedule(schedule_id, is_manual=True, trial=True)
    await _settle()
    assert _notices_for(schedule_id) == []


async def test_a_reaped_run_posts_a_notice(schedule_id):  # noqa: F811
    scheduler._push_reaped_run(scheduler.ReapedRunPush(
        org_id=None, user_id=None, schedule_name="Reaped task", conversation_id=None,
        schedule_id=schedule_id, run_id=uuid4(), outcome="uncertain",
    ))
    await _settle()
    (notice,) = _notices_for(schedule_id)
    assert (notice.title, notice.data["outcome"]) == ("Reaped task", "uncertain")


async def test_a_notice_that_cant_be_written_never_fails_the_run(schedule_id, turns, monkeypatch):  # noqa: F811
    def broken(*_args, **_kwargs):
        raise RuntimeError("database gone")

    monkeypatch.setattr(notices, "post_notice", broken)
    turns.reason = "error"
    (slot,) = _poll(schedule_id)
    await _run(slot)
    await _settle()
    assert _notices_for(schedule_id) == []


# ── API ──


async def test_the_api_lists_counts_and_reads():
    session = get_open_session()
    try:
        notice = NoticeService(ScopedSession(session, SYSTEM_SCOPE)).post(
            kind=SCHEDULED_RUN_FAILED, title="From the API test", data={},
        )
    finally:
        session.close()

    async with _client() as client:
        listed = (await client.get("/api/v1/notices/")).json()
        assert any(n["id"] == str(notice.id) and n["readAt"] is None for n in listed["notices"])
        before = (await client.get("/api/v1/notices/unread-count")).json()["unreadCount"]
        assert before == listed["unreadCount"] >= 1

        assert (await client.post(f"/api/v1/notices/{notice.id}/read")).status_code == 204
        after = (await client.get("/api/v1/notices/unread-count")).json()["unreadCount"]
        assert after == before - 1

        assert (await client.post(f"/api/v1/notices/{uuid4()}/read")).status_code == 404
        assert (await client.post("/api/v1/notices/read-all")).json() == {"unreadCount": 0}
        assert (await client.get("/api/v1/notices/unread-count")).json() == {"unreadCount": 0}
