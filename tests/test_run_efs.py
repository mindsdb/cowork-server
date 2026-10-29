"""run_efs: shared file store work in its own threads, with a bounded wait for
a slot, a share per person, and a cancel that waits for the work.

Each test holds slots with calls blocked on an Event, the way a call on the
hard NFS mount blocks while the file server doesn't answer.
"""
from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from pathlib import Path

import pytest

import cowork.db.session as db_session
from cowork.db import units
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.db.units import FilesystemBusy, on_efs_thread, run_efs
from cowork.services.artifact_lock import artifact_lock

ALICE = TenantScope(org_mode=True, org_id="org-1", user_id="alice")
BOB = TenantScope(org_mode=True, org_id="org-1", user_id="bob")


@pytest.fixture(autouse=True)
def one_second_wait(monkeypatch):
    monkeypatch.setattr(db_session.settings.database, "pool_timeout", 1)


@contextlib.asynccontextmanager
async def holding(count: int, scope: TenantScope):
    """``count`` calls running on the file store, blocked until the block ends."""
    release = threading.Event()
    started = threading.Semaphore(0)

    def blocked():
        started.release()
        release.wait(timeout=30)

    calls = [asyncio.create_task(run_efs(blocked, scope=scope)) for _ in range(count)]
    for _ in range(count):
        await asyncio.to_thread(started.acquire)
    try:
        yield
    finally:
        release.set()
        await asyncio.gather(*calls)


async def test_one_person_holding_their_share_leaves_slots_for_everyone_else():
    async with holding(units.EFS_SHARE, ALICE):
        with pytest.raises(FilesystemBusy):
            await run_efs(lambda: "alice again", scope=ALICE)
        assert await asyncio.wait_for(run_efs(lambda: "bob", scope=BOB), timeout=0.5) == "bob"


async def test_a_full_file_store_refuses_a_call_after_pool_timeout():
    loop = asyncio.get_running_loop()
    async with holding(units.EFS_SLOTS, LOCAL_SCOPE):
        started = loop.time()
        with pytest.raises(FilesystemBusy):
            await run_efs(lambda: "late", scope=LOCAL_SCOPE)
        waited = loop.time() - started
    assert 0.8 <= waited <= 1 + 1


async def test_bookkeeping_without_a_deadline_waits_for_a_slot_and_runs():
    """A turn's artifact diff waits past POOL_TIMEOUT rather than fail: in org
    mode nothing reconciles a turn whose diff was refused."""
    ran = []
    async with holding(units.EFS_SLOTS, LOCAL_SCOPE):
        diff = asyncio.create_task(
            run_efs(lambda: ran.append("diff") or "recorded", scope=LOCAL_SCOPE, deadline=False)
        )
        await asyncio.sleep(1.5)
        assert not diff.done() and ran == []
    assert await asyncio.wait_for(diff, timeout=2) == "recorded"
    assert ran == ["diff"]


async def test_a_stop_during_the_work_waits_for_it_and_leaves_no_lock_behind(tmp_path: Path):
    folder = tmp_path / "artifact"
    folder.mkdir()
    entered = threading.Event()
    finished = []

    def diff_under_the_revision_lock():
        with artifact_lock(folder):
            entered.set()
            time.sleep(0.5)
            finished.append(time.monotonic())

    turn = asyncio.create_task(run_efs(diff_under_the_revision_lock, scope=ALICE, deadline=False))
    await asyncio.to_thread(entered.wait, 5)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn
    unwound = time.monotonic()

    assert finished and finished[0] <= unwound
    assert not (folder / ".revisions" / ".lock").exists()


async def test_a_stop_while_waiting_for_a_slot_leaves_at_once_and_the_work_never_runs():
    ran = []
    async with holding(units.EFS_SLOTS, LOCAL_SCOPE):
        waiting = asyncio.create_task(run_efs(lambda: ran.append("ran"), scope=LOCAL_SCOPE))
        await asyncio.sleep(0.1)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiting, timeout=0.5)
    await asyncio.sleep(0.1)
    assert ran == []


async def test_the_work_knows_it_runs_on_a_file_store_thread():
    seen = await run_efs(lambda: (on_efs_thread(), threading.get_ident()), scope=LOCAL_SCOPE)
    assert seen[0] is True
    # Worker threads are pooled: the mark is the call's, not the thread's.
    after = await asyncio.to_thread(lambda: (on_efs_thread(), threading.get_ident()))
    assert after[0] is False
    assert on_efs_thread() is False
