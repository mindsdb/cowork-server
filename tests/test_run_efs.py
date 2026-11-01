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
from types import SimpleNamespace

import anyio
import pytest

import cowork.db.session as db_session
from cowork.db import units
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.db.units import DatabaseBusy, FilesystemBusy, on_efs_thread, run_db, run_efs
from cowork.services.artifact_lock import artifact_lock

ALICE = TenantScope(org_mode=True, org_id="org-1", user_id="alice")
BOB = TenantScope(org_mode=True, org_id="org-1", user_id="bob")
# A channel turn's scope (scope_for_org): the organization's, no person's.
ORG_1_CHANNELS = TenantScope(org_mode=True, org_id="org-1")


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


async def test_an_organizations_work_with_no_person_holds_one_share_and_leaves_slots_for_everyone():
    """Channel bookkeeping waits for a slot rather than fail, but one
    organization's turns still hold no more slots than one person."""
    release = threading.Event()
    started = threading.Semaphore(0)
    running = []

    def blocked():
        running.append(threading.get_ident())
        started.release()
        release.wait(timeout=30)

    turns = [
        asyncio.create_task(run_efs(blocked, scope=ORG_1_CHANNELS, deadline=False))
        for _ in range(units.EFS_SLOTS)
    ]
    try:
        for _ in range(units.EFS_SHARE):
            await asyncio.to_thread(started.acquire)
        await asyncio.sleep(0.2)
        assert len(running) == units.EFS_SHARE
        assert await asyncio.wait_for(run_efs(lambda: "bob", scope=BOB), timeout=0.5) == "bob"
    finally:
        release.set()
        await asyncio.gather(*turns)
    assert len(running) == units.EFS_SLOTS


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


@pytest.fixture
def file_units_take_unit_slots(monkeypatch):
    """File units take a unit slot as they do on Postgres. The suite's
    database is SQLite, where they take none (``_file_unit_slots``), and its
    one-connection pool stands in for Postgres's here."""
    monkeypatch.setattr(units, "_file_unit_slots", units._slots)


async def test_a_file_unit_takes_a_unit_slot_on_postgres_and_none_on_sqlite():
    postgres = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
    assert units._file_unit_slots(postgres) is units._slots(postgres)
    assert units._file_unit_slots(units._engine()) is None


async def test_a_database_unit_runs_beside_a_long_file_unit_on_sqlite():
    """On SQLite, the desktop's database, the one unit slot orders database
    units. A file unit's work, such as a large upload, can run past
    POOL_TIMEOUT, so a file unit takes no unit slot there: one that held it
    would refuse every database unit behind it with DatabaseBusy."""
    budget = db_session.settings.database.pool_timeout
    started = threading.Event()
    release = threading.Event()

    def long_file_work(session):
        started.set()
        release.wait(timeout=30)
        return "uploaded"

    upload = asyncio.create_task(units.run_efs_unit(long_file_work, scope=LOCAL_SCOPE))
    try:
        await asyncio.to_thread(started.wait, 5)
        await asyncio.sleep(budget)
        answered = await asyncio.wait_for(run_db(lambda session: "now", scope=LOCAL_SCOPE), timeout=0.5)
    finally:
        release.set()

    assert answered == "now"
    assert await upload == "uploaded"


async def test_a_file_unit_waits_one_pool_timeout_in_all(one_connection_pool, file_units_take_unit_slots):
    """A database unit on the file store's threads waits for its file store
    slot and then for a unit slot under one POOL_TIMEOUT, and it waits for the
    unit slot on the event loop. Without the unit slot its thread waited
    inside the pool for a connection, a second POOL_TIMEOUT after the file
    store slot freed, and held that slot all the while."""
    budget = db_session.settings.database.pool_timeout
    loop = asyncio.get_running_loop()
    unit_slots = units._slots(one_connection_pool)
    ran = []

    async def file_store_busy_for(seconds: float) -> None:
        async with holding(units.EFS_SLOTS, LOCAL_SCOPE):
            await asyncio.sleep(seconds)

    held_elsewhere = one_connection_pool.connect()
    try:
        async with unit_slots:
            busy = asyncio.create_task(file_store_busy_for(0.6 * budget))
            await asyncio.sleep(0.1)
            started = loop.time()
            unit = asyncio.create_task(units.run_efs_unit(ran.append, scope=LOCAL_SCOPE))
            await busy
            await asyncio.sleep(0.2 * budget)
            threads_while_waiting = units._threads().borrowed_tokens
            with pytest.raises(DatabaseBusy):
                await unit
            elapsed = loop.time() - started
    finally:
        held_elsewhere.close()

    assert 0.8 * budget <= elapsed <= 1.2 * budget, (
        f"refused {elapsed:.2f}s after the call; POOL_TIMEOUT is {budget}s"
    )
    assert threads_while_waiting == 0, "a thread waited inside the pool"
    assert ran == []


async def test_a_file_unit_queued_on_a_stuck_file_store_holds_no_database_slot(file_units_take_unit_slots):
    """A file unit takes its file store slot before its unit slot. Queued
    behind a stuck file store it holds nothing the database needs, so a
    database unit still answers at once."""
    async with holding(units.EFS_SLOTS, LOCAL_SCOPE):
        queued = [
            asyncio.create_task(units.run_efs_unit(lambda session: "late", scope=LOCAL_SCOPE))
            for _ in range(3)
        ]
        await asyncio.sleep(0.1)
        borrowed = units._slots(units._engine()).borrowed_tokens
        answered = await asyncio.wait_for(run_db(lambda session: "now", scope=LOCAL_SCOPE), timeout=0.5)
        outcomes = await asyncio.gather(*queued, return_exceptions=True)

    assert borrowed == 0
    assert answered == "now"
    assert all(isinstance(outcome, FilesystemBusy) for outcome in outcomes)


@pytest.mark.parametrize("cancel", ["anyio scope", "two native cancels"])
async def test_a_release_step_runs_on_the_file_store_threads_when_its_caller_is_cancelled(cancel):
    """A ``finally`` that closes a file through ``run_efs_cleanup`` still
    closes it on a file store thread when the caller is cancelled while it
    waits for a slot: by an anyio cancel scope, which cancels again on every
    pass of the loop (a disconnect, a middleware's task group), or by a second
    native cancel (Stop, then shutdown). Through a plain ``run_efs`` call the
    close never ran."""
    closed_on: list[bool] = []
    entered = asyncio.Event()

    async def response() -> None:
        try:
            entered.set()
            await asyncio.sleep(10)
        finally:
            await units.run_efs_cleanup(lambda: closed_on.append(on_efs_thread()), scope=ALICE)

    async with holding(units.EFS_SLOTS, LOCAL_SCOPE):
        if cancel == "two native cancels":
            task = asyncio.create_task(response())
            await entered.wait()
            task.cancel()
            await asyncio.sleep(0.1)
            task.cancel()
        else:
            async def in_a_cancelled_scope() -> None:
                with anyio.CancelScope() as scope:
                    asyncio.get_running_loop().call_soon(scope.cancel)
                    await response()

            task = asyncio.create_task(in_a_cancelled_scope())
            await entered.wait()
        await asyncio.sleep(0.2)
        assert closed_on == [] and not task.done()
    if cancel == "two native cancels":
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
    else:
        await asyncio.wait_for(task, timeout=2)

    assert closed_on == [True]
