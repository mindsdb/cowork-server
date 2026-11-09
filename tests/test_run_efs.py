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
# Identities a request carries: the app reads X-User-Id as a UUID.
ORG_UUID = "6f0d1c2e-3a4b-4c5d-8e6f-7a8b9c0d1e2f"
ALICE_UUID = "11111111-1111-4111-8111-111111111111"
BOB_UUID = "22222222-2222-4222-8222-222222222222"


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


def test_the_slots_and_the_share_default_to_100_and_10_and_read_the_environment(monkeypatch):
    from cowork.common.settings.app_settings import StorageSettings

    monkeypatch.delenv("EFS_SLOTS", raising=False)
    monkeypatch.delenv("EFS_SHARE", raising=False)
    defaults = StorageSettings()
    assert (defaults.efs_slots, defaults.efs_share) == (100, 10)

    monkeypatch.setenv("EFS_SLOTS", "40")
    monkeypatch.setenv("EFS_SHARE", "6")
    configured = StorageSettings()
    assert (configured.efs_slots, configured.efs_share) == (40, 6)


async def test_the_limiters_take_their_sizes_from_the_settings(monkeypatch):
    monkeypatch.setattr(db_session.settings.storage, "efs_slots", 3)
    monkeypatch.setattr(db_session.settings.storage, "efs_share", 2)

    assert units._efs_slots().total_tokens == 3
    assert units._efs_share(ALICE).total_tokens == 2


async def test_one_person_holding_their_share_leaves_slots_for_everyone_else():
    async with holding(db_session.settings.storage.efs_share, ALICE):
        with pytest.raises(FilesystemBusy):
            await run_efs(lambda: "alice again", scope=ALICE)
        assert await asyncio.wait_for(run_efs(lambda: "bob", scope=BOB), timeout=0.5) == "bob"


async def test_a_refusal_names_the_full_share_and_whose_it_is():
    async with holding(db_session.settings.storage.efs_share, ALICE):
        with pytest.raises(FilesystemBusy) as refused:
            await run_efs(lambda: "alice again", scope=ALICE)

    assert (refused.value.limiter, refused.value.share) == ("efs_share", "alice")


async def test_a_refusal_names_the_full_slots_and_the_waiting_callers_share(monkeypatch):
    monkeypatch.setattr(db_session.settings.storage, "efs_slots", 2)
    async with holding(2, BOB):
        with pytest.raises(FilesystemBusy) as refused:
            await run_efs(lambda: "alice", scope=ALICE)
        with pytest.raises(FilesystemBusy) as channel_refused:
            await run_efs(lambda: "channel", scope=ORG_1_CHANNELS)

    assert (refused.value.limiter, refused.value.share) == ("run_efs", "alice")
    assert (channel_refused.value.limiter, channel_refused.value.share) == ("run_efs", "org:org-1")


async def test_a_refusal_on_the_desktop_names_the_slots_and_no_share():
    async with holding(db_session.settings.storage.efs_slots, LOCAL_SCOPE):
        with pytest.raises(FilesystemBusy) as refused:
            await run_efs(lambda: "late", scope=LOCAL_SCOPE)

    assert (refused.value.limiter, refused.value.share) == ("run_efs", None)


@pytest.fixture()
def org_app(monkeypatch):
    """The app in MindsHub mode, served on the test's own event loop, so the
    file store limiters a test fills are the ones its requests wait for."""
    import httpx

    from cowork.common.settings.app_settings import get_app_settings
    from cowork.server import create_app

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "enforce")
    get_app_settings.cache_clear()
    transport = httpx.ASGITransport(app=create_app(), raise_app_exceptions=False)
    yield httpx.AsyncClient(transport=transport, base_url="http://test")
    get_app_settings.cache_clear()


@pytest.mark.parametrize("full", ["share", "slots"])
async def test_a_refused_request_logs_the_limit_that_ran_out_and_whose_share(org_app, monkeypatch, caplog, full):
    """Through the real route and the app's own 503 handler, so a refusal
    raised without its limit, or a handler that drops it, fails here. The
    server logs through the logger setup_logging returns, cowork.common.logger."""
    import logging

    alice = TenantScope(org_mode=True, org_id=ORG_UUID, user_id=ALICE_UUID)
    bob = TenantScope(org_mode=True, org_id=ORG_UUID, user_id=BOB_UUID)
    if full == "slots":
        monkeypatch.setattr(db_session.settings.storage, "efs_slots", 2)
    holder, count = (alice, db_session.settings.storage.efs_share) if full == "share" else (bob, 2)

    with caplog.at_level(logging.WARNING, logger="cowork.common.logger"):
        async with holding(count, holder):
            async with org_app as client:
                response = await client.get(
                    "/api/v1/artifacts/", headers={"X-User-Id": ALICE_UUID, "X-Organization-Id": ORG_UUID}
                )

    assert response.status_code == 503, response.text
    [record] = [
        record for record in caplog.records
        if record.name == "cowork.common.logger" and getattr(record, "event", None) == "filesystem_busy"
    ]
    assert record.levelno == logging.WARNING
    assert record.route == "/api/v1/artifacts/"
    assert (record.limiter, record.share) == ("efs_share" if full == "share" else "run_efs", ALICE_UUID)


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
        for _ in range(db_session.settings.storage.efs_slots)
    ]
    try:
        for _ in range(db_session.settings.storage.efs_share):
            await asyncio.to_thread(started.acquire)
        await asyncio.sleep(0.2)
        assert len(running) == db_session.settings.storage.efs_share
        assert await asyncio.wait_for(run_efs(lambda: "bob", scope=BOB), timeout=0.5) == "bob"
    finally:
        release.set()
        await asyncio.gather(*turns)
    assert len(running) == db_session.settings.storage.efs_slots


async def test_a_full_file_store_refuses_a_call_after_pool_timeout():
    loop = asyncio.get_running_loop()
    async with holding(db_session.settings.storage.efs_slots, LOCAL_SCOPE):
        started = loop.time()
        with pytest.raises(FilesystemBusy):
            await run_efs(lambda: "late", scope=LOCAL_SCOPE)
        waited = loop.time() - started
    assert 0.8 <= waited <= 1 + 1


async def test_bookkeeping_without_a_deadline_waits_for_a_slot_and_runs():
    """A turn's artifact diff waits past POOL_TIMEOUT rather than fail: in org
    mode nothing reconciles a turn whose diff was refused."""
    ran = []
    async with holding(db_session.settings.storage.efs_slots, LOCAL_SCOPE):
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
    async with holding(db_session.settings.storage.efs_slots, LOCAL_SCOPE):
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
        async with holding(db_session.settings.storage.efs_slots, LOCAL_SCOPE):
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
    async with holding(db_session.settings.storage.efs_slots, LOCAL_SCOPE):
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


async def test_file_units_at_their_cap_leave_a_database_unit_its_slot(monkeypatch):
    """Three database unit slots, as Postgres's pool would give, and file
    units capped at two (EFS_UNIT_SLOTS). Two file units stuck on the file
    store hold their cap, and a third waits on the cap rather than for the
    last unit slot, so a database unit still gets that slot and answers.
    Without the cap the third took it."""
    monkeypatch.setattr(db_session.settings.database, "efs_unit_slots", 2)
    slots = anyio.CapacityLimiter(3)
    monkeypatch.setattr(units, "_slots", lambda _engine: slots)
    monkeypatch.setattr(units, "_file_unit_slots", lambda _engine: slots)
    release = threading.Event()
    started = threading.Semaphore(0)

    def stuck(session):
        started.release()
        release.wait(timeout=30)
        return "file"

    files = [asyncio.create_task(units.run_efs_unit(stuck, scope=LOCAL_SCOPE)) for _ in range(3)]
    try:
        for _ in range(2):
            assert await asyncio.to_thread(started.acquire, True, 5), "a file unit never started"
        await asyncio.sleep(0.1)
        waiting_on_the_cap = units._efs_unit_slots().statistics().tasks_waiting
        answered = await asyncio.wait_for(run_db(lambda session: "database", scope=LOCAL_SCOPE), timeout=0.5)
    finally:
        release.set()
        await asyncio.gather(*files, return_exceptions=True)

    assert waiting_on_the_cap == 1
    assert answered == "database"


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

    async with holding(db_session.settings.storage.efs_slots, LOCAL_SCOPE):
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
