"""The host's admission lock must work across workers and survive Stop."""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading

import pytest
import sqlalchemy as sa

from cowork.db import session as db_session
from cowork.services import live_app_admission as admission
from tests.integration.prereq import missing_prerequisite


async def test_sqlite_workers_share_the_database_file_lock(tmp_path, monkeypatch):
    database = tmp_path / "shared.db"
    engine = sa.create_engine(f"sqlite:///{database}")
    monkeypatch.setattr(db_session, "get_engine", lambda _uri: engine)
    # Another interpreter gets its own Python locks and SQLAlchemy engine.
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", """
import asyncio, sys
from cowork.services.live_app_admission import live_app_launch_lock
async def main():
    async with live_app_launch_lock():
        print('held', flush=True)
        await asyncio.to_thread(sys.stdin.readline)
asyncio.run(main())
"""],
        env={**os.environ, "DATABASE_URI": f"sqlite:///{database}"},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert await asyncio.wait_for(asyncio.to_thread(child.stdout.readline), 15) == "held\n"
        monkeypatch.setattr(admission, "LAUNCH_LOCK_TIMEOUT_SECONDS", 0.1)
        from anton.core.artifacts.live_app_slots import LiveAppLaunchError

        with pytest.raises(LiveAppLaunchError, match="busy") as error:
            async with admission.live_app_launch_lock():
                pytest.fail("a different worker still holds admission")
        assert error.value.cause == "slot_busy"
        child.stdin.write("release\n")
        child.stdin.flush()
        assert await asyncio.wait_for(asyncio.to_thread(child.wait), 15) == 0
        async with admission.live_app_launch_lock():
            pass
    finally:
        if child.poll() is None:
            child.kill()
            await asyncio.to_thread(child.wait)
        child.stdin.close()
        child.stdout.close()
        child.stderr.close()
        engine.dispose()


async def test_cancel_during_acquisition_releases_a_lock_acquired_by_the_thread(monkeypatch):
    entered, finish, cleanup = threading.Event(), threading.Event(), threading.Event()
    original = admission._LaunchLock.try_acquire
    original_close = admission._LaunchLock.close

    def observed_close(lock):
        cleanup.set()
        original_close(lock)

    monkeypatch.setattr(admission._LaunchLock, "close", observed_close)

    def slow_acquire(lock):
        acquired = original(lock)
        entered.set()
        assert finish.wait(5)
        return acquired

    monkeypatch.setattr(admission._LaunchLock, "try_acquire", slow_acquire)

    async def waiting():
        async with admission.live_app_launch_lock():
            pytest.fail("cancelled acquisition must never enter")

    task = asyncio.create_task(waiting())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        # Cancellation cannot release admission while its acquisition worker
        # is still running, even if that worker already took the OS lock.
        assert not await asyncio.to_thread(cleanup.wait, 0.1)
        assert not task.done()
    finally:
        finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    monkeypatch.setattr(admission._LaunchLock, "try_acquire", original)
    async with admission.live_app_launch_lock():
        pass


async def test_repeated_cancel_during_release_waits_until_physical_unlock(monkeypatch):
    released, finish = threading.Event(), threading.Event()
    held = asyncio.Event()
    original = admission._LaunchLock.close

    def slow_close(lock):
        released.set()
        assert finish.wait(5)
        original(lock)

    monkeypatch.setattr(admission._LaunchLock, "close", slow_close)

    async def holding():
        async with admission.live_app_launch_lock():
            held.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(holding())
    await held.wait()
    task.cancel()
    try:
        assert await asyncio.to_thread(released.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        rival = admission._LaunchLock(db_session.get_engine(db_session.settings.database.uri))
        rival.open()
        try:
            assert rival.try_acquire() is False
        finally:
            original(rival)
    finally:
        finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    monkeypatch.setattr(admission._LaunchLock, "close", original)
    async with admission.live_app_launch_lock():
        pass


async def test_cancelled_waiter_does_not_unlock_holder(monkeypatch):
    started = asyncio.Event()
    async with admission.live_app_launch_lock():
        async def waiter():
            started.set()
            async with admission.live_app_launch_lock():
                pytest.fail("holder still owns admission")
        task = asyncio.create_task(waiter())
        await started.wait()
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        rival = admission._LaunchLock(db_session.get_engine(db_session.settings.database.uri))
        rival.open()
        try:
            assert rival.try_acquire() is False
        finally:
            rival.close()


async def test_try_lock_reports_a_busy_lock_without_waiting_or_unlocking_the_holder(monkeypatch):
    monkeypatch.setattr(admission, "_LOCK_RETRY_SECONDS", 5.0)
    async with admission.live_app_launch_lock():
        started = asyncio.get_running_loop().time()
        async with admission.try_live_app_launch_lock() as acquired:
            assert acquired is False
        assert asyncio.get_running_loop().time() - started < 1
        rival = admission._LaunchLock(db_session.get_engine(db_session.settings.database.uri))
        rival.open()
        try:
            assert rival.try_acquire() is False
        finally:
            rival.close()
    async with admission.try_live_app_launch_lock() as acquired:
        assert acquired is True
        rival = admission._LaunchLock(db_session.get_engine(db_session.settings.database.uri))
        rival.open()
        try:
            assert rival.try_acquire() is False
        finally:
            rival.close()
    async with admission.live_app_launch_lock():
        pass


def test_database_path_aliases_share_lock(tmp_path):
    actual = tmp_path / "actual.db"
    actual.touch()
    alias = tmp_path / "alias.db"
    alias.symlink_to(actual)
    first = admission._LaunchLock(sa.create_engine(f"sqlite:///{actual}"))
    second = admission._LaunchLock(sa.create_engine(f"sqlite:///{alias}"))
    try:
        first.open()
        second.open()
        assert first.try_acquire() is True
        assert second.try_acquire() is False
        first.close()
        assert second.try_acquire() is True
    finally:
        first.close()
        second.close()


def test_memory_database_has_process_lock():
    engine = sa.create_engine("sqlite:///:memory:")
    first, second = admission._LaunchLock(engine), admission._LaunchLock(engine)
    try:
        first.open()
        second.open()
        assert first.try_acquire() is True
        assert second.try_acquire() is False
        first.close()
        assert second.try_acquire() is True
    finally:
        first.close()
        second.close()


@pytest.mark.postgres_service
async def test_postgres_workers_use_a_separate_session_and_cancel_releases_it(monkeypatch):
    uri = os.environ.get("COWORK_TEST_POSTGRES_URI")
    if not uri:
        missing_prerequisite(
            "set COWORK_TEST_POSTGRES_URI to an isolated PostgreSQL test database",
            required_by="COWORK_REQUIRE_SERVICES",
        )
    engine = sa.create_engine(uri, pool_size=1, max_overflow=0, pool_timeout=0.1)
    monkeypatch.setattr(db_session, "get_engine", lambda _uri: engine)
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", """
import asyncio, sys
from cowork.services.live_app_admission import live_app_launch_lock
async def main():
    async with live_app_launch_lock():
        print('held', flush=True)
        await asyncio.to_thread(sys.stdin.readline)
asyncio.run(main())
"""],
        env={**os.environ, "DATABASE_URI": uri},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert await asyncio.wait_for(asyncio.to_thread(child.stdout.readline), 15) == "held\n"
        monkeypatch.setattr(admission, "LAUNCH_LOCK_TIMEOUT_SECONDS", 0.1)
        from anton.core.artifacts.live_app_slots import LiveAppLaunchError

        with pytest.raises(LiveAppLaunchError, match="busy"):
            async with admission.live_app_launch_lock():
                pytest.fail("PostgreSQL must exclude another worker")
        child.stdin.write("release\n")
        child.stdin.flush()
        assert await asyncio.wait_for(asyncio.to_thread(child.wait), 15) == 0

        # Exhaust the ORM pool. Admission still acquires its own physical
        # session, and never leaves a long-running database transaction.
        with engine.connect() as orm_connection:
            held = asyncio.Event()
            async def holder():
                async with admission.live_app_launch_lock():
                    held.set()
                    await asyncio.Event().wait()
            task = asyncio.create_task(holder())
            try:
                await asyncio.wait_for(held.wait(), 5)
                states = orm_connection.execute(sa.text(
                    "SELECT state FROM pg_stat_activity WHERE datname = current_database() "
                    "AND pid IN (SELECT pid FROM pg_locks WHERE locktype = 'advisory' AND granted)"
                )).scalars().all()
                assert states == ["idle"]
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            # Physical connection closure released the session advisory lock.
            async with admission.live_app_launch_lock():
                pass
    finally:
        if child.poll() is None:
            child.kill()
            await asyncio.to_thread(child.wait)
        child.stdin.close()
        child.stdout.close()
        child.stderr.close()
        engine.dispose()
