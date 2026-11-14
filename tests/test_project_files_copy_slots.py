"""run_efs_copy: a cloud turn's copy of its project's files takes one of the
event loop's copy slots before its file store slot. A copy holds its file
store slot for up to its time budget, so without a cap of their own, copies
started by a few accounts could hold every file store slot and refuse every
other organization's file work.

Each test holds slots with calls blocked on an Event, as a copy on the hard
NFS mount blocks while the file server is slow.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

import cowork.db.session as db_session
from cowork.common.settings.app_settings import TurnQueueSettings
from cowork.db import units
from cowork.db.scoped import TenantScope
from cowork.db.units import FilesystemBusy, run_efs, run_efs_copy

BOB = TenantScope(org_mode=True, org_id="org-2", user_id="bob")


def member(index: int) -> TenantScope:
    return TenantScope(org_mode=True, org_id="org-1", user_id=f"member-{index}")


@pytest.fixture(autouse=True)
def one_second_wait(monkeypatch):
    monkeypatch.setattr(db_session.settings.database, "pool_timeout", 1)


def test_the_copy_slots_default_to_8_and_read_the_environment(monkeypatch):
    monkeypatch.delenv("COWORK_TURN_PROJECT_FILES_COPY_SLOTS", raising=False)
    assert TurnQueueSettings().project_files_copy_slots == 8

    monkeypatch.setenv("COWORK_TURN_PROJECT_FILES_COPY_SLOTS", "3")
    assert TurnQueueSettings().project_files_copy_slots == 3


async def test_the_limiter_takes_its_size_from_the_settings_and_the_watchdog_watches_it(monkeypatch):
    monkeypatch.setenv("COWORK_TURN_PROJECT_FILES_COPY_SLOTS", "3")

    assert units._project_files_copy_slots().total_tokens == 3
    assert units.loop_limiters()["run_efs_copy"] is units._project_files_copy_slots()


async def test_full_copy_slots_leave_the_file_store_to_everyone_else_and_refuse_the_next_copy(monkeypatch):
    """Twenty people start a copy on a loop with ten file store slots. Eight
    copy at once and hold eight file store slots; the rest wait for a copy
    slot holding none, so another person's file store work gets a slot at
    once, and each waiting copy is refused with FilesystemBusy after
    POOL_TIMEOUT, failing only its own turn."""
    monkeypatch.delenv("COWORK_TURN_PROJECT_FILES_COPY_SLOTS", raising=False)
    monkeypatch.setattr(db_session.settings.storage, "efs_slots", 10)
    slots = TurnQueueSettings().project_files_copy_slots
    release = threading.Event()
    started = threading.Semaphore(0)
    ran: list[int] = []

    def copying(index: int) -> int:
        ran.append(index)
        started.release()
        release.wait(timeout=30)
        return index

    loop = asyncio.get_running_loop()
    began = loop.time()
    copies = [
        asyncio.create_task(run_efs_copy(lambda index=index: copying(index), scope=member(index)))
        for index in range(20)
    ]
    try:
        for _ in range(slots):
            await asyncio.to_thread(started.acquire)
        await asyncio.sleep(0.2)
        assert units._efs_slots().borrowed_tokens == slots

        assert await asyncio.wait_for(run_efs(lambda: "bob", scope=BOB), timeout=0.5) == "bob"

        waiting = [copy for index, copy in enumerate(copies) if index not in ran]
        assert len(waiting) == 20 - slots
        await asyncio.wait(waiting, timeout=3)
        assert all(isinstance(copy.exception(), FilesystemBusy) for copy in waiting)
        assert 0.8 <= loop.time() - began <= 2.5
    finally:
        release.set()
        await asyncio.gather(*copies, return_exceptions=True)

    assert len(ran) == slots
    assert units._project_files_copy_slots().borrowed_tokens == 0
    assert units._efs_slots().borrowed_tokens == 0
