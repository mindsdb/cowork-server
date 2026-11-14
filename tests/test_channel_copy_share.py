"""run_efs_copy's file store share: an organization's channel copies hold a
share of their own (``org:<org_id>:copy``), so its new bindings, staging and
artifact sends keep the organization's own share while copies run.

Each test holds slots with calls blocked on an Event, as a copy on the hard
NFS mount blocks while the file server is slow.
"""
from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Awaitable, Callable

import pytest

import cowork.db.session as db_session
from cowork.db import units
from cowork.db.scoped import TenantScope
from cowork.db.units import FilesystemBusy, run_efs, run_efs_copy

# A channel turn's scope (scope_for_org): the organization's, no person's.
ORG_1_CHANNELS = TenantScope(org_mode=True, org_id="org-1")
ALICE = TenantScope(org_mode=True, org_id="org-1", user_id="alice")


@pytest.fixture(autouse=True)
def small_share(monkeypatch):
    """A share of 2 and a one-second wait; the copy slots stay above the
    share, so copies alone can fill it."""
    monkeypatch.setattr(db_session.settings.database, "pool_timeout", 1)
    monkeypatch.setattr(db_session.settings.storage, "efs_share", 2)
    monkeypatch.setenv("COWORK_TURN_PROJECT_FILES_COPY_SLOTS", "8")


@contextlib.asynccontextmanager
async def holding(count: int, run: Callable[..., Awaitable], scope: TenantScope):
    """``count`` calls of ``run`` on the file store, blocked until the block ends."""
    release = threading.Event()
    started = threading.Semaphore(0)

    def blocked():
        started.release()
        release.wait(timeout=30)

    calls = [asyncio.create_task(run(blocked, scope=scope)) for _ in range(count)]
    for _ in range(count):
        await asyncio.to_thread(started.acquire)
    try:
        yield
    finally:
        release.set()
        await asyncio.gather(*calls)


async def test_an_organizations_channel_copies_leave_its_share_to_its_other_work():
    async with holding(db_session.settings.storage.efs_share, run_efs_copy, ORG_1_CHANNELS):
        bound = await asyncio.wait_for(run_efs(lambda: "new binding", scope=ORG_1_CHANNELS), timeout=0.5)
    assert bound == "new binding"


async def test_an_organizations_other_work_leaves_its_copies_a_share():
    async with holding(db_session.settings.storage.efs_share, run_efs, ORG_1_CHANNELS):
        copied = await asyncio.wait_for(run_efs_copy(lambda: "copied", scope=ORG_1_CHANNELS), timeout=0.5)
    assert copied == "copied"


async def test_an_organizations_copies_still_hold_no_more_than_one_share():
    async with holding(db_session.settings.storage.efs_share, run_efs_copy, ORG_1_CHANNELS):
        with pytest.raises(FilesystemBusy):
            await run_efs_copy(lambda: "a third copy", scope=ORG_1_CHANNELS)


async def test_the_copy_share_is_keyed_apart_for_an_organization_and_shared_for_a_person():
    assert units._efs_share(ORG_1_CHANNELS, for_copy=True) is not units._efs_share(ORG_1_CHANNELS)
    assert units._efs_share(ALICE, for_copy=True) is units._efs_share(ALICE)
