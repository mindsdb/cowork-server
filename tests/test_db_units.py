"""Database units: one pooled connection per unit, held from its first
statement to its end; one POOL_TIMEOUT budget for a unit's waits; and a
caller that is cancelled waits for a unit that has started."""
from __future__ import annotations

import asyncio
import threading
from uuid import uuid4

import pytest
from sqlalchemy import event

from cowork.db.scoped import LOCAL_SCOPE
from cowork.db import units
from cowork.db.units import DatabaseBusy, run_db, unit_session
from cowork.models.conversation import Conversation
from cowork.services.conversations import ConversationService
from cowork.services.projects import GENERAL_PROJECT_ID


async def test_a_unit_checks_out_one_connection_however_often_it_commits(one_connection_pool):
    """Services commit inside a unit. Each commit keeps the unit's connection,
    so the unit waits on the pool once, before its first statement, and never
    again while it holds a connection."""
    checkouts = []

    def count_checkout(dbapi_connection, record, proxy):
        checkouts.append(threading.get_ident())

    event.listen(one_connection_pool, "checkout", count_checkout)
    try:

        def two_conversations(session) -> bool:
            service = ConversationService(session)
            service.create_conversation(topic="first", project_id=GENERAL_PROJECT_ID)
            service.create_conversation(topic="second", project_id=GENERAL_PROJECT_ID)
            # The advisory-lock engine reads the database URL off this bind.
            return session.get_bind() is one_connection_pool

        bind_is_the_engine = await run_db(two_conversations, scope=LOCAL_SCOPE)
    finally:
        event.remove(one_connection_pool, "checkout", count_checkout)

    assert len(checkouts) == 1
    assert checkouts[0] != threading.get_ident(), "the unit checked out on the event loop's thread"
    assert one_connection_pool.pool.checkedout() == 0
    assert bind_is_the_engine


async def test_a_cancelled_caller_waits_for_its_unit_and_the_write_lands():
    """Every turn cancel in this server is a native Task.cancel(). The caller's
    cleanup must not run while its unit is still writing, so the cancel waits
    for the unit, through repeated cancels, and then propagates."""
    topic = f"drained-{uuid4()}"
    entered = threading.Event()
    release = threading.Event()

    def save(session) -> None:
        entered.set()
        release.wait(timeout=10)
        ConversationService(session).create_conversation(topic=topic, project_id=GENERAL_PROJECT_ID)

    caller = asyncio.create_task(run_db(save, scope=LOCAL_SCOPE))
    while not entered.is_set():
        await asyncio.sleep(0.01)
    caller.cancel()
    await asyncio.sleep(0.05)
    caller.cancel()  # Stop after the idle watchdog, say
    await asyncio.sleep(0.05)
    waited_for_the_unit = not caller.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await caller

    assert waited_for_the_unit
    with unit_session(scope=LOCAL_SCOPE) as session:
        saved = session.exec(session.select(Conversation).where(Conversation.topic == topic)).all()
    assert len(saved) == 1


async def _until(condition, *, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        assert loop.time() < deadline, "timed out waiting"
        await asyncio.sleep(0.01)


async def test_the_slot_wait_and_the_connection_wait_share_one_pool_timeout(one_connection_pool):
    """A unit that spent most of POOL_TIMEOUT waiting for a slot has only the
    rest of it to wait for a connection, so its caller is refused about
    POOL_TIMEOUT after it asked, not up to twice that. The unit never runs,
    and the connection it got late goes back unused."""
    budget = units.db_session.settings.database.pool_timeout
    slots = units._slots(one_connection_pool)
    slot_held = asyncio.Event()

    async def hold_the_slot():
        async with slots:
            slot_held.set()
            await asyncio.sleep(0.6 * budget)

    ran = []
    held_elsewhere = one_connection_pool.connect()
    try:
        holder = asyncio.create_task(hold_the_slot())
        await slot_held.wait()
        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(DatabaseBusy):
            await run_db(lambda session: ran.append(threading.get_ident()), scope=LOCAL_SCOPE)
        elapsed = loop.time() - started
        await holder
    finally:
        held_elsewhere.close()

    assert elapsed <= 1.2 * budget, f"refused {elapsed:.2f}s after the call; POOL_TIMEOUT is {budget}s"
    await _until(lambda: slots.borrowed_tokens == 0 and one_connection_pool.pool.checkedout() == 0)
    assert ran == []


async def test_a_caller_cancelled_while_its_unit_waits_for_a_connection_leaves_at_once(
    one_connection_pool,
):
    """Before a unit holds a connection nothing has run, so a Stop need not
    wait out the pool: the caller leaves at once and the unit never runs,
    even once the connection frees."""
    ran = []
    held_elsewhere = one_connection_pool.connect()
    try:
        caller = asyncio.create_task(
            run_db(lambda session: ran.append(threading.get_ident()), scope=LOCAL_SCOPE)
        )
        await asyncio.sleep(0.3)  # the unit's thread is now waiting on the pool
        caller.cancel()
        await asyncio.sleep(0.1)
        left_at_once = caller.done()
    finally:
        held_elsewhere.close()
    with pytest.raises(asyncio.CancelledError):
        await caller

    assert left_at_once
    await _until(lambda: units._slots(one_connection_pool).borrowed_tokens == 0)
    assert ran == []
