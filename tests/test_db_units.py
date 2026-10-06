"""Database units: one pooled connection per unit, held from its first
statement to its end, and a caller that is cancelled waits for its unit."""
from __future__ import annotations

import asyncio
import threading
from uuid import uuid4

import pytest
from sqlalchemy import event

from cowork.db.scoped import LOCAL_SCOPE
from cowork.db.units import run_db, unit_session
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
