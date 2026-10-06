"""Database units of work: short sessions, run off the event loop.

A unit is a function of one session. It reads and writes what it needs, then
the session commits and closes, so a pooled connection is held only while a
unit runs. ``unit_session`` is that session for synchronous code. ``run_db``
runs a unit in a worker thread for async code, so a wait for a connection
never stalls the event loop.

Every wait is bounded by POOL_TIMEOUT and ends in ``DatabaseBusy``, which the
app answers with 503 (cowork.server).
"""
from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, TypeVar
from uuid import UUID
from weakref import WeakValueDictionary

import anyio
from anyio.lowlevel import RunVar
from sqlalchemy.engine import Engine
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from cowork.db import session as db_session
from cowork.db.scoped import ScopedSession, TenantScope

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DatabaseBusy(PoolTimeoutError):
    """No database connection freed within POOL_TIMEOUT.

    A subclass of SQLAlchemy's pool timeout, so one ``except`` clause and one
    exception handler cover both a unit that waited for its slot and any other
    checkout that waited for a connection.
    """

    status_code = 503


def busy_retry_seconds() -> int:
    """Seconds a refused caller should wait before trying again: POOL_TIMEOUT,
    and at least one."""
    return max(1, db_session.settings.database.pool_timeout)


def _engine() -> Engine:
    return db_session.get_engine(db_session.settings.database.uri)


# Per event loop, like anyio's own default thread limiter.
_SLOTS: RunVar[anyio.CapacityLimiter] = RunVar("cowork_db_unit_slots")
_THREADS: RunVar[anyio.CapacityLimiter] = RunVar("cowork_db_unit_threads")


def _slots(engine: Engine) -> anyio.CapacityLimiter:
    """This event loop's unit slots, one per connection the pool can lend.

    Units queue here, where the wait is bounded and holds no thread, rather
    than inside the pool. SQLite gets one slot: its writers serialize on the
    database file anyway, and one unit at a time keeps the desktop's writes in
    the order the event loop gave them before units ran in threads.
    """
    try:
        return _SLOTS.get()
    except LookupError:
        pass
    if engine.dialect.name == "sqlite":
        total = 1
    else:
        database = db_session.settings.database
        total = database.pool_size + database.max_overflow
    limiter = anyio.CapacityLimiter(total)
    _SLOTS.set(limiter)
    return limiter


def _threads() -> anyio.CapacityLimiter:
    """The worker threads units run on, kept apart from anyio's default limiter.

    Sync routes share that limiter's 40 threads, so a unit waiting on the pool
    must not take one of them. The slots already cap how many units run, so
    this limiter never makes a unit wait.
    """
    try:
        return _THREADS.get()
    except LookupError:
        pass
    limiter = anyio.CapacityLimiter(math.inf)
    _THREADS.set(limiter)
    return limiter


@contextmanager
def unit_session(*, scope: TenantScope) -> Iterator[ScopedSession]:
    """A tenant-scoped session that holds one pooled connection throughout.

    Services commit inside a unit, and each commit keeps the connection, so
    the only wait for the pool comes before the first statement. The block's
    work is committed when it returns and rolled back when it raises; the
    connection goes back to the pool either way.
    """
    engine = _engine()
    try:
        connection = engine.connect()
    except PoolTimeoutError as exc:
        raise DatabaseBusy("no database connection freed within POOL_TIMEOUT") from exc
    with connection:
        raw = db_session.get_session_factory(engine)(bind=connection)
        try:
            yield ScopedSession(raw, scope)
            raw.commit()
        except BaseException:
            raw.rollback()
            raise
        finally:
            raw.close()


def _run_unit(fn: Callable[[ScopedSession], T], scope: TenantScope) -> T:
    with unit_session(scope=scope) as session:
        return fn(session)


async def run_db(fn: Callable[[ScopedSession], T], *, scope: TenantScope) -> T:
    """Run ``fn`` as one unit of work in a worker thread and return its result.

    ``fn`` gets a ``unit_session``. Its result crosses back to the event loop,
    so it must be plain data, a pydantic model, or detached rows with every
    attribute the caller reads already loaded.

    The wait for a slot and the wait for a connection are each bounded by
    POOL_TIMEOUT and end in DatabaseBusy. Once the unit runs, a cancel of the
    caller waits for it to finish before it propagates: the session belongs to
    the thread, and unwinding the caller first would run its cleanup while the
    unit is still writing.
    """
    slots = _slots(_engine())
    with anyio.move_on_after(db_session.settings.database.pool_timeout) as admission:
        await slots.acquire()
    if admission.cancelled_caught:
        raise DatabaseBusy("no database unit slot freed within POOL_TIMEOUT")
    try:
        return await _finish_even_if_cancelled(
            anyio.to_thread.run_sync(_run_unit, fn, scope, limiter=_threads())
        )
    finally:
        slots.release()


async def _finish_even_if_cancelled(work: Coroutine[Any, Any, T]) -> T:
    """Await ``work`` in its own task, and make a cancel of this task wait for it.

    Every turn cancel in this server is a native ``Task.cancel()``: Stop, the
    idle watchdog, a turn delete and shutdown. anyio's shield around a worker
    thread holds off only anyio's own cancel scopes, so without this the
    caller would unwind while its unit still runs. Repeated cancels are
    absorbed until the work ends, then the cancel is re-raised.
    """
    task = asyncio.ensure_future(work)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.wait([task])
            except asyncio.CancelledError:
                continue
        if not task.cancelled() and task.exception() is not None:
            logger.warning(
                "a database unit failed after its caller was cancelled",
                exc_info=task.exception(),
            )
        raise


_writes_by_conversation: WeakValueDictionary[UUID, asyncio.Lock] = WeakValueDictionary()


@asynccontextmanager
async def conversation_writes(conversation_id: UUID) -> AsyncIterator[None]:
    """One writer at a time for a conversation's messages.

    ConversationService._next_seq numbers a message max(seq) + 1, which holds
    only while one unit at a time writes the conversation. Hold this around
    each awaited unit that saves messages. run_db waits for its thread even
    when cancelled, so the lock is released only once the write has ended.
    """
    lock = _writes_by_conversation.get(conversation_id)
    if lock is None:
        lock = asyncio.Lock()
        _writes_by_conversation[conversation_id] = lock
    async with lock:
        yield
