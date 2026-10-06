"""Database units of work: short sessions, run off the event loop.

A unit is a function of one session. It reads and writes what it needs, then
the session commits and closes, so a pooled connection is held only while a
unit runs. ``unit_session`` is that session for synchronous code. ``run_db``
runs a unit in a worker thread for async code, so a wait for a connection
never stalls the event loop.

A unit's waits, first for a slot and then for a connection, share one
POOL_TIMEOUT budget and end in ``DatabaseBusy``, which the app answers with 503
(cowork.server).
"""
from __future__ import annotations

import asyncio
import logging
import math
import threading
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


class _Abandoned(Exception):
    """The unit's caller stopped waiting before the unit got a connection."""


class _Handoff:
    """Settles, once, whether a unit runs or its caller has stopped waiting.

    The unit's thread calls ``admit`` once it holds a connection; the caller
    calls ``give_up`` when its budget runs out, or when it is cancelled, before
    that. The first of the two wins, under a lock: either the unit runs to its
    end while its caller waits for it, or it never runs and its connection
    goes straight back to the pool.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._admitted = False
        self._given_up = False

    @property
    def given_up(self) -> bool:
        with self._lock:
            return self._given_up

    def admit(self) -> bool:
        """For the unit's thread: True when the unit may run."""
        with self._lock:
            if not self._given_up:
                self._admitted = True
            return self._admitted

    def give_up(self) -> bool:
        """For the caller: True when the unit will not run."""
        with self._lock:
            if not self._admitted:
                self._given_up = True
            return self._given_up


def _run_unit(fn: Callable[[ScopedSession], T], scope: TenantScope, handoff: _Handoff) -> T:
    with unit_session(scope=scope) as session:
        if not handoff.admit():
            raise _Abandoned
        return fn(session)


async def _unit_task(
    fn: Callable[[ScopedSession], T], scope: TenantScope, handoff: _Handoff, deadline: float,
) -> T:
    """A unit's own task: take a slot by ``deadline``, then run the unit in a
    worker thread, holding the slot until the thread ends."""
    slots = _slots(_engine())
    with anyio.CancelScope(deadline=deadline) as slot_wait:
        await slots.acquire()
    if slot_wait.cancelled_caught:
        raise DatabaseBusy("no database unit slot freed within POOL_TIMEOUT")
    try:
        if handoff.given_up:
            raise _Abandoned
        return await anyio.to_thread.run_sync(_run_unit, fn, scope, handoff, limiter=_threads())
    finally:
        slots.release()


def _retrieve_outcome(unit: asyncio.Future[Any]) -> None:
    """Mark a unit's exception as seen. A unit whose caller stopped waiting
    ends with nobody awaiting it, and that is expected."""
    if not unit.cancelled():
        unit.exception()


async def run_db(fn: Callable[[ScopedSession], T], *, scope: TenantScope) -> T:
    """Run ``fn`` as one unit of work in a worker thread and return its result.

    ``fn`` gets a ``unit_session``. Its result crosses back to the event loop,
    so it must be plain data, a pydantic model, or detached rows with every
    attribute the caller reads already loaded.

    The wait for a slot and the wait for a connection share one POOL_TIMEOUT
    budget, so a caller is refused with DatabaseBusy about POOL_TIMEOUT after
    it asked, however that time splits between the two. A caller that is
    refused or cancelled before its unit has a connection leaves at once, and
    the unit never runs. Once the unit runs, a cancel of the caller waits for
    it to finish before it propagates: the session belongs to the thread, and
    unwinding the caller first would run its cleanup while the unit is still
    writing.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + db_session.settings.database.pool_timeout
    handoff = _Handoff()
    unit = asyncio.ensure_future(_unit_task(fn, scope, handoff, deadline))
    unit.add_done_callback(_retrieve_outcome)
    try:
        await asyncio.wait([unit], timeout=max(0.0, deadline - loop.time()))
    except asyncio.CancelledError:
        if not handoff.give_up():
            await _drain(unit)
        raise
    if not unit.done() and handoff.give_up():
        raise DatabaseBusy("no database connection freed within POOL_TIMEOUT")
    return await _finish_even_if_cancelled(unit)


async def run_to_completion(work: Coroutine[Any, Any, T]) -> T:
    """Run ``work`` in its own task. A cancel of the caller waits for it to end,
    then propagates.

    For a step a cancel must not split, such as saving an answer and writing
    the frame that reports it: the step ends whole, and the cancel arrives
    after it.
    """
    return await _finish_even_if_cancelled(asyncio.ensure_future(work))


async def _finish_even_if_cancelled(task: asyncio.Future[T]) -> T:
    """Await ``task``, and make a cancel of the caller wait for it.

    Every turn cancel in this server is a native ``Task.cancel()``: Stop, the
    idle watchdog, a turn delete and shutdown. anyio's shield around a worker
    thread holds off only anyio's own cancel scopes, so without this the
    caller would unwind while its work still runs. Repeated cancels are
    absorbed until the work ends, then the cancel is re-raised.
    """
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await _drain(task)
        raise


async def _drain(task: asyncio.Future[Any]) -> None:
    """Wait for ``task`` to end, absorbing cancels, and log a failure that its
    cancelled caller will never see."""
    while not task.done():
        try:
            await asyncio.wait([task])
        except asyncio.CancelledError:
            continue
    if not task.cancelled() and task.exception() is not None:
        logger.warning(
            "work its caller waited for failed after the caller was cancelled",
            exc_info=task.exception(),
        )


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
