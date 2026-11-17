"""Database units of work: short sessions, run off the event loop.

A unit is a function of one session. It reads and writes what it needs, then
the session commits and closes, so a pooled connection is held only while a
unit runs. ``unit_session`` is that session for synchronous code. ``run_db``
runs a unit in a worker thread for async code, so a wait for a connection
never stalls the event loop.

A caller of ``run_db`` waits first for a slot and then for a connection, and
gives up at one deadline, POOL_TIMEOUT after it asked, with ``DatabaseBusy``,
which the app answers with 503 (cowork.server). A worker thread still waiting
in the pool after that keeps its slot until the pool's own timeout passes or
a connection frees. When a connection frees first, the thread hands it
straight back without running the unit.

``run_efs`` does the same for work on the shared file store (``COWORK_HOME``,
an NFS hard mount on hosted deployments). A stuck call there blocks its
thread with no time limit, so EFS work gets its own EFS_SLOTS threads, apart
from anyio's default limiter, which sync routes and sync dependencies share.
Only the wait for a slot has a deadline: a started call keeps its slot until
it returns. One person holds at most EFS_SHARE of the slots, so one heavy
client is refused with ``FilesystemBusy`` before anyone else waits. Both come
from the environment (StorageSettings in cowork.common.settings.app_settings).
A unit that also works on the file store (``run_efs_unit``) holds a pooled
connection while the store answers, so those units together hold at most
EFS_UNIT_SLOTS of the database unit slots (DatabaseSettings), and units stuck
on the store never hold every connection.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, replace
from typing import Any, Literal, TypeVar
from uuid import UUID
from weakref import WeakValueDictionary

import anyio
from anyio.lowlevel import RunVar
from sqlalchemy.engine import Engine
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from starlette.exceptions import HTTPException

from cowork.db import session as db_session
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.turnqueue.models import MAX_ARTIFACT_NAME_CHARS

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DatabaseBusy(PoolTimeoutError):
    """No database connection freed within POOL_TIMEOUT.

    A subclass of SQLAlchemy's pool timeout, so one ``except`` clause and one
    exception handler cover both a unit that waited for its slot and any other
    checkout that waited for a connection.
    """

    status_code = 503


class FilesystemBusy(Exception):
    """No shared-file-store slot freed within POOL_TIMEOUT.

    ``limiter`` names what the caller was waiting for when the time ran out:
    ``efs_share``, its share of the slots (EFS_SHARE), or ``run_efs``, the
    event loop's slots (EFS_SLOTS). ``share`` is that share's key
    (_efs_share_key), None where nobody's share applies. The app answers it
    with 503 and Retry-After (cowork.server), as it does DatabaseBusy, and
    logs both.
    """

    status_code = 503

    def __init__(
        self,
        message: str = "no shared file store slot freed within POOL_TIMEOUT",
        *,
        limiter: Literal["efs_share", "run_efs"] | None = None,
        share: str | None = None,
    ) -> None:
        super().__init__(message)
        self.limiter = limiter
        self.share = share


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
    the order the event loop gave them before units ran in threads. A
    negative MAX_OVERFLOW lets the pool open connections without limit, so
    units then have no slot limit either.
    """
    try:
        return _SLOTS.get()
    except LookupError:
        pass
    database = db_session.settings.database
    if engine.dialect.name == "sqlite":
        total = 1
    elif database.max_overflow < 0:
        total = math.inf
    else:
        total = database.pool_size + database.max_overflow
    limiter = anyio.CapacityLimiter(total)
    _SLOTS.set(limiter)
    return limiter


def _threads() -> anyio.CapacityLimiter:
    """The worker threads units and EFS work run on, kept apart from anyio's
    default limiter.

    Sync routes share that limiter's 40 threads, so a unit waiting on the pool
    or a call stuck on the file store must not take one of them. This limiter
    never makes anyone wait. ``run_db`` and ``run_efs`` callers are capped by
    their slots; ``run_on_unit_threads`` callers take no slot, and each of
    their threads waits at most the pool's own timeout for a connection.
    """
    try:
        return _THREADS.get()
    except LookupError:
        pass
    limiter = anyio.CapacityLimiter(math.inf)
    _THREADS.set(limiter)
    return limiter


async def run_on_unit_threads(fn: Callable[[], T]) -> T:
    """Run ``fn`` on the threads units run on, off anyio's default limiter.
    For database work that is no unit: a session's own rollback or close, or
    a sync helper that reads settings through a session of its own
    (get_user_settings), which a coroutine must not run on the event loop.

    A cancel of the caller, an anyio cancel scope's or a native
    ``Task.cancel()`` such as Stop, waits for ``fn`` to return before it
    propagates (``run_to_completion``), so the caller never unwinds while
    ``fn`` still uses what the caller is about to release.
    """
    return await run_to_completion(anyio.to_thread.run_sync(fn, limiter=_threads()))


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
        except BaseException as exc:
            # Starlette answers an HTTPException with no log line, so a 5xx
            # the unit raises, such as a lock that stayed busy, is logged here,
            # as get_session logs the request session's.
            if isinstance(exc, HTTPException):
                db_session.log_http_error(exc)
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


#: How long the lifespan's last step waits for turn-end artifact records
#: (drain_turn_end_records). The chart's grace period counts it
#: (tests/test_chart_values.py).
TURN_END_RECORD_DRAIN_SECONDS = 8.0
#: The most names the line for a cut turn-end record carries in `touched`,
#: the first in sorted order; its `count` says how many the record held.
TURN_END_RECORD_NAMES_LOGGED = 50
#: The most bytes those names take in the line, each as the JSON console
#: line encodes it: a multibyte character costs up to 4 bytes and a control
#: character, which JSON escapes, 6. So the one line an operator searches for
#: stays under the 16 KiB a container runtime splits a line at, whatever
#: characters the names hold.
TURN_END_RECORD_TOUCHED_BYTES = 12 * 1024


@dataclass(frozen=True)
class TurnEndRecord:
    """The turn a turn-end artifact record belongs to, for the line that names
    it when a shutdown cuts the record."""

    conversation_id: UUID
    project_id: str | None
    # The slugs the record is about: the pod's list of the artifacts the
    # turn's tools touched, empty for a pod that sends no usable list, until
    # the diff names the new slugs it writes rows for (note_turn_end_slugs).
    touched: tuple[str, ...]


_turn_end_records: dict[asyncio.Future[Any], TurnEndRecord] = {}


async def run_turn_end_record(*, work: Coroutine[Any, Any, T], record: TurnEndRecord) -> T:
    """``run_to_completion`` for a turn's artifact diff and record, tracked so
    the lifespan's last step waits for it before the event loop closes and
    cancels it (``drain_turn_end_records``)."""
    task = asyncio.ensure_future(work)
    _turn_end_records[task] = record
    task.add_done_callback(lambda done: _turn_end_records.pop(done, None))
    return await _finish_even_if_cancelled(task)


def _logged_names(*, names: list[str]) -> list[str]:
    """The first of ``names`` while they fit in TURN_END_RECORD_TOUCHED_BYTES,
    at most TURN_END_RECORD_NAMES_LOGGED of them. Each name counts as the JSON
    console line encodes it (``ensure_ascii`` off, as
    structured_logging.make_formatter renders), plus its separator; a lone
    surrogate, such as a folder name's byte that is not UTF-8, counts as the
    escape a console writes for it."""
    kept: list[str] = []
    size = 0
    for name in names[:TURN_END_RECORD_NAMES_LOGGED]:
        size += len(json.dumps(name, ensure_ascii=False).encode("utf-8", "backslashreplace")) + 1
        if size > TURN_END_RECORD_TOUCHED_BYTES:
            break
        kept.append(name)
    return kept


def note_turn_end_slugs(*, slugs: Iterable[str]) -> None:
    """Name the slugs the running turn-end record writes rows for, once its
    diff has found them, so the ERROR a shutdown's cut logs names them
    (drain_turn_end_records). The record's own task calls it; any other
    caller changes nothing."""
    task = asyncio.current_task()
    record = _turn_end_records.get(task)
    if record is not None:
        _turn_end_records[task] = replace(record, touched=tuple(slugs))


async def drain_turn_end_records(*, timeout: float) -> None:
    """Wait up to ``timeout`` for the turn-end records still running, for the
    end of shutdown. Each one still running then is cut by the event loop's
    close, so its rows are lost, and an ERROR names its turn and the slugs
    its diff found, or, when the diff had not answered, the pod's list (none
    for a pod that sends no usable list). The line carries the first of them
    in sorted order, none longer than MAX_ARTIFACT_NAME_CHARS, at most
    TURN_END_RECORD_NAMES_LOGGED and TURN_END_RECORD_TOUCHED_BYTES of them
    (_logged_names), and `count`, how many the record held."""
    if not _turn_end_records:
        return
    _, unfinished = await asyncio.wait(list(_turn_end_records), timeout=timeout)
    for task in unfinished:
        record = _turn_end_records[task]
        named = sorted(name for name in record.touched if len(name) <= MAX_ARTIFACT_NAME_CHARS)
        logger.error(
            "A turn's artifact record was still running at shutdown; its rows are lost",
            extra={
                "event": "shutdown_turn_record_lost",
                "conversation_id": str(record.conversation_id),
                "project_id": record.project_id,
                "touched": _logged_names(names=named),
                "count": len(record.touched),
            },
        )


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
    cancelled caller will never see.

    The shield holds off an enclosing anyio cancel scope, which would
    otherwise cancel each wait again on every pass of the event loop.
    """
    with anyio.CancelScope(shield=True):
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


# EFS_SLOTS, the threads EFS work may hold at once per event loop, and
# EFS_SHARE, the slots one person, or one organization's work with no person,
# may hold at once, are settings (StorageSettings.efs_slots and efs_share, 100
# and 10 by default). Nothing caps an organization: ten busy people fill the
# default EFS_SLOTS.
_EFS_SLOTS: RunVar[anyio.CapacityLimiter] = RunVar("cowork_efs_slots")
_EFS_SHARES: RunVar[WeakValueDictionary[str, anyio.CapacityLimiter]] = RunVar("cowork_efs_shares")
_efs_thread = threading.local()


def _efs_slots() -> anyio.CapacityLimiter:
    """This event loop's EFS slots."""
    try:
        return _EFS_SLOTS.get()
    except LookupError:
        pass
    limiter = anyio.CapacityLimiter(db_session.settings.storage.efs_slots)
    _EFS_SLOTS.set(limiter)
    return limiter


def _efs_share_key(scope: TenantScope) -> str | None:
    """The key of the caller's share of the EFS slots (_efs_share): its
    person's user id, ``org:<org_id>`` for an organization's work with no
    person, or None where one person owns the deployment."""
    if not scope.partitioned:
        return None
    return scope.user_id or f"org:{scope.org_id}"


def _efs_share(scope: TenantScope) -> anyio.CapacityLimiter | None:
    """The caller's share of the EFS slots, or None where one person owns the
    deployment (the desktop, the operator view) and there is nobody to keep
    apart. A share lives while someone waits for it or holds it.

    Work in a partitioned scope with no person, such as a channel turn
    (scope_for_org), shares as one person keyed ``org:<org_id>``, so one
    organization's channel bookkeeping cannot take every slot. That key
    never matches a person's: ``TrustedHeaderMiddleware.dispatch``
    (cowork/principal.py) parses X-User-Id as a UUID, so no person's key
    starts with ``org:``.
    """
    key = _efs_share_key(scope)
    if key is None:
        return None
    try:
        shares = _EFS_SHARES.get()
    except LookupError:
        shares = WeakValueDictionary()
        _EFS_SHARES.set(shares)
    share = shares.get(key)
    if share is None:
        share = anyio.CapacityLimiter(db_session.settings.storage.efs_share)
        shares[key] = share
    return share


def on_efs_thread() -> bool:
    """True while the current thread runs work ``run_efs`` started."""
    return getattr(_efs_thread, "active", False)


def _run_marked(fn: Callable[[], T]) -> T:
    # Worker threads are pooled across limiters, so the mark is per call.
    _efs_thread.active = True
    try:
        return fn()
    finally:
        _efs_thread.active = False


class _EfsStart:
    """Whether an EFS task has handed its work to a thread. Read and set on
    the event loop only."""

    started = False


async def _efs_task(
    fn: Callable[[], T],
    *,
    share: anyio.CapacityLimiter | None,
    share_key: str | None,
    unit_slots: tuple[anyio.CapacityLimiter, ...],
    deadline: float,
    start: _EfsStart,
) -> T:
    """Take the caller's share, then a slot, by ``deadline``; for a database
    unit, then each of ``unit_slots`` in order by the same deadline; then run
    ``fn`` in a worker thread, holding them all until the thread ends.

    The file store slot comes first: a caller queued behind a stuck file store
    then holds no unit slot, so database work elsewhere keeps its slots.
    """
    held: list[anyio.CapacityLimiter] = []
    waiting_for: Literal["efs_share", "run_efs"] | None = None
    try:
        with anyio.CancelScope(deadline=deadline) as slot_wait:
            for name, limiter in (("efs_share", share), ("run_efs", _efs_slots())):
                if limiter is not None:
                    waiting_for = name
                    await limiter.acquire()
                    held.append(limiter)
        if slot_wait.cancelled_caught:
            raise FilesystemBusy(limiter=waiting_for, share=share_key)
        if unit_slots:
            with anyio.CancelScope(deadline=deadline) as unit_wait:
                for limiter in unit_slots:
                    await limiter.acquire()
                    held.append(limiter)
            if unit_wait.cancelled_caught:
                raise DatabaseBusy("no database unit slot freed within POOL_TIMEOUT")
        start.started = True
        return await anyio.to_thread.run_sync(_run_marked, fn, limiter=_threads())
    finally:
        for limiter in reversed(held):
            limiter.release()


async def _run_on_efs(
    fn: Callable[[], T], *, scope: TenantScope, deadline: bool, unit_slots: tuple[anyio.CapacityLimiter, ...],
) -> T:
    """``run_efs`` for both of its callers: ``unit_slots`` is what a database
    unit (``run_efs_unit``) takes after its file store slot, in order, or
    nothing."""
    loop = asyncio.get_running_loop()
    until = loop.time() + db_session.settings.database.pool_timeout if deadline else math.inf
    start = _EfsStart()
    work = asyncio.ensure_future(
        _efs_task(
            fn, share=_efs_share(scope), share_key=_efs_share_key(scope), unit_slots=unit_slots, deadline=until,
            start=start,
        )
    )
    work.add_done_callback(_retrieve_outcome)
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError:
        if not start.started:
            work.cancel()
        await _drain(work)
        raise


async def run_efs(fn: Callable[[], T], *, scope: TenantScope, deadline: bool = True) -> T:
    """Run ``fn``, which works on the shared file store, in a worker thread and
    return its result.

    The wait for a slot gives up POOL_TIMEOUT after the call with
    ``FilesystemBusy``. ``deadline=False`` waits for a slot as long as it
    takes, for bookkeeping nothing would redo if it were refused, such as a
    turn's artifact diff. A caller cancelled before ``fn`` starts leaves at
    once and ``fn`` never runs. Once ``fn`` runs, a cancel of the caller waits
    for it to return before it propagates, as ``run_db`` does, so a lock it
    holds is released before the caller unwinds.
    """
    return await _run_on_efs(fn, scope=scope, deadline=deadline, unit_slots=())


async def run_efs_cleanup(fn: Callable[[], T], *, scope: TenantScope) -> T:
    """Run ``fn``, a release step on the shared file store such as a file's
    close, on ``run_efs`` with no deadline, even when the caller is cancelled.

    A cancelled caller's own ``run_efs`` call that has no slot yet never runs
    its work, so a ``finally`` that closes a file through it would leave the
    descriptor to the garbage collector, on the event loop. Here the call runs
    in a task of its own, and a cancel, an anyio cancel scope's or a native
    ``Task.cancel()``, waits for it before it propagates (``run_to_completion``).
    """
    return await run_to_completion(run_efs(fn, scope=scope, deadline=False))


_EFS_UNIT_SLOTS: RunVar[anyio.CapacityLimiter] = RunVar("cowork_efs_unit_slots")


def _efs_unit_slots() -> anyio.CapacityLimiter:
    """This event loop's cap on the database unit slots file units hold at
    once: EFS_UNIT_SLOTS (``DatabaseSettings.efs_unit_slots``). A file unit
    takes it after its file store slot and before ``run_db``'s unit slot, so
    file units stuck on the file store hold at most that many connections and
    database units keep the rest."""
    try:
        return _EFS_UNIT_SLOTS.get()
    except LookupError:
        pass
    limiter = anyio.CapacityLimiter(db_session.settings.database.efs_unit_slots)
    _EFS_UNIT_SLOTS.set(limiter)
    return limiter


def _file_unit_slots(engine: Engine) -> anyio.CapacityLimiter | None:
    """The unit slots a file unit (``run_efs_unit``) takes after its file
    store slot: ``run_db``'s, which count the pool's connections. None on
    SQLite, the desktop's database, whose one slot orders database units
    rather than counting connections: a file unit holding it through a long
    upload or delete would refuse every database unit behind it."""
    if engine.dialect.name == "sqlite":
        return None
    return _slots(engine)


async def run_efs_unit(
    fn: Callable[[ScopedSession], T], *, scope: TenantScope, deadline: bool = True
) -> T:
    """Run ``fn``, a database unit that also works on the shared file store,
    on the file store's threads (``run_efs``), with a ``unit_session``.

    For work whose database and file store steps are not split yet, such as a
    route that creates a project's row and its folder together. It holds its
    connection while the file store answers, as a sync route did, but it takes
    none of anyio's default threads and never runs on the event loop.

    After its file store slot it takes one of the file units' own slots
    (``_efs_unit_slots``), then a unit slot, as ``run_db`` does, by the same
    deadline: the wait for all of them gives up POOL_TIMEOUT after the call,
    with ``FilesystemBusy`` or ``DatabaseBusy``, and its thread never waits
    inside a full pool. A connection that a checkout outside the units still
    holds can keep it waiting for the pool's own timeout, which refuses it
    with ``DatabaseBusy``, as ``unit_session`` does. On SQLite it takes no
    unit slot (``_file_unit_slots``) and runs beside the database units.
    """

    def unit() -> T:
        with unit_session(scope=scope) as session:
            return fn(session)

    database_slots = _file_unit_slots(_engine())
    unit_slots = () if database_slots is None else (_efs_unit_slots(), database_slots)
    return await _run_on_efs(unit, scope=scope, deadline=deadline, unit_slots=unit_slots)


def loop_limiters() -> dict[str, anyio.CapacityLimiter]:
    """The limiters a request can queue on, by name, for this event loop.
    Call it on the loop: each one is per loop."""
    return {
        "anyio_default": anyio.to_thread.current_default_thread_limiter(),
        "run_db": _slots(_engine()),
        "run_efs": _efs_slots(),
        "run_efs_unit": _efs_unit_slots(),
    }


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
