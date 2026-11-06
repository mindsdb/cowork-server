"""Redis clients for the turn queue and the Redis stream backend.

``get_redis`` and ``get_sync_redis`` build one module-level client each on
``COWORK_TURN_REDIS_URL``, on first use. ``reset_redis`` drops them (used by
tests, and by anything that needs to pick up a changed URL).

Hosted Cowork reaches Redis through one URL whose Service follows the primary,
so both clients ride through a dropped connection, a Redis restart or a primary
failover without failing the call in flight:

- A call retries ConnectionError, TimeoutError, ReadOnlyError and OSError with
  jittered exponential backoff until one deadline, ``ASYNC_CALL_BUDGET_S`` or
  ``SYNC_CALL_BUDGET_S`` after it started. Every retry opens a new connection,
  so after a READONLY reply from a demoted primary the next attempt goes back
  through the URL and reaches the new one.
- AuthenticationError and AuthorizationError are raised at once. redis-py
  makes them ConnectionError subclasses, so without the explicit exclusion a
  wrong password would be resent for the whole budget.
- The deadline lives in a context variable (``redis_deadline``). redis-py runs
  retry loops inside a command's own loop (a reconnect, a health-check PING),
  and those spend what is left of the call's budget instead of a new one.
- The async pool connects outside its lock, so callers that need a connection
  do not queue behind one caller's connect retries.
- ``send_once`` sends a command at most once, for the writes that duplicate
  when resent after a lost reply.
- Blocking reads use ``REDIS_BLOCK_MS``, below ``SOCKET_TIMEOUT_S``, so a block
  never reads as a dead socket. ``xread_blocking`` reads the ``-UNBLOCKED``
  reply a demoted primary sends its blocked readers as an empty read.
- ``last_redis_failure_at`` says when this process last saw a Redis error.
"""
from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar

import redis as syncredis
import redis.asyncio as aioredis
from redis.asyncio.client import Pipeline as AsyncPipeline
from redis.asyncio.retry import Retry as AsyncRetry
from redis.backoff import ExponentialWithJitterBackoff
from redis.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ReadOnlyError,
    ResponseError,
)
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from redis.retry import Retry as SyncRetry

T = TypeVar("T")

#: How long opening a TCP connection to Redis may take.
SOCKET_CONNECT_TIMEOUT_S = 2.0
#: How long any one reply may take. Above every blocking read's BLOCK
#: (``REDIS_BLOCK_MS``), so a reader blocked on an idle stream is never cut as
#: a dead socket.
SOCKET_TIMEOUT_S = 5.0
#: BLOCK, in milliseconds, for every blocking read this server makes.
REDIS_BLOCK_MS = 2000
#: A pooled connection idle this long is PINGed before its next command, so a
#: connection a failover left half-open is replaced before a write goes out on it.
HEALTH_CHECK_INTERVAL_S = 15
#: How long one call on the async client keeps retrying. Sized for a Sentinel
#: failover that notices a lost primary within about 10 s, then promotes a
#: replica and moves the Service to it; a slower failover outlasts it.
ASYNC_CALL_BUDGET_S = 30.0
#: The same for the blocking client, whose callers each hold a threadpool thread.
SYNC_CALL_BUDGET_S = 10.0

#: The errors a call retries. OSError is the refused or timed-out connect, which
#: redis-py's connect loop sees raw before it wraps it in a ConnectionError.
RETRIED_ERRORS: tuple[type[Exception], ...] = (RedisConnectionError, RedisTimeoutError, ReadOnlyError, OSError)
#: RedisConnectionError subclasses that no retry can fix.
NEVER_RETRIED_ERRORS: tuple[type[Exception], ...] = (AuthenticationError, AuthorizationError)

_client: aioredis.Redis | None = None
_sync_client: syncredis.Redis | None = None

_deadline: ContextVar[float | None] = ContextVar("cowork_redis_deadline", default=None)
_send_once: ContextVar[bool] = ContextVar("cowork_redis_send_once", default=False)
_sync_connections_opened: ContextVar[int] = ContextVar("cowork_redis_sync_connections_opened", default=0)
_last_failure_at: float | None = None


def _url() -> str:
    return os.environ.get("COWORK_TURN_REDIS_URL", "redis://localhost:6379/0")


def cancel_flag_key(correlation_id: str) -> str:
    """The key a ``/cancel`` writes and whoever runs the turn polls.

    Three processes agree on this name: the endpoint sets it, the producer
    clears a stale one before each turn, and scratchpad-controller's
    ``_cancel_key`` rebuilds it. Kept in one place on this side so the local
    users cannot drift from each other.
    """
    return f"cowork:cancel:{correlation_id}"


# The one cancel reason that changes what a turn saves: the UI cancelled it
# after hearing nothing for its idle window, so it saves as a stall, not a Stop.
STALLED_CANCEL_REASON = "stalled"


def cancel_cause_key(correlation_id: str) -> str:
    """Why a ``/cancel`` asked this turn to stop, when the reason matters.

    Separate from ``cancel_flag_key`` because scratchpad-controller's
    ``_clear_cancel`` deletes the flag right after it publishes the cancelled
    reply, so the flag may be gone when the replica that owns the turn hears
    of the cancel. Only this server reads and writes it: the replica that
    takes the ``/cancel`` sets or clears it in one transaction with the flag
    (``_request_cancel``), the owner reads it when the cancel comes back, and
    the producer clears a stale one before each turn.
    """
    return f"cowork:cancel_cause:{correlation_id}"


def answer_queue_key(correlation_id: str) -> str:
    """The list /answer pushes a remote turn's ask_user answers onto.

    scratchpad-controller's ``answers.answer_key`` rebuilds the same name and
    pops it on the replica running the turn; kept here for the same reason as
    ``cancel_flag_key``.
    """
    return f"cowork:answer:{correlation_id}"


def reply_stream_key(conversation_id: str) -> str:
    """The stream scratchpad-controller relays a turn's replies onto.

    Both the producer (writing the job) and ``turnqueue/answers.py`` (reading
    the pod's verdict) need this name; kept here so the two cannot drift.
    """
    return f"scratchpad:reply:{conversation_id}"


@contextmanager
def redis_deadline(*, budget_s: float) -> Iterator[float]:
    """Run the Redis calls inside under one deadline, ``budget_s`` from now.

    An enclosing deadline that ends sooner wins, so a retry loop that starts
    inside another (redis-py reconnects inside a command's retry loop) spends
    what is left of the outer budget rather than a new one. Yields the
    deadline, a ``time.monotonic()`` value.
    """
    deadline = time.monotonic() + budget_s
    enclosing = _deadline.get()
    if enclosing is not None:
        deadline = min(deadline, enclosing)
    token = _deadline.set(deadline)
    try:
        yield deadline
    finally:
        _deadline.reset(token)


@contextmanager
def send_once() -> Iterator[None]:
    """Send each Redis command inside at most once.

    For a write that duplicates when it is sent twice, such as an XADD or an
    RPUSH whose reply was lost after Redis applied it: the failure is raised
    instead of resending the command. Retries that cannot send it twice still
    happen: opening a connection before the command goes out, and a READONLY
    reply, which says Redis refused the write.
    """
    token = _send_once.set(True)
    try:
        yield
    finally:
        _send_once.reset(token)


def last_redis_failure_at() -> float | None:
    """``time.monotonic()`` when this process last saw a Redis call fail, or None.

    Set by every error a retry loop catches, whether it retried or raised, and
    by an ``-UNBLOCKED`` read. Something that would infer a lost turn from a
    missing key can hold off while this is recent.
    """
    return _last_failure_at


def sync_connections_opened() -> int:
    """How many connections the blocking client opened for calls in this context."""
    return _sync_connections_opened.get()


def _note_failure() -> None:
    global _last_failure_at
    _last_failure_at = time.monotonic()


def _backoff() -> ExponentialWithJitterBackoff:
    return ExponentialWithJitterBackoff(cap=2.0, base=0.25)


def _resend_refused(*, error: Exception) -> bool:
    """Whether ``send_once`` forbids resending a command that failed with ``error``.

    A READONLY reply says Redis refused the write, so resending it cannot
    duplicate it. Any other failure may have come after Redis applied it.
    """
    return _send_once.get() and not isinstance(error, ReadOnlyError)


class _AsyncDeadlineRetry(AsyncRetry):
    """Retries a redis-py operation until the call's deadline, never on an auth error."""

    __slots__ = ("_budget_s",)

    def __init__(self, *, budget_s: float) -> None:
        # The deadline ends the loop, so the attempt count is unbounded.
        super().__init__(_backoff(), -1, RETRIED_ERRORS)
        self._budget_s = budget_s

    async def call_with_retry(self, do: Callable[[], Awaitable[T]], fail: Callable[[Exception], Any]) -> T:
        with redis_deadline(budget_s=self._budget_s) as deadline:
            self._backoff.reset()
            failures = 0
            while True:
                try:
                    return await do()
                except self._supported_errors as error:
                    _note_failure()
                    if isinstance(error, NEVER_RETRIED_ERRORS):
                        raise
                    failures += 1
                    await fail(error)
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise error
                    # Capped at what is left, so the last attempt runs at the deadline.
                    await asyncio.sleep(min(self._backoff.compute(failures), left))


class _SyncDeadlineRetry(SyncRetry):
    """The blocking client's ``_AsyncDeadlineRetry``."""

    def __init__(self, *, budget_s: float) -> None:
        super().__init__(_backoff(), -1, RETRIED_ERRORS)
        self._budget_s = budget_s

    def call_with_retry(self, do: Callable[[], T], fail: Callable[[Exception], Any]) -> T:
        with redis_deadline(budget_s=self._budget_s) as deadline:
            self._backoff.reset()
            failures = 0
            while True:
                try:
                    return do()
                except self._supported_errors as error:
                    _note_failure()
                    if isinstance(error, NEVER_RETRIED_ERRORS):
                        raise
                    failures += 1
                    fail(error)
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise error
                    time.sleep(min(self._backoff.compute(failures), left))


class _AsyncPool(aioredis.ConnectionPool):
    """An async pool that connects outside its lock, and holds its clients' call budget.

    redis-py 5's pool holds its lock while a new connection connects, retries
    included, so every caller that needs a connection, even one an idle pooled
    connection would serve, waits behind that one connect. This takes the
    connection under the lock and connects it after, as redis-py 8 does.
    """

    def __init__(self, *, call_budget_s: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.call_budget_s = call_budget_s

    async def get_connection(self, command_name=None, *keys, **options):
        async with self._lock:
            connection = self.get_available_connection()
        try:
            await self.ensure_connection(connection)
        except BaseException:
            await self.release(connection)
            raise
        return connection


class _SyncPool(syncredis.ConnectionPool):
    """A blocking pool that counts, per context, the connections it opens.

    redis-py 5's blocking pool already connects outside its lock.
    """

    def __init__(self, *, call_budget_s: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.call_budget_s = call_budget_s

    def make_connection(self):
        connection = super().make_connection()
        connection.register_connect_callback(self._count_connect)
        return connection

    def _count_connect(self, connection) -> None:
        _sync_connections_opened.set(_sync_connections_opened.get() + 1)


class _AsyncPipeline(AsyncPipeline):
    """A pipeline that runs each execute under one deadline and honors ``send_once``."""

    connection_pool: _AsyncPool

    async def execute(self, raise_on_error: bool = True):
        with redis_deadline(budget_s=self.connection_pool.call_budget_s):
            return await super().execute(raise_on_error)

    async def immediate_execute_command(self, *args, **options):
        with redis_deadline(budget_s=self.connection_pool.call_budget_s):
            return await super().immediate_execute_command(*args, **options)

    async def _disconnect_raise_reset(self, conn, error):
        await super()._disconnect_raise_reset(conn, error)
        if _resend_refused(error=error):
            await self.reset()
            raise error


class _AsyncRedis(aioredis.Redis):
    """An async client that runs each call under one deadline and honors ``send_once``.

    The deadline covers taking a connection from the pool too, so a connect
    that retries and the command's own retries share one budget.
    """

    connection_pool: _AsyncPool

    async def execute_command(self, *args, **options):
        with redis_deadline(budget_s=self.connection_pool.call_budget_s):
            return await super().execute_command(*args, **options)

    async def _disconnect_raise(self, conn, error):
        await super()._disconnect_raise(conn, error)
        if _resend_refused(error=error):
            raise error

    def pipeline(self, transaction: bool = True, shard_hint: str | None = None) -> _AsyncPipeline:
        return _AsyncPipeline(self.connection_pool, self.response_callbacks, transaction, shard_hint)


class _SyncRedis(syncredis.Redis):
    """A blocking client that runs each call under one deadline.

    Nothing uses its pipelines, which get no call-wide deadline: their connect
    and command retry loops each start one.
    """

    connection_pool: _SyncPool

    def execute_command(self, *args, **options):
        with redis_deadline(budget_s=self.connection_pool.call_budget_s):
            return super().execute_command(*args, **options)


def _client_options() -> dict[str, Any]:
    """The redis-py connection options both clients share."""
    return {
        "decode_responses": True,
        "socket_connect_timeout": SOCKET_CONNECT_TIMEOUT_S,
        "socket_timeout": SOCKET_TIMEOUT_S,
        "health_check_interval": HEALTH_CHECK_INTERVAL_S,
        # redis-py 5 retries only the errors listed here: a Retry alone retries nothing.
        "retry_on_error": list(RETRIED_ERRORS),
    }


def build_async_redis(*, url: str, call_budget_s: float = ASYNC_CALL_BUDGET_S) -> aioredis.Redis:
    """An async client on ``url`` with this module's timeouts, retries and deadline."""
    pool = _AsyncPool.from_url(
        url, call_budget_s=call_budget_s, retry=_AsyncDeadlineRetry(budget_s=call_budget_s), **_client_options(),
    )
    return _AsyncRedis.from_pool(pool)


def build_sync_redis(*, url: str, call_budget_s: float = SYNC_CALL_BUDGET_S) -> syncredis.Redis:
    """A blocking client on ``url`` with this module's timeouts, retries and deadline."""
    pool = _SyncPool.from_url(
        url, call_budget_s=call_budget_s, retry=_SyncDeadlineRetry(budget_s=call_budget_s), **_client_options(),
    )
    return _SyncRedis.from_pool(pool)


async def xread_blocking(
    *, r: aioredis.Redis, streams: dict[str, str], count: int | None = None, block_ms: int = REDIS_BLOCK_MS,
) -> list:
    """XREAD with BLOCK, reading an ``-UNBLOCKED`` reply as an empty read.

    A primary demoted to a replica, by a failover or REPLICAOF, answers each
    reader blocked on it ``-UNBLOCKED``. Nothing was read and nothing was lost,
    so the caller's loop reads again from the same position.
    """
    try:
        return await r.xread(streams, count=count, block=block_ms)
    except ResponseError as error:
        if not str(error).startswith("UNBLOCKED"):
            raise
        _note_failure()
        return []


def get_redis() -> aioredis.Redis:
    global _client
    if _client is None:
        _client = build_async_redis(url=_url())
    return _client


def get_sync_redis() -> syncredis.Redis:
    """Blocking client on the same URL, for callers with no event loop.

    Conversation delete runs in a threadpool thread (the endpoint is a sync
    ``def``), so it cannot await the async client.
    """
    global _sync_client
    if _sync_client is None:
        _sync_client = build_sync_redis(url=_url())
    return _sync_client


def reset_redis() -> None:
    global _client, _sync_client
    _client = None
    _sync_client = None
