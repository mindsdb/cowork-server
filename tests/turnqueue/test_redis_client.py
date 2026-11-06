"""The Redis clients' timeouts, retry rules and deadline, without a Redis.

tests/integration/test_redis_failover.py drives the same clients against a real
Redis: a dropped connection, a READONLY primary, a lost reply.
"""
import asyncio
import time

import pytest
from redis.exceptions import (
    AuthenticationError,
    AuthorizationError,
    BusyLoadingError,
    ReadOnlyError,
    ResponseError,
)
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from cowork.turnqueue import redis_client
from cowork.turnqueue.redis_client import get_redis, reset_redis


def test_get_redis_is_singleton(monkeypatch):
    monkeypatch.setenv("COWORK_TURN_REDIS_URL", "redis://localhost:6379/0")
    reset_redis()
    a = get_redis()
    b = get_redis()
    assert a is b


def test_every_blocking_read_blocks_for_less_than_the_socket_timeout():
    """redis-py does not widen socket_timeout for a blocking command, so a BLOCK
    at or above it reads an idle stream as a dead socket and reconnects."""
    assert redis_client.REDIS_BLOCK_MS < redis_client.SOCKET_TIMEOUT_S * 1000


@pytest.mark.parametrize(
    "build",
    [
        lambda: redis_client.build_async_redis(url="redis://localhost:6379/0"),
        lambda: redis_client.build_sync_redis(url="redis://localhost:6379/0"),
    ],
    ids=["async", "sync"],
)
def test_both_clients_carry_the_timeouts_and_the_retried_errors(build):
    """A Retry with no retry_on_error retries nothing in redis-py 5, and a
    client with no socket timeout hangs on a primary that stopped answering."""
    connection = build().connection_pool.make_connection()

    assert connection.socket_connect_timeout == redis_client.SOCKET_CONNECT_TIMEOUT_S
    assert connection.socket_timeout == redis_client.SOCKET_TIMEOUT_S
    assert connection.health_check_interval == redis_client.HEALTH_CHECK_INTERVAL_S
    assert set(connection.retry_on_error) == set(redis_client.RETRIED_ERRORS)
    assert set(redis_client.RETRIED_ERRORS) <= set(connection.retry._supported_errors)


RETRY_CLASSES = pytest.mark.parametrize(
    "retry_class", [redis_client._AsyncDeadlineRetry, redis_client._SyncDeadlineRetry], ids=["async", "sync"],
)


async def _call_with_retry(*, retry, do):
    """``retry.call_with_retry`` on a plain ``do``, for either client's retry class."""
    if isinstance(retry, redis_client._AsyncDeadlineRetry):
        async def attempt():
            return do()

        async def fail(_error):
            return None

        return await retry.call_with_retry(attempt, fail)
    return retry.call_with_retry(do, lambda _error: None)


@RETRY_CLASSES
@pytest.mark.parametrize("error", [AuthenticationError("WRONGPASS"), AuthorizationError("denied")])
async def test_an_auth_error_is_raised_on_its_first_failure(retry_class, error):
    """redis-py makes both ConnectionError subclasses, which the retry list
    names, so without the exclusion a wrong password is resent for the whole
    budget."""
    retry = retry_class(budget_s=5)
    calls = 0

    def do():
        nonlocal calls
        calls += 1
        raise error

    started = time.monotonic()
    with pytest.raises(type(error)):
        await _call_with_retry(retry=retry, do=do)

    assert calls == 1
    assert time.monotonic() - started < 1


async def test_a_retry_loop_inside_another_spends_the_outer_deadline():
    """redis-py reconnects inside a command's retry loop, on the same Retry.
    With a deadline per loop, the inner loop starts a new budget after the outer
    one has spent part of its own, and the call runs past its budget."""
    retry = redis_client._AsyncDeadlineRetry(budget_s=0.6)

    async def refused():
        raise ConnectionRefusedError("refused")

    async def fail(_error):
        return None

    async def command():
        await asyncio.sleep(0.3)
        await retry.call_with_retry(refused, fail)

    started = time.monotonic()
    with pytest.raises(ConnectionRefusedError):
        await retry.call_with_retry(command, fail)

    assert time.monotonic() - started < 0.6 + 0.2


def test_a_retry_loop_inside_another_spends_the_outer_deadline_on_the_blocking_client():
    """The blocking client's twin: its pool's connect retry runs inside the
    command's, on the same deadline."""
    retry = redis_client._SyncDeadlineRetry(budget_s=0.6)

    def refused():
        raise ConnectionRefusedError("refused")

    def command():
        time.sleep(0.3)
        retry.call_with_retry(refused, lambda _error: None)

    started = time.monotonic()
    with pytest.raises(ConnectionRefusedError):
        retry.call_with_retry(command, lambda _error: None)

    assert time.monotonic() - started < 0.6 + 0.2


@RETRY_CLASSES
async def test_a_failure_records_when_it_happened(retry_class):
    retry = retry_class(budget_s=0.2)
    before = time.monotonic()

    def do():
        raise RedisConnectionError("gone")

    with pytest.raises(RedisConnectionError):
        await _call_with_retry(retry=retry, do=do)

    assert redis_client.last_redis_failure_at() >= before


@pytest.mark.parametrize(
    "error, resent",
    [
        (ReadOnlyError("READONLY You can't write against a read only replica."), True),
        (BusyLoadingError("LOADING Redis is loading the dataset in memory"), True),
        (RedisConnectionError("Connection closed by server."), False),
        (RedisTimeoutError("Timeout reading from socket"), False),
    ],
    ids=["readonly", "loading", "connection-closed", "timeout"],
)
async def test_a_write_sent_once_is_sent_again_only_after_redis_refused_it(monkeypatch, error, resent):
    """READONLY and LOADING say Redis refused the command without running it,
    so sending it again cannot duplicate it. Any other failure may have come
    after Redis applied it."""
    client = redis_client.build_async_redis(url="redis://localhost:6379/0", call_budget_s=2)
    pool = client.connection_pool
    connection = pool.make_connection()
    sends: list[str] = []

    async def get_connection(*_args, **_kwargs):
        return connection

    async def release(_connection):
        return None

    async def disconnect(*_args, **_kwargs):
        return None

    async def send_command_parse_response(_conn, command_name, *_args, **_options):
        sends.append(command_name)
        if len(sends) == 1:
            raise error
        return "1-0"

    monkeypatch.setattr(pool, "get_connection", get_connection)
    monkeypatch.setattr(pool, "release", release)
    monkeypatch.setattr(connection, "disconnect", disconnect)
    monkeypatch.setattr(client, "_send_command_parse_response", send_command_parse_response)

    with redis_client.send_once():
        if resent:
            assert await client.xadd("jobs", {"payload": "job"}) == "1-0"
        else:
            with pytest.raises(type(error)):
                await client.xadd("jobs", {"payload": "job"})

    assert sends == (["XADD", "XADD"] if resent else ["XADD"])


class _UnblockedOnce:
    """A client whose first XREAD gets the reply a demoted primary sends its blocked readers."""

    def __init__(self, *, message: str) -> None:
        self._message = message
        self.reads = 0

    async def xread(self, streams, count=None, block=None):
        self.reads += 1
        if self.reads == 1:
            raise ResponseError(self._message)
        return [["s", [["1-0", {"k": "v"}]]]]


async def test_an_unblocked_reply_reads_as_an_empty_read():
    client = _UnblockedOnce(message="UNBLOCKED force unblock from blocking operation, instance state changed (master -> replica?)")
    before = time.monotonic()

    assert await redis_client.xread_blocking(r=client, streams={"s": "0-0"}) == []
    assert await redis_client.xread_blocking(r=client, streams={"s": "0-0"}) == [["s", [["1-0", {"k": "v"}]]]]
    assert redis_client.last_redis_failure_at() >= before


async def test_any_other_error_reply_is_raised():
    client = _UnblockedOnce(message="WRONGTYPE Operation against a key holding the wrong kind of value")

    with pytest.raises(ResponseError, match="WRONGTYPE"):
        await redis_client.xread_blocking(r=client, streams={"s": "0-0"})
