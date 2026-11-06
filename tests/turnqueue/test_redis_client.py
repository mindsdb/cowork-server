"""The Redis clients' timeouts, retry rules and deadline, without a Redis.

tests/integration/test_redis_failover.py drives the same clients against a real
Redis: a dropped connection, a READONLY primary, a lost reply.
"""
import asyncio
import time

import pytest
from redis.exceptions import AuthenticationError, AuthorizationError, ResponseError
from redis.exceptions import ConnectionError as RedisConnectionError

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


@pytest.mark.parametrize("error", [AuthenticationError("WRONGPASS"), AuthorizationError("denied")])
async def test_an_auth_error_is_raised_on_its_first_failure(error):
    """redis-py makes both ConnectionError subclasses, which the retry list
    names, so without the exclusion a wrong password is resent for the whole
    budget."""
    retry = redis_client._AsyncDeadlineRetry(budget_s=5)
    calls = 0

    async def do():
        nonlocal calls
        calls += 1
        raise error

    async def fail(_error):
        return None

    started = time.monotonic()
    with pytest.raises(type(error)):
        await retry.call_with_retry(do, fail)

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


async def test_a_failure_records_when_it_happened():
    retry = redis_client._AsyncDeadlineRetry(budget_s=0.2)
    before = time.monotonic()

    async def do():
        raise RedisConnectionError("gone")

    async def fail(_error):
        return None

    with pytest.raises(RedisConnectionError):
        await retry.call_with_retry(do, fail)

    assert redis_client.last_redis_failure_at() >= before


class _UnblockedOnce:
    """A client whose first XREAD gets the reply a demoted primary sends its blocked readers."""

    def __init__(self, message: str) -> None:
        self._message = message
        self.reads = 0

    async def xread(self, streams, count=None, block=None):
        self.reads += 1
        if self.reads == 1:
            raise ResponseError(self._message)
        return [["s", [["1-0", {"k": "v"}]]]]


async def test_an_unblocked_reply_reads_as_an_empty_read():
    client = _UnblockedOnce("UNBLOCKED force unblock from blocking operation, instance state changed (master -> replica?)")
    before = time.monotonic()

    assert await redis_client.xread_blocking(r=client, streams={"s": "0-0"}) == []
    assert await redis_client.xread_blocking(r=client, streams={"s": "0-0"}) == [["s", [["1-0", {"k": "v"}]]]]
    assert redis_client.last_redis_failure_at() >= before


async def test_any_other_error_reply_is_raised():
    client = _UnblockedOnce("WRONGTYPE Operation against a key holding the wrong kind of value")

    with pytest.raises(ResponseError, match="WRONGTYPE"):
        await redis_client.xread_blocking(r=client, streams={"s": "0-0"})
