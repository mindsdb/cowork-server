"""The Redis turn buffer's reader through a Redis blip or failover.

The client resends an append whose reply it lost, which can leave two entries
with one seq, and a demoted primary answers a blocked reader ``-UNBLOCKED``.
tests/integration/test_redis_failover.py covers the same against a real Redis.
"""
import asyncio
import json

import fakeredis.aioredis
import pytest
from redis.exceptions import ResponseError

from cowork.streaming import buffer as buffer_mod
from cowork.streaming.buffer import RedisStreamBuffer, stream_key
from cowork.turnqueue.redis_client import REDIS_BLOCK_MS, SOCKET_TIMEOUT_S


@pytest.fixture
def fake_redis(monkeypatch):
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(buffer_mod, "get_redis", lambda: client)
    return client


async def _write(*, client, key, seq, type_="Delta", text=""):
    """One entry as RedisStreamBuffer.append writes it, so a test can repeat a seq."""
    await client.xadd(key, {"seq": str(seq), "ts": "t", "type": type_, "data": json.dumps({"text": text})})


async def test_a_replayed_seq_written_twice_is_read_once(fake_redis):
    key = stream_key("dup-replay", 1)
    await _write(client=fake_redis, key=key, seq=0, text="a")
    await _write(client=fake_redis, key=key, seq=1, text="b")
    await _write(client=fake_redis, key=key, seq=1, text="b")
    await _write(client=fake_redis, key=key, seq=2, type_="Done")

    reader = RedisStreamBuffer(conversation_id="dup-replay", turn_id=1)
    seen = [rec async for rec in reader.tail(from_seq=0)]

    assert [rec.seq for rec in seen] == [0, 1, 2]


async def test_a_live_seq_written_twice_is_read_once(fake_redis):
    key = stream_key("dup-live", 1)
    await _write(client=fake_redis, key=key, seq=0, text="a")
    reader = RedisStreamBuffer(conversation_id="dup-live", turn_id=1)
    seen = []

    async def read():
        async for rec in reader.tail(from_seq=0):
            seen.append(rec)

    task = asyncio.create_task(read())
    await asyncio.sleep(0.05)  # past the replay, blocked on XREAD
    await _write(client=fake_redis, key=key, seq=1, text="b")
    await _write(client=fake_redis, key=key, seq=1, text="b")
    await _write(client=fake_redis, key=key, seq=2, type_="Done")
    await asyncio.wait_for(task, timeout=5)

    assert [rec.seq for rec in seen] == [0, 1, 2]


async def test_a_reader_reads_again_after_an_unblocked_reply(fake_redis, monkeypatch):
    """A primary demoted while the reader is blocked answers ``-UNBLOCKED``;
    raised, it would end the client's stream mid-answer."""
    real_xread = fake_redis.xread
    reads = 0

    async def xread(*args, **kwargs):
        nonlocal reads
        reads += 1
        if reads == 1:
            raise ResponseError(
                "UNBLOCKED force unblock from blocking operation, instance state changed (master -> replica?)")
        return await real_xread(*args, **kwargs)

    monkeypatch.setattr(fake_redis, "xread", xread)
    writer = RedisStreamBuffer(conversation_id="unblocked", turn_id=1)
    await writer.append("Delta", {"text": "a"})
    reader = RedisStreamBuffer(conversation_id="unblocked", turn_id=1)
    seen = []

    async def read():
        async for rec in reader.tail(from_seq=0):
            seen.append(rec)

    task = asyncio.create_task(read())
    await asyncio.sleep(0.05)
    await writer.append("Delta", {"text": "b"})
    await writer.close("completed")
    await asyncio.wait_for(task, timeout=5)

    assert reads >= 2
    assert [rec.type for rec in seen] == ["Delta", "Delta", "Done"]


def test_the_live_read_blocks_for_less_than_the_socket_timeout():
    assert RedisStreamBuffer._BLOCK_MS == REDIS_BLOCK_MS
    assert RedisStreamBuffer._BLOCK_MS < SOCKET_TIMEOUT_S * 1000
