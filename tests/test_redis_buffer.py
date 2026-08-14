"""Redis-backed turn buffer: the storage any replica can read.

The file backend keeps a turn's records in a file on one pod's disk, so a
second replica cannot serve a reconnect. These tests pin the Redis backend
that replaces it.
"""
import asyncio
import json

import fakeredis.aioredis
import pytest

from cowork.streaming import buffer as buffer_mod
from cowork.streaming.buffer import REDIS_BUFFER_TTL_SECONDS, RedisStreamBuffer, stream_key


@pytest.fixture
def fake_redis(monkeypatch):
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(buffer_mod, "get_redis", lambda: client)
    return client


@pytest.mark.asyncio
async def test_append_assigns_contiguous_seqs(fake_redis):
    buf = RedisStreamBuffer(conversation_id="c1", turn_id=3)
    assert await buf.append("Delta", {"text": "he"}) == 0
    assert await buf.append("Delta", {"text": "llo"}) == 1

    entries = await fake_redis.xrange(stream_key("c1", 3))
    assert [int(fields["seq"]) for _id, fields in entries] == [0, 1]
    assert json.loads(entries[0][1]["data"]) == {"text": "he"}
    assert entries[0][1]["type"] == "Delta"


@pytest.mark.asyncio
async def test_close_writes_one_terminal_record(fake_redis):
    buf = RedisStreamBuffer(conversation_id="c1", turn_id=1)
    await buf.append("Delta", {"text": "hi"})
    await buf.close("completed")
    await buf.close("completed")  # idempotent: a producer may close twice

    entries = await fake_redis.xrange(stream_key("c1", 1))
    assert [fields["type"] for _id, fields in entries] == ["Delta", "Done"]


@pytest.mark.asyncio
async def test_close_carries_extra_into_the_terminal_record(fake_redis):
    buf = RedisStreamBuffer(conversation_id="c1", turn_id=1)
    await buf.close("error", {"message": "boom"})

    _id, fields = (await fake_redis.xrange(stream_key("c1", 1)))[0]
    assert fields["type"] == "Error"
    assert json.loads(fields["data"])["message"] == "boom"


@pytest.mark.asyncio
async def test_every_write_refreshes_the_ttl(fake_redis):
    """Without this every turn's key is immortal. The transcript lives in
    Postgres; this is only a replay buffer."""
    buf = RedisStreamBuffer(conversation_id="c1", turn_id=1)
    await buf.append("Delta", {"text": "a"})
    ttl = await fake_redis.ttl(stream_key("c1", 1))
    assert 0 < ttl <= REDIS_BUFFER_TTL_SECONDS


@pytest.mark.asyncio
async def test_tail_replays_then_stops_at_the_terminal(fake_redis):
    writer = RedisStreamBuffer(conversation_id="c2", turn_id=1)
    await writer.append("Delta", {"text": "a"})
    await writer.append("Delta", {"text": "b"})
    await writer.close("completed")

    reader = RedisStreamBuffer(conversation_id="c2", turn_id=1)
    seen = [rec async for rec in reader.tail(from_seq=0)]

    assert [r.type for r in seen] == ["Delta", "Delta", "Done"]
    assert [r.seq for r in seen] == [0, 1, 2]
    assert seen[-1].is_terminal


@pytest.mark.asyncio
async def test_tail_honours_from_seq(fake_redis):
    writer = RedisStreamBuffer(conversation_id="c3", turn_id=1)
    for text in ("a", "b", "c"):
        await writer.append("Delta", {"text": text})
    await writer.close("completed")

    reader = RedisStreamBuffer(conversation_id="c3", turn_id=1)
    seen = [rec async for rec in reader.tail(from_seq=2)]

    assert [r.seq for r in seen] == [2, 3]


@pytest.mark.asyncio
async def test_tail_follows_a_live_writer(fake_redis):
    """The reader is a separate object, standing in for another replica."""
    writer = RedisStreamBuffer(conversation_id="c4", turn_id=1)
    await writer.append("Delta", {"text": "first"})

    reader = RedisStreamBuffer(conversation_id="c4", turn_id=1)
    seen = []

    async def read():
        async for rec in reader.tail(from_seq=0):
            seen.append(rec)

    task = asyncio.create_task(read())
    await asyncio.sleep(0.05)          # reader is now blocked on XREAD
    await writer.append("Delta", {"text": "second"})
    await writer.close("completed")
    await asyncio.wait_for(task, timeout=5)

    assert [r.data.get("text") for r in seen[:2]] == ["first", "second"]
    assert seen[-1].type == "Done"


@pytest.mark.asyncio
async def test_refresh_reads_state_written_by_another_process(fake_redis):
    writer = RedisStreamBuffer(conversation_id="c5", turn_id=1)
    await writer.append("Delta", {"text": "a"})

    reader = RedisStreamBuffer(conversation_id="c5", turn_id=1)
    await reader.refresh()
    assert reader.latest_seq == 1
    assert reader.is_closed is False

    await writer.close("completed")
    await reader.refresh()
    assert reader.latest_seq == 2
    assert reader.is_closed is True


@pytest.mark.asyncio
async def test_refresh_on_a_stream_that_does_not_exist(fake_redis):
    reader = RedisStreamBuffer(conversation_id="never-ran", turn_id=1)
    await reader.refresh()
    assert reader.latest_seq == 0
    assert reader.is_closed is False
