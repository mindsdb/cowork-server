"""Two independent readers against one real Redis.

Every other test fakes half the split. This is the arrangement that runs in
production: the reader shares nothing with the writer except Redis.
"""

from __future__ import annotations

import asyncio
import os

import pytest
import redis.asyncio as aioredis

from cowork.streaming import turn_index
from cowork.streaming.buffer import RedisStreamBuffer

pytestmark = pytest.mark.integration


@pytest.fixture
async def redis_url(monkeypatch):
    url = os.environ.get("COWORK_TEST_REDIS_URL", "redis://localhost:6379/1")
    client = aioredis.from_url(url, decode_responses=True)
    try:
        await client.ping()
    except Exception:
        pytest.skip("no Redis reachable for integration test")
    await client.flushdb()
    monkeypatch.setenv("COWORK_TURN_REDIS_URL", url)
    monkeypatch.setenv("COWORK_STREAM_BACKEND", "redis")
    from cowork.turnqueue.redis_client import reset_redis
    reset_redis()
    yield url
    await client.aclose()
    reset_redis()


@pytest.mark.asyncio
async def test_a_second_reader_follows_a_turn_it_did_not_start(redis_url):
    await turn_index.record_turn(
        "conv-x", turn_id=1, correlation_id="corr-x", org_id="o1", user_id="u1")

    writer = RedisStreamBuffer(conversation_id="conv-x", turn_id=1)
    await writer.append("sse", {"sse": "event: a\ndata: {}\n\n"})

    # Replica B: knows nothing except what Redis says.
    turn = await turn_index.get_turn("conv-x")
    assert turn["correlation_id"] == "corr-x"
    reader = RedisStreamBuffer(
        conversation_id="conv-x", turn_id=int(turn["turn_id"]))
    await reader.refresh()
    assert reader.latest_seq == 1
    assert reader.is_closed is False

    seen = []

    async def follow():
        async for rec in reader.tail(from_seq=0):
            seen.append(rec)

    task = asyncio.create_task(follow())
    await asyncio.sleep(0.1)
    await writer.append("sse", {"sse": "event: b\ndata: {}\n\n"})
    await writer.close("completed")
    await asyncio.wait_for(task, timeout=10)

    assert [r.data.get("sse") for r in seen[:2]] == [
        "event: a\ndata: {}\n\n", "event: b\ndata: {}\n\n"]
    assert seen[-1].is_terminal


@pytest.mark.asyncio
async def test_a_finished_turn_reads_as_closed_from_anywhere(redis_url):
    """Liveness is the terminal record, so it survives the process that wrote it."""
    await turn_index.record_turn(
        "conv-y", turn_id=1, correlation_id="corr-y", org_id=None, user_id=None)
    writer = RedisStreamBuffer(conversation_id="conv-y", turn_id=1)
    await writer.append("sse", {"sse": "event: a\ndata: {}\n\n"})
    await writer.close("completed")

    reader = RedisStreamBuffer(conversation_id="conv-y", turn_id=1)
    await reader.refresh()

    assert reader.is_closed is True
    assert reader.latest_seq == 2
