"""Two independent readers against one real Redis.

Every other test fakes half the split. This is the arrangement that runs in
production: the reader shares nothing with the writer except Redis.

`.github/workflows/tests-services.yml` runs these against a Redis service
container on every pull request. `make test/integration` leaves them out,
since the deployed environments it targets give the runner no Redis of its own.
"""

from __future__ import annotations

import asyncio
import os

import pytest
import redis.asyncio as aioredis

from cowork.streaming import turn_index
from cowork.streaming.backend import new_buffer
from cowork.streaming.buffer import RedisStreamBuffer
from cowork.streaming.registry import RunRegistry, TurnInProgress
from cowork.streaming.turn_claim import CLAIMS_KEY, claim_key
from tests.integration.prereq import missing_prerequisite

pytestmark = [pytest.mark.integration, pytest.mark.redis_service]


@pytest.fixture
async def redis_url(monkeypatch):
    url = os.environ.get("COWORK_TEST_REDIS_URL", "redis://localhost:6379/1")
    client = aioredis.from_url(url, decode_responses=True)
    try:
        await client.ping()
    except Exception:
        missing_prerequisite(
            f"no Redis reachable at {url}", required_by="COWORK_REQUIRE_SERVICES")
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


@pytest.mark.asyncio
async def test_a_duplicate_send_on_the_other_replica_is_refused(redis_url):
    """Two registries are two replicas: each knows only its own turns, and
    the turn claim in Redis is what they share."""
    replica_a, replica_b = RunRegistry(), RunRegistry()
    started, release = asyncio.Event(), asyncio.Event()

    async def answer_once_released(buffer):
        await buffer.append("sse", {"sse": "event: a\ndata: {}\n\n"})
        started.set()
        await release.wait()
        await buffer.append("sse", {"sse": "event: response.completed\ndata: {}\n\n"})
        await buffer.close("completed")

    async def answer(buffer):
        await buffer.close("completed")

    first = await replica_a.start(
        conversation_id="conv-dup", turn_id=1, open_buffer=lambda: new_buffer("conv-dup", 1),
        produce=answer_once_released,
    )
    await asyncio.wait_for(started.wait(), timeout=10)
    opened = []

    async def open_on_b():
        opened.append(1)
        return await new_buffer("conv-dup", 1)

    with pytest.raises(TurnInProgress) as refused:
        await replica_b.start(conversation_id="conv-dup", turn_id=1, open_buffer=open_on_b, produce=answer)

    client = aioredis.from_url(redis_url, decode_responses=True)
    try:
        assert refused.value.turn_id == 1
        assert opened == []
        assert (await turn_index.get_turn("conv-dup"))["run_id"] == first.run_id
        assert await client.xlen("cowork:stream:conv-dup:1") == 1
        release.set()
        await asyncio.wait_for(first.task, timeout=10)
        assert await client.exists(claim_key("conv-dup"), CLAIMS_KEY) == 0

        after = await replica_b.start(
            conversation_id="conv-dup", turn_id=3, open_buffer=lambda: new_buffer("conv-dup", 3),
            produce=answer,
        )
        await asyncio.wait_for(after.task, timeout=10)
    finally:
        await client.aclose()
