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

from cowork.streaming import turn_claim, turn_index
from cowork.streaming.backend import new_buffer
from cowork.streaming.buffer import RedisStreamBuffer
from cowork.streaming.registry import RunRegistry, TurnInProgress, TurnLifecycle
from cowork.streaming.turn_claim import CLAIMS_KEY, BootPassResult, TurnClaim, claim_key
from cowork.turnqueue.redis_client import cancel_flag_key
from tests.integration.prereq import missing_prerequisite, report_redis_version

pytestmark = [pytest.mark.integration, pytest.mark.redis_service]


@pytest.fixture
async def redis_url(monkeypatch, pytestconfig):
    url = os.environ.get("COWORK_TEST_REDIS_URL", "redis://localhost:6379/1")
    client = aioredis.from_url(url, decode_responses=True)
    try:
        await client.ping()
    except Exception:
        missing_prerequisite(
            f"no Redis reachable at {url}", required_by="COWORK_REQUIRE_SERVICES")
    await report_redis_version(client=client, config=pytestconfig)
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
    assert turn.correlation_id == "corr-x"
    reader = RedisStreamBuffer(conversation_id="conv-x", turn_id=turn.turn_id)
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
        assert (await turn_index.get_turn("conv-dup")).run_id == first.run_id
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


@pytest.mark.asyncio
async def test_a_stale_send_that_waited_for_the_other_replicas_turn_is_refused(redis_url):
    """Replica B's send read the history before replica A's turn 2 saved its
    question, and reached B while A wrote its final frame. B waits for A to
    release, and the stamp A's release writes refuses it in the transaction
    that would take the claim, so A's stream keeps its records."""
    replica_a, replica_b = RunRegistry(), RunRegistry()
    final_written, release = asyncio.Event(), asyncio.Event()

    async def finishing(buffer):
        await buffer.append("sse", {"sse": "event: a\ndata: {}\n\n"})
        await buffer.append("sse", {"sse": "event: response.completed\ndata: {}\n\n"})
        final_written.set()
        await release.wait()
        await buffer.close("completed")

    async def answer(buffer):
        await buffer.close("completed")

    first = await replica_a.start(
        conversation_id="conv-stale", turn_id=2, open_buffer=lambda: new_buffer("conv-stale", 2),
        produce=finishing, lifecycle=TurnLifecycle(saved_question=True),
    )
    await asyncio.wait_for(final_written.wait(), timeout=10)
    stale = asyncio.create_task(replica_b.start(
        conversation_id="conv-stale", turn_id=2, open_buffer=lambda: new_buffer("conv-stale", 2),
        produce=answer,
    ))
    await asyncio.sleep(0.3)
    assert not stale.done()
    release.set()
    await asyncio.wait_for(first.task, timeout=10)

    with pytest.raises(TurnInProgress) as refused:
        await asyncio.wait_for(stale, timeout=15)

    client = aioredis.from_url(redis_url, decode_responses=True)
    try:
        assert refused.value.turn_id == 2
        assert await client.xlen("cowork:stream:conv-stale:2") == 3
        assert await client.exists(claim_key("conv-stale"), CLAIMS_KEY) == 0
    finally:
        await client.aclose()


async def _until(condition, *, timeout: float = 10) -> None:
    async with asyncio.timeout(timeout):
        while not await condition():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_starts_racing_on_one_conversation_admit_exactly_one(redis_url):
    """Four replicas get the same send at the same instant, and each finds no
    holder. Real Redis replies arrive off a socket, so the four transactions
    interleave, and the claim's compare-and-set admits exactly one."""
    replicas = [RunRegistry() for _ in range(4)]
    release = asyncio.Event()

    async def answer_once_released(buffer):
        await buffer.append("sse", {"sse": "event: a\ndata: {}\n\n"})
        await release.wait()
        await buffer.close("completed")

    results = await asyncio.gather(*(
        runs.start(
            conversation_id="conv-race", turn_id=1, open_buffer=lambda: new_buffer("conv-race", 1),
            produce=answer_once_released,
        )
        for runs in replicas
    ), return_exceptions=True)

    admitted = [result for result in results if not isinstance(result, BaseException)]
    assert len(admitted) == 1, results
    assert all(isinstance(result, TurnInProgress) for result in results if result is not admitted[0]), results
    release.set()
    await asyncio.wait_for(admitted[0].task, timeout=10)


@pytest.mark.asyncio
async def test_a_running_turn_renews_its_claim_and_takes_it_back_once_it_is_gone(redis_url, monkeypatch):
    """Renewal on Redis 8: the WATCH on the claim, and the second WATCH on the
    turn's stream once the key is gone, which a renewal makes after its GET."""
    monkeypatch.setattr(turn_claim, "RENEW_INTERVAL_SECONDS", 0.05)
    started, release = asyncio.Event(), asyncio.Event()

    async def answer_once_released(buffer):
        await buffer.append("sse", {"sse": "event: a\ndata: {}\n\n"})
        started.set()
        await release.wait()
        await buffer.close("completed")

    handle = await RunRegistry().start(
        conversation_id="conv-renew", turn_id=1, open_buffer=lambda: new_buffer("conv-renew", 1),
        produce=answer_once_released,
    )
    await asyncio.wait_for(started.wait(), timeout=10)
    client = aioredis.from_url(redis_url, decode_responses=True)
    try:
        await client.pexpire(claim_key("conv-renew"), 2_000)

        async def renewed() -> bool:
            return await client.pttl(claim_key("conv-renew")) > 60_000

        await _until(renewed)
        await client.delete(claim_key("conv-renew"))

        async def taken_back() -> bool:
            return await client.get(claim_key("conv-renew")) == handle.claim.model_dump_json()

        await _until(taken_back)
        release.set()
        await asyncio.wait_for(handle.task, timeout=10)
        assert await client.exists(claim_key("conv-renew"), CLAIMS_KEY) == 0
    finally:
        release.set()
        await client.aclose()


@pytest.mark.asyncio
async def test_the_boot_pass_releases_this_pods_claims_and_stops_their_sandbox_turns(redis_url):
    """The boot pass on Redis 8: one HGETALL and one MGET over every listing,
    then a compare-and-delete per claim."""
    def claim(run_id: str, pod: str) -> TurnClaim:
        return TurnClaim(run_id=run_id, turn_id=1, correlation_id=f"corr-{run_id}", pod=pod, started_at=1.0)

    for conversation_id, run_id, pod in (
        ("own-held", "run-held", "pod-a"), ("own-lapsed", "run-lapsed", "pod-a"),
        ("other-live", "run-live", "pod-b"), ("other-lapsed", "run-gone", "pod-old"),
    ):
        await turn_claim.acquire(claim=claim(run_id, pod), conversation_id=conversation_id)
    client = aioredis.from_url(redis_url, decode_responses=True)
    try:
        await client.delete(claim_key("own-lapsed"), claim_key("other-lapsed"))

        result = await turn_claim.release_pod_claims(pod="pod-a")

        assert result == BootPassResult(released=1, pruned=1)
        assert await client.hkeys(CLAIMS_KEY) == ["other-live"]
        assert await client.exists(claim_key("own-held")) == 0
        assert await client.get(claim_key("other-live")) == claim("run-live", "pod-b").model_dump_json()
        flagged = {
            run_id: await client.exists(cancel_flag_key(f"corr-{run_id}"))
            for run_id in ("run-held", "run-lapsed", "run-live", "run-gone")
        }
        assert flagged == {"run-held": 1, "run-lapsed": 1, "run-live": 0, "run-gone": 0}
    finally:
        await client.aclose()
