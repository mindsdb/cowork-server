"""The cross-replica turn claim: one run answers a conversation at a time,
whichever replica a question reaches.

Unit tests on fakeredis. tests/test_responses_multi_replica.py covers
admission through two registries and the endpoints, and
tests/integration/test_two_replicas.py the same against a real Redis.
"""
from __future__ import annotations

import asyncio
import importlib
import logging
from uuid import uuid4

import fakeredis
import fakeredis.aioredis
import pytest
import yaml
from redis.exceptions import ConnectionError as RedisConnectionError

from cowork.streaming import backend as backend_mod
from cowork.streaming import buffer as buffer_mod
from cowork.streaming import cancellation, turn_claim, turn_index
from cowork.streaming.buffer import FileStreamBuffer, RedisStreamBuffer, stream_key
from cowork.streaming.registry import RunRegistry, TurnInProgress, TurnLifecycle
from cowork.streaming.turn_claim import (
    CLAIMS_KEY,
    AcquireOutcome,
    BootPassResult,
    RenewOutcome,
    StreamState,
    TurnClaim,
    TurnClaimUnavailable,
    claim_key,
)
from cowork.turnqueue import redis_client
from tests._logging_assertions import assert_log_fields

from _fakes import opens
from _hosted_command import CHART_VALUES, hosted_command

CLAIM_LOGGER = "cowork.streaming.turn_claim"
COMPLETED = {"sse": "event: response.completed\ndata: {}\n\n"}


@pytest.fixture
def server():
    return fakeredis.FakeServer()


@pytest.fixture
def redis(monkeypatch, server):
    """One Redis for every module that reaches it, on the Redis backend."""
    client = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    for module in (turn_claim, turn_index, buffer_mod, cancellation):
        monkeypatch.setattr(module, "get_redis", lambda: client)
    sync_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    for module in (turn_claim, turn_index, backend_mod):
        monkeypatch.setattr(module, "get_sync_redis", lambda: sync_client)
    monkeypatch.setattr(cancellation, "CANCEL_POLL_SECONDS", 0.01)
    monkeypatch.setenv("COWORK_STREAM_BACKEND", "redis")
    return client


def _claim(run_id: str = "run-a", *, turn_id: int = 1, pod: str = "pod-a", question_id=None) -> TurnClaim:
    return TurnClaim(
        run_id=run_id, turn_id=turn_id, correlation_id=f"corr-{run_id}",
        question_id=question_id, pod=pod, started_at=1.0,
    )


async def _stored(redis, conversation_id: str = "c") -> TurnClaim | None:
    raw = await redis.get(claim_key(conversation_id))
    return None if raw is None else TurnClaim.model_validate_json(raw)


async def _listed(redis, conversation_id: str = "c") -> TurnClaim | None:
    raw = await redis.hget(CLAIMS_KEY, conversation_id)
    return None if raw is None else TurnClaim.model_validate_json(raw)


class _Unreachable:
    """A Redis client every command of which fails as a dropped connection does."""

    def __getattr__(self, name):
        def fail(*_args, **_kwargs):
            raise RedisConnectionError("Redis is unreachable")

        return fail


def _warnings(caplog, name: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == name and r.levelno == logging.WARNING]


# ── acquire ──────────────────────────────────────────────────────────


async def test_a_claim_is_taken_once_and_names_its_holder(redis):
    first = await turn_claim.acquire(claim=_claim("run-a"), conversation_id="c")
    second = await turn_claim.acquire(claim=_claim("run-b", pod="pod-b"), conversation_id="c")

    assert first == (AcquireOutcome.ACQUIRED, _claim("run-a"), None)
    assert second == (AcquireOutcome.HELD, _claim("run-a"), None)
    assert await _stored(redis) == _claim("run-a")
    assert await _listed(redis) == _claim("run-a")
    assert 60_000 < await redis.pttl(claim_key("c")) <= turn_claim.CLAIM_TTL_SECONDS * 1000


async def test_taking_a_claim_clears_an_ended_stream_at_its_number_and_leaves_an_open_one(redis):
    """An ended stream at the same number belongs to a turn that saved no
    message. Cleared in the claim's own transaction, another replica that
    finds the claim never reads that turn's terminal record as the new run's.
    A stream with no terminal record may still be written, so it stays."""
    ended = RedisStreamBuffer(conversation_id="c", turn_id=1)
    await ended.append("sse", {"sse": "earlier"})
    await ended.close("error")
    still_open = RedisStreamBuffer(conversation_id="d", turn_id=1)
    await still_open.append("sse", {"sse": "earlier"})

    await turn_claim.acquire(claim=_claim("run-a"), conversation_id="c")
    await turn_claim.acquire(claim=_claim("run-b"), conversation_id="d")

    assert await redis.exists(stream_key("c", 1)) == 0
    assert await redis.xlen(stream_key("d", 1)) == 1


async def test_a_start_the_turn_index_shows_is_stale_is_refused_and_deletes_nothing(redis):
    """The index names a run that saved its question at turn 2, whose stream
    has ended and whose claim is gone. A start at 2 read the history before
    that save: it takes no claim and leaves that turn's stream alone. A start
    at 3 read it after, and is admitted."""
    await turn_index.record_turn("c", turn_id=2, correlation_id="corr-a", org_id=None, user_id=None, run_id="run-a")
    assert await turn_index.stamp_question(conversation_id="c", run_id="run-a", question_id=None) is True
    ended = RedisStreamBuffer(conversation_id="c", turn_id=2)
    await ended.append("sse", COMPLETED)
    await ended.close("completed")

    stale = await turn_claim.acquire(claim=_claim("run-b", turn_id=2), conversation_id="c")

    assert stale == (AcquireOutcome.STALE, None, 2)
    assert await _stored(redis) is None and await _listed(redis) is None
    assert await redis.xlen(stream_key("c", 2)) == 2
    admitted = await turn_claim.acquire(claim=_claim("run-c", turn_id=3), conversation_id="c")
    assert admitted.outcome is AcquireOutcome.ACQUIRED


# ── renew ────────────────────────────────────────────────────────────


async def test_renewing_starts_the_ttl_again_and_writes_the_saved_question(redis):
    await turn_claim.acquire(claim=_claim("run-a"), conversation_id="c")
    await redis.pexpire(claim_key("c"), 1_000)
    question_id = uuid4()

    result = await turn_claim.renew(claim=_claim("run-a", question_id=question_id), conversation_id="c")

    assert result.outcome is RenewOutcome.RENEWED
    assert await redis.pttl(claim_key("c")) > 60_000
    assert (await _stored(redis)).question_id == question_id
    assert (await _listed(redis)).question_id == question_id


@pytest.mark.parametrize("last_record", [None, "sse", "final"], ids=["no-record", "streaming", "finishing"])
async def test_a_missing_claim_is_taken_back_while_its_stream_has_no_terminal(redis, last_record):
    """An eviction, a failover or a stall past the TTL drops a live run's
    key. The run takes it back on its next renewal."""
    await turn_claim.acquire(claim=_claim("run-a"), conversation_id="c")
    stream = RedisStreamBuffer(conversation_id="c", turn_id=1)
    if last_record == "sse":
        await stream.append("sse", {"sse": "partial"})
    elif last_record == "final":
        await stream.append("sse", COMPLETED)
    await redis.delete(claim_key("c"))

    result = await turn_claim.renew(claim=_claim("run-a"), conversation_id="c")

    assert result.outcome is RenewOutcome.REACQUIRED
    assert await _stored(redis) == _claim("run-a")
    assert await redis.pttl(claim_key("c")) > 60_000


async def test_a_missing_claim_is_not_taken_back_once_its_stream_has_ended(redis):
    """A terminal record the run did not write means another reader ended
    the turn while the key was gone."""
    await turn_claim.acquire(claim=_claim("run-a"), conversation_id="c")
    stream = RedisStreamBuffer(conversation_id="c", turn_id=1)
    await stream.append("sse", {"sse": "partial"})
    await stream.close("interrupted")
    await redis.delete(claim_key("c"))

    result = await turn_claim.renew(claim=_claim("run-a"), conversation_id="c")

    assert result == (RenewOutcome.LOST, None)
    assert await _stored(redis) is None


async def test_a_claim_another_run_holds_is_lost_and_left_alone(redis):
    await turn_claim.acquire(claim=_claim("run-b", pod="pod-b"), conversation_id="c")

    result = await turn_claim.renew(claim=_claim("run-a"), conversation_id="c")

    assert result == (RenewOutcome.LOST, _claim("run-b", pod="pod-b"))
    assert await _stored(redis) == _claim("run-b", pod="pod-b")


async def test_a_redis_error_on_renewal_is_an_error_not_a_lost_claim(monkeypatch, caplog, owned_logger):
    logged = owned_logger(CLAIM_LOGGER, level=logging.WARNING)
    monkeypatch.setattr(turn_claim, "get_redis", lambda: _Unreachable())

    result = await turn_claim.renew(claim=_claim("run-a"), conversation_id="c")

    assert result == (RenewOutcome.ERROR, None)
    records = _warnings(caplog, CLAIM_LOGGER)
    assert [(r.event, r.run_id, r.conversation_id) for r in records] == [
        ("turn_claim_renew_failed", "run-a", "c"),
    ]
    assert_log_fields(logged.output(), event="turn_claim_renew_failed", run_id="run-a", conversation_id="c")


async def test_a_renewal_redis_does_not_answer_in_time_is_an_error(redis, monkeypatch):
    monkeypatch.setattr(turn_claim, "REDIS_CALL_SECONDS", 0.05)

    async def hung(*_args, **_kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr(turn_claim, "_renew", hung)

    result = await turn_claim.renew(claim=_claim("run-a"), conversation_id="c")

    assert result.outcome is RenewOutcome.ERROR


# ── release ──────────────────────────────────────────────────────────


async def test_release_deletes_only_the_callers_own_claim(redis):
    await turn_claim.acquire(claim=_claim("run-b", pod="pod-b"), conversation_id="c")

    assert await turn_claim.release(conversation_id="c", run_id="run-a") is False
    assert await _stored(redis) == _claim("run-b", pod="pod-b")
    assert await _listed(redis) == _claim("run-b", pod="pod-b")

    assert await turn_claim.release(conversation_id="c", run_id="run-b") is True
    assert await redis.exists(claim_key("c"), CLAIMS_KEY) == 0


async def test_release_removes_a_runs_listing_left_after_its_key_lapsed(redis):
    await turn_claim.acquire(claim=_claim("run-a"), conversation_id="c")
    await redis.delete(claim_key("c"))

    assert await turn_claim.release(conversation_id="c", run_id="run-a") is False
    assert await redis.hexists(CLAIMS_KEY, "c") == 0


async def test_the_blocking_release_deletes_only_the_callers_own_claim(redis):
    await turn_claim.acquire(claim=_claim("run-b", pod="pod-b"), conversation_id="c")

    assert await asyncio.to_thread(turn_claim.release_sync, conversation_id="c", run_id="run-a") is False
    assert await _stored(redis) == _claim("run-b", pod="pod-b")
    assert await asyncio.to_thread(turn_claim.release_sync, conversation_id="c", run_id="run-b") is True
    assert await redis.exists(claim_key("c"), CLAIMS_KEY) == 0


# ── admission ────────────────────────────────────────────────────────


async def test_an_acquire_cut_short_after_redis_applied_it_leaves_no_claim(redis, monkeypatch):
    """A timeout can fire after Redis took the claim. Left there, the
    client's retry would be refused by its own refused attempt for a whole
    TTL."""
    monkeypatch.setattr(turn_claim, "REDIS_CALL_SECONDS", 0.05)
    real_acquire = turn_claim.acquire

    async def applied_then_hung(*, claim, conversation_id):
        await real_acquire(claim=claim, conversation_id=conversation_id)
        await asyncio.sleep(10)

    monkeypatch.setattr(turn_claim, "acquire", applied_then_hung)
    with pytest.raises(TurnClaimUnavailable):
        await turn_claim.claim_turn(claim=_claim("run-a"), conversation_id="c")

    assert await _stored(redis) is None
    assert await _listed(redis) is None
    monkeypatch.setattr(turn_claim, "acquire", real_acquire)
    retried = await turn_claim.claim_turn(claim=_claim("run-b"), conversation_id="c")
    assert retried.acquired


async def test_a_cancelled_admission_leaves_no_claim(redis, monkeypatch):
    real_acquire = turn_claim.acquire
    applied, release = asyncio.Event(), asyncio.Event()

    async def applied_then_held(*, claim, conversation_id):
        result = await real_acquire(claim=claim, conversation_id=conversation_id)
        applied.set()
        await release.wait()
        return result

    monkeypatch.setattr(turn_claim, "acquire", applied_then_held)
    admitting = asyncio.create_task(turn_claim.claim_turn(claim=_claim("run-a"), conversation_id="c"))
    await asyncio.wait_for(applied.wait(), timeout=2)
    admitting.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await admitting
    assert await _stored(redis) is None


@pytest.mark.parametrize(
    "last_record,held",
    [(None, True), ("sse", True), ("final", False), ("terminal", False)],
    ids=["no-record", "streaming", "finishing", "ended"],
)
async def test_the_pre_gate_counts_only_a_holder_still_streaming(redis, last_record, held):
    """A holder writing its last frames, or one that ended, is left to
    RunRegistry.start, which waits for it or clears its claim."""
    await turn_claim.acquire(claim=_claim("run-a"), conversation_id="c")
    stream = RedisStreamBuffer(conversation_id="c", turn_id=1)
    if last_record == "sse":
        await stream.append("sse", {"sse": "partial"})
    elif last_record == "final":
        await stream.append("sse", COMPLETED)
    elif last_record == "terminal":
        await stream.append("sse", COMPLETED)
        await stream.close("completed")

    holder = await turn_claim.live_holder(conversation_id="c")

    assert holder == (_claim("run-a") if held else None)


async def test_the_pre_gate_reports_an_unreadable_claim_as_unavailable(monkeypatch):
    monkeypatch.setattr(turn_claim, "get_redis", lambda: _Unreachable())

    with pytest.raises(TurnClaimUnavailable):
        await turn_claim.live_holder(conversation_id="c")


async def test_stream_state_reads_the_last_record(redis):
    stream = RedisStreamBuffer(conversation_id="c", turn_id=1)
    states = [await turn_claim.stream_state(conversation_id="c", turn_id=1)]
    await stream.append("sse", {"sse": "partial"})
    states.append(await turn_claim.stream_state(conversation_id="c", turn_id=1))
    await stream.append("sse", COMPLETED)
    states.append(await turn_claim.stream_state(conversation_id="c", turn_id=1))
    await stream.close("completed")
    states.append(await turn_claim.stream_state(conversation_id="c", turn_id=1))

    assert states == [StreamState.EMPTY, StreamState.LIVE, StreamState.FINISHING, StreamState.ENDED]


# ── keeping a claim ──────────────────────────────────────────────────


@pytest.fixture
def quick_renewals(monkeypatch):
    monkeypatch.setattr(turn_claim, "RENEW_INTERVAL_SECONDS", 0.01)


async def _until(predicate, *, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not await predicate():
        assert loop.time() < deadline, "condition never held"
        await asyncio.sleep(0.01)


async def test_a_kept_claim_carries_the_saved_question_and_stops_once_the_stream_ends(redis, quick_renewals):
    buffer = RedisStreamBuffer(conversation_id="c", turn_id=1)
    lifecycle = TurnLifecycle()
    await turn_claim.acquire(claim=_claim("run-a"), conversation_id="c")
    keeper = asyncio.create_task(turn_claim.keep_claim(
        claim=_claim("run-a"), conversation_id="c", buffer=buffer, lifecycle=lifecycle,
    ))
    lifecycle.question_id = uuid4()

    async def carries_question() -> bool:
        return (await _stored(redis)).question_id == lifecycle.question_id

    await _until(carries_question)
    await buffer.append("sse", COMPLETED)
    await asyncio.wait_for(keeper, timeout=2)

    assert lifecycle.claim_lost is False
    # Released by the registry after the terminal record, never by the keeper.
    assert await _stored(redis) is not None


async def test_a_claim_another_run_took_marks_the_turn_lost_not_deleted(
    redis, quick_renewals, caplog, owned_logger,
):
    logged = owned_logger(CLAIM_LOGGER, level=logging.WARNING)
    buffer = RedisStreamBuffer(conversation_id="c", turn_id=1)
    lifecycle = TurnLifecycle()
    await turn_claim.acquire(claim=_claim("run-a"), conversation_id="c")
    keeper = asyncio.create_task(turn_claim.keep_claim(
        claim=_claim("run-a"), conversation_id="c", buffer=buffer, lifecycle=lifecycle,
    ))
    await redis.set(claim_key("c"), _claim("run-b", pod="pod-b").model_dump_json())

    await asyncio.wait_for(keeper, timeout=2)

    assert lifecycle.claim_lost is True
    assert lifecycle.discarded is False
    records = _warnings(caplog, CLAIM_LOGGER)
    fields = {
        "event": "turn_claim_lost", "conversation_id": "c", "run_id": "run-a",
        "holder_run_id": "run-b", "holder_pod": "pod-b",
    }
    assert [{name: getattr(r, name) for name in fields} for r in records] == [fields]
    assert_log_fields(logged.output(), **fields)
    assert await _stored(redis) == _claim("run-b", pod="pod-b")


async def test_a_redis_outage_while_keeping_a_claim_does_not_lose_it(redis, quick_renewals, monkeypatch):
    buffer = RedisStreamBuffer(conversation_id="c", turn_id=1)
    lifecycle = TurnLifecycle()
    await turn_claim.acquire(claim=_claim("run-a"), conversation_id="c")
    outcomes: list[RenewOutcome] = []
    real_renew = turn_claim.renew
    down = True

    async def recorded_renew(*, claim, conversation_id):
        if down:
            monkeypatch.setattr(turn_claim, "get_redis", lambda: _Unreachable())
        else:
            monkeypatch.setattr(turn_claim, "get_redis", lambda: redis)
        result = await real_renew(claim=claim, conversation_id=conversation_id)
        outcomes.append(result.outcome)
        return result

    monkeypatch.setattr(turn_claim, "renew", recorded_renew)
    keeper = asyncio.create_task(turn_claim.keep_claim(
        claim=_claim("run-a"), conversation_id="c", buffer=buffer, lifecycle=lifecycle,
    ))
    await redis.delete(claim_key("c"))   # lost to the same failover

    async def three_errors() -> bool:
        return outcomes.count(RenewOutcome.ERROR) >= 3

    await _until(three_errors)
    down = False

    async def taken_back() -> bool:
        return RenewOutcome.REACQUIRED in outcomes

    await _until(taken_back)
    await buffer.close("completed")
    await asyncio.wait_for(keeper, timeout=2)

    assert lifecycle.claim_lost is False
    assert not keeper.cancelled()
    assert await _stored(redis) == _claim("run-a")


async def test_a_renewal_that_takes_back_a_deleted_turns_claim_releases_it(redis, quick_renewals, monkeypatch):
    """A delete landed while a renewal ran: it removed the turn's stream and
    then its claim, so the renewal found both gone and took the claim back.
    Kept, that claim would refuse sends into the truncated conversation until
    the producer finished unwinding."""
    buffer = RedisStreamBuffer(conversation_id="c", turn_id=1)
    await buffer.append("sse", {"sse": "partial"})
    lifecycle = TurnLifecycle()
    await turn_claim.acquire(claim=_claim("run-a"), conversation_id="c")
    real_renew = turn_claim.renew
    outcomes: list[RenewOutcome] = []

    async def renew_during_a_delete(*, claim, conversation_id):
        lifecycle.discarded = True
        await redis.delete(stream_key("c", 1))
        await turn_claim.release(conversation_id="c", run_id=claim.run_id)
        result = await real_renew(claim=claim, conversation_id=conversation_id)
        outcomes.append(result.outcome)
        return result

    monkeypatch.setattr(turn_claim, "renew", renew_during_a_delete)

    await asyncio.wait_for(turn_claim.keep_claim(
        claim=_claim("run-a"), conversation_id="c", buffer=buffer, lifecycle=lifecycle,
    ), timeout=2)

    assert outcomes == [RenewOutcome.REACQUIRED]
    assert await redis.exists(claim_key("c"), CLAIMS_KEY) == 0
    assert lifecycle.claim_lost is False


# ── the registry's release ───────────────────────────────────────────


def _waiting_producer(started: asyncio.Event, release: asyncio.Event):
    """A producer that writes a frame, then waits for ``release`` before it
    answers: a turn blocked on an unanswered question, say."""

    async def produce(buffer):
        await buffer.append("sse", {"sse": "partial"})
        started.set()
        await release.wait()
        await buffer.append("sse", COMPLETED)
        await buffer.close("completed")

    return produce


async def test_a_finished_turn_releases_its_claim_and_notes_its_saved_question(redis):
    runs = RunRegistry()
    question_id = uuid4()
    lifecycle = TurnLifecycle(saved_question=True, question_id=question_id)

    async def answer(buffer):
        await buffer.append("sse", COMPLETED)
        await buffer.close("completed")

    handle = await runs.start(
        conversation_id="c", turn_id=1, open_buffer=lambda: backend_mod.new_buffer("c", 1),
        produce=answer, lifecycle=lifecycle,
    )
    await asyncio.wait_for(handle.task, timeout=2)

    assert await redis.exists(claim_key("c"), CLAIMS_KEY) == 0
    entry = await turn_index.get_turn("c")
    assert entry["run_id"] == handle.run_id
    assert (entry["saved_question"], entry["question_id"]) == ("1", str(question_id))


@pytest.mark.parametrize("seal_fails", [False, True])
async def test_a_turn_ended_at_shutdown_with_its_stream_open_is_released_only_after_its_seal(
    redis, monkeypatch, seal_fails,
):
    """A producer cancelled before it wrote a terminal record leaves its
    stream open, so its own release keeps the claim. The shutdown seals the
    stream and then releases it; a seal that fails leaves the claim to its
    TTL, so no successor writes into a stream with no end."""
    runs = RunRegistry()
    started = asyncio.Event()

    async def produce(buffer):
        await buffer.append("sse", {"sse": "partial"})
        started.set()
        await asyncio.Event().wait()

    handle = await runs.start(
        conversation_id="c", turn_id=1, open_buffer=lambda: backend_mod.new_buffer("c", 1),
        produce=produce,
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    if seal_fails:
        async def failing_append(*_args, **_kwargs):
            raise RedisConnectionError("Redis is unreachable")

        monkeypatch.setattr(handle.buffer, "append", failing_append)

    await runs.shutdown(timeout_seconds=1)

    if seal_fails:
        assert await _stored(redis) == handle.claim
        return
    entries = await redis.xrange(stream_key("c", 1))
    assert entries[-1][1]["type"] == "Interrupted"
    assert await _stored(redis) is None


async def test_a_deleted_conversations_turn_releases_its_claim(redis, monkeypatch):
    """The delete releases the claim itself, before the producer has
    unwound: a cancelled producer can wait on a save for seconds, and its
    task's own release would come only then."""
    registry_mod = importlib.import_module("cowork.streaming.registry")
    runs = RunRegistry()
    monkeypatch.setattr(registry_mod, "registry", runs)
    started, unwound = asyncio.Event(), asyncio.Event()

    async def slow_to_unwind(buffer):
        await buffer.append("sse", {"sse": "partial"})
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await unwound.wait()
            raise

    handle = await runs.start(
        conversation_id="c", turn_id=1, open_buffer=lambda: backend_mod.new_buffer("c", 1),
        produce=slow_to_unwind,
    )
    await asyncio.wait_for(started.wait(), timeout=2)

    await asyncio.to_thread(registry_mod.discard_conversation, "c")

    assert handle.lifecycle.discarded is True
    assert not handle.task.done()
    assert await redis.exists(claim_key("c"), CLAIMS_KEY) == 0
    unwound.set()
    await asyncio.gather(handle.task, return_exceptions=True)


async def _saved_turn(runs: RunRegistry, *, conversation_id: str, turn_id: int):
    """A turn that saved its question and has ended, as the handler runs one."""
    from cowork.streaming.registry import mark_question_saved

    lifecycle = TurnLifecycle(question_id=uuid4())

    async def answer(buffer):
        await mark_question_saved(lifecycle=lifecycle, conversation_id=conversation_id)
        await buffer.append("sse", COMPLETED)
        await buffer.close("completed")

    handle = await runs.start(
        conversation_id=conversation_id, turn_id=turn_id,
        open_buffer=lambda: backend_mod.new_buffer(conversation_id, turn_id),
        produce=answer, lifecycle=lifecycle,
    )
    await asyncio.wait_for(handle.task, timeout=2)
    assert (await turn_index.get_turn(conversation_id))["saved_question"] == "1"
    return handle


async def test_a_delete_whose_buffer_removal_fails_still_admits_the_next_send(
    redis, monkeypatch, caplog, owned_logger,
):
    """Turn 3 saved its question, so its index entry refuses a start at 3.
    The user deletes it: the history is back to three messages, so the next
    send is turn 3 again. A Redis error while the buffers were removed used
    to skip the index forget, and every replica then refused that send with
    409 until the entry expired."""
    registry_mod = importlib.import_module("cowork.streaming.registry")
    logged = owned_logger(registry_mod.logger.name, level=logging.WARNING)
    replica_a, replica_b = RunRegistry(), RunRegistry()
    monkeypatch.setattr(registry_mod, "registry", replica_a)
    await _saved_turn(replica_a, conversation_id="d", turn_id=3)

    def failing_removal(_conversation_id):
        raise RedisConnectionError("Redis is unreachable")

    monkeypatch.setattr(backend_mod, "_remove_redis_buffers", failing_removal)

    await asyncio.to_thread(registry_mod.discard_conversation, "d")

    assert await turn_index.get_turn("d") is None
    records = [r for r in _warnings(caplog, registry_mod.logger.name) if r.event == "turn_buffers_remove_failed"]
    assert [r.conversation_id for r in records] == ["d"]
    assert_log_fields(logged.output(), event="turn_buffers_remove_failed", conversation_id="d")
    for runs in (replica_a, replica_b):
        admitted = await runs.start(
            conversation_id="d", turn_id=3, open_buffer=lambda: backend_mod.new_buffer("d", 3),
            produce=_waiting_producer(asyncio.Event(), asyncio.Event()),
        )
        admitted.task.cancel()
        await asyncio.gather(admitted.task, return_exceptions=True)
        await turn_claim.release(conversation_id="d", run_id=admitted.run_id)


async def test_a_delete_whose_index_forget_fails_logs_it_and_still_releases(
    redis, monkeypatch, caplog, owned_logger,
):
    """A forget that cannot reach Redis leaves a stamp that refuses sends at
    or below the deleted turn's number until the entry expires, so it is
    logged with an event an operator can find. The buffers and the claim are
    still removed."""
    registry_mod = importlib.import_module("cowork.streaming.registry")
    logged = owned_logger(turn_index.logger.name, level=logging.WARNING)
    runs = RunRegistry()
    monkeypatch.setattr(registry_mod, "registry", runs)
    started, release = asyncio.Event(), asyncio.Event()
    handle = await runs.start(
        conversation_id="f", turn_id=1, open_buffer=lambda: backend_mod.new_buffer("f", 1),
        produce=_waiting_producer(started, release),
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    monkeypatch.setattr(turn_index, "get_sync_redis", lambda: _Unreachable())

    await asyncio.to_thread(registry_mod.discard_conversation, "f")
    await asyncio.gather(handle.task, return_exceptions=True)

    records = [r for r in _warnings(caplog, turn_index.logger.name) if r.event == "turn_index_forget_failed"]
    assert [r.conversation_id for r in records] == ["f"]
    assert_log_fields(logged.output(), event="turn_index_forget_failed", conversation_id="f")
    assert await redis.exists(stream_key("f", 1), claim_key("f"), CLAIMS_KEY) == 0


# ── the boot pass ────────────────────────────────────────────────────


async def test_the_boot_pass_releases_only_this_pods_claims(redis):
    await turn_claim.acquire(claim=_claim("run-a", pod="pod-a"), conversation_id="c1")
    await turn_claim.acquire(claim=_claim("run-b", pod="pod-a"), conversation_id="c2")
    await turn_claim.acquire(claim=_claim("run-c", pod="pod-b"), conversation_id="c3")

    released = await turn_claim.release_pod_claims(pod="pod-a")

    assert released == BootPassResult(released=2, pruned=0)
    assert await _stored(redis, "c1") is None and await _stored(redis, "c2") is None
    assert await _stored(redis, "c3") == _claim("run-c", pod="pod-b")
    assert await redis.hkeys(CLAIMS_KEY) == ["c3"]


async def test_the_boot_pass_removes_the_listings_of_claims_that_lapsed_on_gone_pods(redis):
    """A pod killed mid-turn leaves its claims' listings behind once the
    keys lapse, under a pod name no later process carries. Only a release
    removed a listing, so they piled up and every boot read them all."""
    await turn_claim.acquire(claim=_claim("run-dead", pod="cowork-server-old"), conversation_id="lapsed")
    await redis.delete(claim_key("lapsed"))
    await turn_claim.acquire(claim=_claim("run-gone", pod="cowork-server-old"), conversation_id="taken")
    await redis.set(claim_key("taken"), _claim("run-new", pod="cowork-server-xyz").model_dump_json())
    await turn_claim.acquire(claim=_claim("run-live", pod="cowork-server-xyz"), conversation_id="live")

    result = await turn_claim.release_pod_claims(pod="cowork-server-new")

    assert result == BootPassResult(released=0, pruned=2)
    assert await redis.hkeys(CLAIMS_KEY) == ["live"]
    assert await _stored(redis, "taken") == _claim("run-new", pod="cowork-server-xyz")
    assert await _stored(redis, "live") == _claim("run-live", pod="cowork-server-xyz")
    # A claim re-taken after the pass read the keys (a stalled run's renewal
    # takes back its lapsed key) keeps its listing.
    assert await turn_claim._unlist_lapsed(conversation_id="live", run_id="run-live") is False
    assert await _listed(redis, "live") == _claim("run-live", pod="cowork-server-xyz")


async def test_the_server_releases_the_claims_its_pod_left_on_boot(redis, monkeypatch, caplog, owned_logger):
    from cowork import server

    logged = owned_logger(server.logger.name, level=logging.WARNING)
    monkeypatch.setenv("POD_NAME", "cowork-server-abc")
    await turn_claim.acquire(claim=_claim("run-a", pod="cowork-server-abc"), conversation_id="c1")
    await turn_claim.acquire(claim=_claim("run-b", pod="cowork-server-xyz"), conversation_id="c2")
    await turn_claim.acquire(claim=_claim("run-c", pod="cowork-server-old"), conversation_id="c3")
    await redis.delete(claim_key("c3"))

    await server._release_own_turn_claims()

    assert await _stored(redis, "c1") is None
    assert await _stored(redis, "c2") is not None
    assert await redis.hkeys(CLAIMS_KEY) == ["c2"]
    records = _warnings(caplog, server.logger.name)
    assert [(r.event, r.pod, r.count) for r in records] == [
        ("turn_claims_released_at_boot", "cowork-server-abc", 1),
        ("turn_claim_listings_pruned_at_boot", "cowork-server-abc", 1),
    ]
    assert_log_fields(logged.output(), event="turn_claims_released_at_boot", pod="cowork-server-abc", count=1)
    assert_log_fields(logged.output(), event="turn_claim_listings_pruned_at_boot", pod="cowork-server-abc", count=1)


def test_the_pod_name_falls_back_to_the_host_name(monkeypatch):
    monkeypatch.delenv("POD_NAME", raising=False)
    monkeypatch.setattr(turn_claim.socket, "gethostname", lambda: "cowork-server-host")
    assert turn_claim.pod_name() == "cowork-server-host"
    monkeypatch.setenv("POD_NAME", "cowork-server-pod")
    assert turn_claim.pod_name() == "cowork-server-pod"


def test_a_hosted_pod_runs_one_server_process():
    """The boot pass releases every claim carrying its pod's name, which is
    right only while a pod runs one server process: a second worker in the
    same pod would release the first one's live claims. Uvicorn takes the
    worker count from --workers in the command; tests/test_chart_values.py
    already fails a values file that sets WEB_CONCURRENCY or UVICORN_*."""
    command = hosted_command()
    assert "uvicorn" in command
    workers = [arg for arg in command if arg.startswith("--workers")]
    if workers:
        value = workers[0].split("=", 1)[1] if "=" in workers[0] else command[command.index(workers[0]) + 1]
        assert int(value) == 1, f"the hosted command runs {value} workers: {command}"


def test_the_claim_outlasts_the_stall_the_liveness_probe_forgives():
    """A stalled event loop renews nothing. The kubelet restarts a pod whose
    loop stalls longer than the liveness window, so a claim that outlasts it
    lapses only for a process that is gone or about to be.

    The window is failureThreshold probes apart, plus the timeoutSeconds the
    last of them waits for an answer: a stall that ends inside that wait is
    forgiven. When the stall starts, the claim has run down by up to one
    renewal interval, and the last renewal may have taken its whole Redis
    bound to land."""
    deployment = yaml.safe_load(CHART_VALUES.read_text())["deployment"]
    probe = deployment["livenessProbe"]
    forgiven = probe["failureThreshold"] * probe["periodSeconds"] + probe["timeoutSeconds"]
    run_down = turn_claim.RENEW_INTERVAL_SECONDS + turn_claim.REDIS_CALL_SECONDS

    assert turn_claim.CLAIM_TTL_SECONDS > forgiven + run_down
    # Two renewals, each waiting out its bound, still land inside one TTL.
    assert 2 * (turn_claim.RENEW_INTERVAL_SECONDS + turn_claim.REDIS_CALL_SECONDS) < turn_claim.CLAIM_TTL_SECONDS


# ── the file backend ─────────────────────────────────────────────────


async def test_the_file_backend_never_reaches_redis(monkeypatch, tmp_path):
    """Desktop and the self-hosted container run the file backend with no
    Redis. Admission, the turn, its release, a delete and the boot pass
    reach none of the claim's code there."""
    from cowork import server
    from cowork.api.v1.endpoints import responses as responses_ep
    from cowork.db.scoped import TenantScope
    registry_mod = importlib.import_module("cowork.streaming.registry")
    calls: list[str] = []

    def no_redis(name):
        def called(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(f"{name} reached Redis on the file backend")

        return called

    for module in (turn_claim, turn_index, buffer_mod, cancellation, redis_client, responses_ep):
        monkeypatch.setattr(module, "get_redis", no_redis(f"{module.__name__}.get_redis"))
    monkeypatch.setattr(turn_claim, "get_sync_redis", no_redis("turn_claim.get_sync_redis"))
    # discard_conversation's turn index cleanup reaches Redis on every backend
    # today and logs its own failure; it is not the claim's, so not counted.
    monkeypatch.setattr(turn_index, "get_sync_redis", lambda: fakeredis.FakeRedis(decode_responses=True))
    monkeypatch.setenv("COWORK_STREAM_BACKEND", "file")
    runs = RunRegistry()
    monkeypatch.setattr(registry_mod, "registry", runs)
    monkeypatch.setattr(responses_ep, "registry", runs)
    conversation_id = str(uuid4())

    await responses_ep._refuse_while_answering(
        conversation=conversation_id, scope=TenantScope(org_mode=False),
    )
    started, release = asyncio.Event(), asyncio.Event()
    lifecycle = TurnLifecycle(question_id=uuid4())
    waiting = _waiting_producer(started, release)

    async def saves_then_waits(buffer):
        await registry_mod.mark_question_saved(lifecycle=lifecycle, conversation_id=conversation_id)
        await waiting(buffer)

    handle = await runs.start(
        conversation_id=conversation_id, turn_id=0,
        open_buffer=opens(FileStreamBuffer(tmp_path / "turn.jsonl")),
        produce=saves_then_waits, lifecycle=lifecycle,
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    assert lifecycle.saved_question is True
    release.set()
    await asyncio.wait_for(handle.task, timeout=2)
    second = await runs.start(
        conversation_id=conversation_id, turn_id=2,
        open_buffer=opens(FileStreamBuffer(tmp_path / "turn2.jsonl")),
        produce=_waiting_producer(asyncio.Event(), asyncio.Event()),
    )
    await runs.shutdown(timeout_seconds=1)
    await asyncio.to_thread(registry_mod.discard_conversation, conversation_id)
    await server._release_own_turn_claims()

    assert handle.claim is None and second.claim is None
    assert calls == []


async def test_a_relays_own_reader_neither_ends_its_quiet_turn_nor_admits_a_second_question(
    redis, monkeypatch,
):
    """The POST stream reads the producer's own buffer object. A turn that
    goes quiet past the reader's idle cut (an ask_user question left
    unanswered) is still running and holds its claim, so the reader keeps
    waiting and the same replica still refuses a second question."""
    monkeypatch.setattr(buffer_mod, "REDIS_TAIL_IDLE_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr(RedisStreamBuffer, "_BLOCK_MS", 20)
    runs = RunRegistry()
    started, release = asyncio.Event(), asyncio.Event()
    handle = await runs.start(
        conversation_id="c", turn_id=1, open_buffer=lambda: backend_mod.new_buffer("c", 1),
        produce=_waiting_producer(started, release),
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    reading = asyncio.create_task(_drain(handle.buffer.tail(from_seq=0)))
    await asyncio.sleep(0.3)   # many idle windows

    assert handle.is_answering
    with pytest.raises(TurnInProgress):
        await runs.start(
            conversation_id="c", turn_id=1, open_buffer=lambda: backend_mod.new_buffer("c", 1),
            produce=_waiting_producer(asyncio.Event(), asyncio.Event()),
        )
    release.set()
    records = await asyncio.wait_for(reading, timeout=2)
    await asyncio.wait_for(handle.task, timeout=2)
    assert [r.type for r in records] == ["sse", "sse", "Done"]


async def _drain(records):
    return [record async for record in records]
