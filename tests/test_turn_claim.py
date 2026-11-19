"""The cross-replica turn claim: one run answers a conversation at a time,
whichever replica a question reaches.

Unit tests on fakeredis. tests/test_responses_multi_replica.py covers
admission through two registries and the endpoints, and
tests/integration/test_two_replicas.py the same against a real Redis.
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
from functools import partial
from uuid import uuid4

import fakeredis
import fakeredis.aioredis
import pytest
import yaml
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import WatchError

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
from cowork.turnqueue.redis_client import cancel_flag_key
from tests._logging_assertions import assert_log_fields

from _fakes import UnreachableRedis, opens
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
    for module in (turn_claim, turn_index, backend_mod, cancellation):
        monkeypatch.setattr(module, "get_sync_redis", lambda: sync_client)
    monkeypatch.setattr(cancellation, "CANCEL_POLL_SECONDS", 0.01)
    monkeypatch.setenv("COWORK_STREAM_BACKEND", "redis")
    return client


def _claim(run_id: str = "run-a", *, turn_id: int = 1, pod: str = "pod-a") -> TurnClaim:
    return TurnClaim(run_id=run_id, turn_id=turn_id, correlation_id=f"corr-{run_id}", pod=pod, started_at=1.0)


async def _stored(redis, conversation_id: str = "c") -> TurnClaim | None:
    raw = await redis.get(claim_key(conversation_id))
    return None if raw is None else TurnClaim.model_validate_json(raw)


async def _listed(redis, conversation_id: str = "c") -> TurnClaim | None:
    raw = await redis.hget(CLAIMS_KEY, conversation_id)
    return None if raw is None else TurnClaim.model_validate_json(raw)


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
    assert await turn_index.stamp_question(conversation_id="c", run_id="run-a", released=True) is True
    ended = RedisStreamBuffer(conversation_id="c", turn_id=2)
    await ended.append("sse", COMPLETED)
    await ended.close("completed")

    stale = await turn_claim.acquire(claim=_claim("run-b", turn_id=2), conversation_id="c")

    assert stale == (AcquireOutcome.STALE, None, 2)
    assert await _stored(redis) is None and await _listed(redis) is None
    assert await redis.xlen(stream_key("c", 2)) == 2
    admitted = await turn_claim.acquire(claim=_claim("run-c", turn_id=3), conversation_id="c")
    assert admitted.outcome is AcquireOutcome.ACQUIRED


@pytest.fixture
def yielding_reads(monkeypatch):
    """Each fakeredis reply suspends the event loop once, as a reply read off
    a real socket does. Unpatched, fakeredis answers without suspending, so a
    gather of starts runs each one's transaction in one step and no two can
    race."""
    real_read = fakeredis.aioredis.FakeBaseAsyncConnection.read_response

    async def read_response(self, **kwargs):
        await asyncio.sleep(0)
        return await real_read(self, **kwargs)

    monkeypatch.setattr(fakeredis.aioredis.FakeBaseAsyncConnection, "read_response", read_response)


async def test_starts_racing_on_one_conversation_admit_exactly_one(redis, yielding_reads):
    """Two replicas that get the same send at the same instant both find no
    holder. The claim's compare-and-set admits one of them; a plain read then
    write would admit both, and both would write the turn's stream from seq
    0."""
    replicas = [RunRegistry() for _ in range(4)]
    release = asyncio.Event()

    results = await asyncio.gather(*(
        runs.start(
            conversation_id="c", turn_id=1, open_buffer=lambda: backend_mod.new_buffer("c", 1),
            produce=_waiting_producer(asyncio.Event(), release),
        )
        for runs in replicas
    ), return_exceptions=True)

    admitted = [result for result in results if not isinstance(result, BaseException)]
    assert len(admitted) == 1, results
    assert all(isinstance(result, TurnInProgress) for result in results if result is not admitted[0]), results
    assert await _stored(redis) == admitted[0].claim
    release.set()
    await asyncio.wait_for(admitted[0].task, timeout=2)


# ── renew ────────────────────────────────────────────────────────────


async def test_renewing_starts_the_ttl_again(redis):
    await turn_claim.acquire(claim=_claim("run-a"), conversation_id="c")
    await redis.pexpire(claim_key("c"), 1_000)

    result = await turn_claim.renew(claim=_claim("run-a"), conversation_id="c")

    assert result.outcome is RenewOutcome.RENEWED
    assert await redis.pttl(claim_key("c")) > 60_000
    assert await _stored(redis) == _claim("run-a") == await _listed(redis)


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
    monkeypatch.setattr(turn_claim, "get_redis", lambda: UnreachableRedis())

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


async def test_a_renewal_whose_stream_a_reader_ends_meanwhile_is_lost(redis, monkeypatch):
    """The run's key is gone, so its renewal also watches the run's stream
    before it takes the key back. A reader that ends the stream (a dead
    relay's idle cut) between that read and the EXEC aborts it, and the
    retry reads the terminal record: the claim is lost, never taken back
    over a stream that has ended."""
    stream = RedisStreamBuffer(conversation_id="c", turn_id=1)
    await stream.append("sse", {"sse": "partial"})

    async def a_reader_ends_the_stream():
        await redis.xadd(stream_key("c", 1), {"seq": "1", "ts": "", "type": "Interrupted", "data": "{}"})

    raced = _race_before_exec(monkeypatch, redis, a_reader_ends_the_stream, times=1)

    result = await turn_claim.renew(claim=_claim("run-a"), conversation_id="c")

    assert raced == [1]
    assert result == (RenewOutcome.LOST, None)
    assert await _stored(redis) is None


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


async def test_a_release_leaves_a_claim_value_it_cannot_read(redis):
    """Nothing but turn_claim writes the key, so a value no TurnClaim parses
    is not the caller's. Release leaves it and answers False, where a raise
    would have release_quietly try it again for a whole TTL."""
    await redis.set(claim_key("c"), "not a claim", px=60_000)
    await redis.hset(CLAIMS_KEY, "c", "not a claim")

    assert await turn_claim.release(conversation_id="c", run_id="run-a") is False
    assert await redis.get(claim_key("c")) == "not a claim"
    assert await redis.hget(CLAIMS_KEY, "c") == "not a claim"


# ── compare-and-set under contention ─────────────────────────────────


def _race_before_exec(monkeypatch, client, write, *, times: int) -> list[int]:
    """Run ``write`` just before each of the next ``times`` EXECs on
    ``client``: another connection's write to a key the transaction watches,
    which makes Redis abort that EXEC. Returns one entry per race run."""
    pipeline = client.pipeline
    raced: list[int] = []

    def racing_pipeline(*args, **kwargs):
        pipe = pipeline(*args, **kwargs)
        execute = pipe.execute
        if inspect.iscoroutinefunction(execute):
            async def execute_after_the_race(*exec_args, **exec_kwargs):
                if len(raced) < times:
                    raced.append(1)
                    await write()
                return await execute(*exec_args, **exec_kwargs)
        else:
            def execute_after_the_race(*exec_args, **exec_kwargs):
                if len(raced) < times:
                    raced.append(1)
                    write()
                return execute(*exec_args, **exec_kwargs)
        pipe.execute = execute_after_the_race
        return pipe

    monkeypatch.setattr(client, "pipeline", racing_pipeline)
    return raced


async def _acquired(redis) -> None:
    await turn_claim.acquire(claim=_claim("run-a"), conversation_id="c")


async def _lapsed(redis) -> None:
    await _acquired(redis)
    await redis.delete(claim_key("c"))


async def _recorded(redis) -> None:
    await turn_index.record_turn("c", turn_id=1, correlation_id="corr-run-a", org_id=None, user_id=None, run_id="run-a")


async def _nothing(redis) -> None:
    return None


async def _rewrite_the_claim(redis, sync_redis) -> None:
    """The same value again: nothing changes, but it is a write."""
    await redis.set(claim_key("c"), await redis.get(claim_key("c")), keepttl=True)


def _rewrite_the_claim_blocking(redis, sync_redis) -> None:
    sync_redis.set(claim_key("c"), sync_redis.get(claim_key("c")), keepttl=True)


async def _take_and_drop_the_claim(redis, sync_redis) -> None:
    """Another run takes the claim and lets it go, leaving it as absent as
    it was."""
    await redis.set(claim_key("c"), _claim("run-b", pod="pod-b").model_dump_json())
    await redis.delete(claim_key("c"))


async def _rewrite_the_index(redis, sync_redis) -> None:
    await redis.hset(turn_index.turn_key("c"), "org_id", "")


# Each compare-and-set: what is there first, which client the race writes
# through, the race's write, the call, its answer after one race, and its
# answer (or the exception it raises) once the race wins every attempt.
_CAS_RACES = {
    "acquire": (
        _nothing, "async", _take_and_drop_the_claim,
        lambda: turn_claim.claim_turn(claim=_claim("run-a"), conversation_id="c"),
        lambda result: result.outcome is AcquireOutcome.ACQUIRED, TurnClaimUnavailable,
    ),
    "renew": (
        _acquired, "async", _rewrite_the_claim,
        lambda: turn_claim.renew(claim=_claim("run-a"), conversation_id="c"),
        lambda result: result.outcome is RenewOutcome.RENEWED,
        lambda result: result.outcome is RenewOutcome.ERROR,
    ),
    "release": (
        _acquired, "async", _rewrite_the_claim,
        lambda: turn_claim.release(conversation_id="c", run_id="run-a"),
        lambda result: result is True, WatchError,
    ),
    "release_sync": (
        _acquired, "sync", _rewrite_the_claim_blocking,
        lambda: asyncio.to_thread(turn_claim.release_sync, conversation_id="c", run_id="run-a"),
        lambda result: result is True, WatchError,
    ),
    "unlist_lapsed": (
        _lapsed, "async", _take_and_drop_the_claim,
        lambda: turn_claim._unlist_lapsed(conversation_id="c", run_id="run-a"),
        lambda result: result is True, lambda result: result is False,
    ),
    "record_channel_turn": (
        _nothing, "async", _take_and_drop_the_claim,
        lambda: turn_index.record_channel_turn(
            conversation_id="c", turn_id=1, correlation_id="corr-channel", org_id=None, user_id=None,
        ),
        lambda result: result is True, WatchError,
    ),
    "stamp_question": (
        _recorded, "async", _rewrite_the_index,
        lambda: turn_index.stamp_question(conversation_id="c", run_id="run-a", released=False),
        lambda result: result is True, WatchError,
    ),
}


def _raced(monkeypatch, redis, *, client: str, write, times: int) -> list[int]:
    sync_redis = turn_claim.get_sync_redis()
    raced_client = redis if client == "async" else sync_redis
    return _race_before_exec(monkeypatch, raced_client, partial(write, redis, sync_redis), times=times)


@pytest.mark.parametrize("race", list(_CAS_RACES))
async def test_a_compare_and_set_another_client_raced_reads_again_and_lands(redis, monkeypatch, race):
    """Another client's write to a watched key between WATCH and EXEC aborts
    the EXEC: a relay appending to a watched stream, or another replica's
    start. The call reads again and tries once more rather than failing."""
    setup, client, write, call, answered, _exhausted = _CAS_RACES[race]
    await setup(redis)
    raced = _raced(monkeypatch, redis, client=client, write=write, times=1)

    result = await call()

    assert raced == [1]
    assert answered(result), result


@pytest.mark.parametrize("race", list(_CAS_RACES))
async def test_a_compare_and_set_raced_on_every_attempt_gives_up(redis, monkeypatch, race):
    """A key that keeps changing is not waited on forever: after three
    aborted attempts the call gives up. Admission answers that 503, a
    renewal reads it as an error to try at the next interval, and the boot
    pass leaves the listing to the next pass."""
    setup, client, write, call, _answered, exhausted = _CAS_RACES[race]
    await setup(redis)
    raced = _raced(monkeypatch, redis, client=client, write=write, times=3)

    if isinstance(exhausted, type):
        with pytest.raises(exhausted):
            await call()
    else:
        result = await call()
        assert exhausted(result), result
    assert len(raced) == 3


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


async def test_an_admission_whose_undo_redis_does_not_answer_is_undone_once_redis_answers(redis, monkeypatch):
    """Redis applied the acquire and then stalled, so the acquire timed out,
    the client got 503, and the undo timed out in the same stall. The claim
    used to stay for its whole TTL: the client's retry, on any replica, met
    its own refused attempt's claim over an empty stream and was refused 409
    while nothing ran."""
    monkeypatch.setattr(turn_claim, "REDIS_CALL_SECONDS", 0.05)
    monkeypatch.setattr(turn_claim, "RELEASE_RETRY_SECONDS", 0.01)
    real_acquire, real_release = turn_claim.acquire, turn_claim.release
    releases = 0

    async def applied_then_hung(*, claim, conversation_id):
        await real_acquire(claim=claim, conversation_id=conversation_id)
        await asyncio.sleep(10)

    async def release_after_three_stalls(**kwargs):
        # The undo's own release and its first two retries meet the stall.
        nonlocal releases
        releases += 1
        if releases <= 3:
            await asyncio.sleep(10)
        return await real_release(**kwargs)

    async def answered(buffer):
        await buffer.append("sse", COMPLETED)
        await buffer.close("completed")

    monkeypatch.setattr(turn_claim, "acquire", applied_then_hung)
    monkeypatch.setattr(turn_claim, "release", release_after_three_stalls)
    replica_a, replica_b = RunRegistry(), RunRegistry()
    with pytest.raises(TurnClaimUnavailable):
        await replica_a.start(
            conversation_id="c", turn_id=4, open_buffer=lambda: backend_mod.new_buffer("c", 4), produce=answered,
        )
    assert (await _stored(redis)).turn_id == 4

    # Redis answers again, after two more stalled releases.
    monkeypatch.setattr(turn_claim, "acquire", real_acquire)

    async def undone() -> bool:
        return await redis.exists(claim_key("c"), CLAIMS_KEY) == 0

    await _until(undone)
    assert releases == 4
    assert await turn_claim.live_holder(conversation_id="c") is None
    for runs in (replica_b, replica_a):
        handle = await runs.start(
            conversation_id="c", turn_id=4, open_buffer=lambda: backend_mod.new_buffer("c", 4), produce=answered,
        )
        await asyncio.wait_for(handle.task, timeout=2)


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
    monkeypatch.setattr(turn_claim, "get_redis", lambda: UnreachableRedis())

    with pytest.raises(TurnClaimUnavailable):
        await turn_claim.live_holder(conversation_id="c")


async def test_a_stale_start_is_refused_before_it_waits_for_another_runs_final_frame(redis):
    """Another replica's run at turn 2 saved its question and is writing its
    final client frame. A start at 2 read the history before that save. The
    turn index check before the claim refuses it at once; acquire alone
    would refuse it only after admission waited up to
    ENDING_HOLDER_WAIT_SECONDS for that run to release its claim."""
    await turn_claim.acquire(claim=_claim("run-b", turn_id=2, pod="pod-b"), conversation_id="c")
    await turn_index.record_turn("c", turn_id=2, correlation_id="corr-run-b", org_id=None, user_id=None, run_id="run-b")
    assert await turn_index.stamp_question(conversation_id="c", run_id="run-b", released=False) is True
    await RedisStreamBuffer(conversation_id="c", turn_id=2).append("sse", COMPLETED)

    async def answered(buffer):
        await buffer.close("completed")

    with pytest.raises(TurnInProgress) as refused:
        await asyncio.wait_for(RunRegistry().start(
            conversation_id="c", turn_id=2, open_buffer=lambda: backend_mod.new_buffer("c", 2), produce=answered,
        ), timeout=1)

    assert refused.value.turn_id == 2
    assert await _stored(redis) == _claim("run-b", turn_id=2, pod="pod-b")


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


async def test_a_kept_claim_is_renewed_and_stops_once_the_stream_ends(redis, quick_renewals):
    buffer = RedisStreamBuffer(conversation_id="c", turn_id=1)
    lifecycle = TurnLifecycle()
    await turn_claim.acquire(claim=_claim("run-a"), conversation_id="c")
    await redis.pexpire(claim_key("c"), 1_000)
    keeper = asyncio.create_task(turn_claim.keep_claim(
        claim=_claim("run-a"), conversation_id="c", buffer=buffer, lifecycle=lifecycle,
    ))

    async def renewed() -> bool:
        return await redis.pttl(claim_key("c")) > 60_000

    await _until(renewed)
    await buffer.append("sse", COMPLETED)
    await asyncio.wait_for(keeper, timeout=2)

    # Released by the registry after the terminal record, never by the keeper.
    assert await _stored(redis) is not None


async def test_a_claim_another_run_took_is_logged_once_and_taken_back_once_that_run_lets_it_go(
    redis, quick_renewals, monkeypatch, caplog, owned_logger,
):
    """Run A's key lapsed, and run B's start took the claim and then let it go
    again (its buffer failed to open, say). A's keeper used to stop at its
    first lost renewal, so A answered the rest of its turn holding no claim,
    and a send on any replica was admitted beside it."""
    logged = owned_logger(CLAIM_LOGGER, level=logging.WARNING)
    buffer = RedisStreamBuffer(conversation_id="c", turn_id=1)
    await buffer.append("sse", {"sse": "partial"})
    lifecycle = TurnLifecycle()
    await turn_claim.acquire(claim=_claim("run-a"), conversation_id="c")
    outcomes: list[RenewOutcome] = []
    real_renew = turn_claim.renew

    async def recorded_renew(*, claim, conversation_id):
        result = await real_renew(claim=claim, conversation_id=conversation_id)
        outcomes.append(result.outcome)
        return result

    monkeypatch.setattr(turn_claim, "renew", recorded_renew)
    keeper = asyncio.create_task(turn_claim.keep_claim(
        claim=_claim("run-a"), conversation_id="c", buffer=buffer, lifecycle=lifecycle,
    ))
    await redis.set(claim_key("c"), _claim("run-b", pod="pod-b").model_dump_json())

    async def lost_three_times() -> bool:
        return outcomes.count(RenewOutcome.LOST) >= 3

    await _until(lost_three_times)
    assert not keeper.done()
    assert await _stored(redis) == _claim("run-b", pod="pod-b")
    assert await turn_claim.release(conversation_id="c", run_id="run-b") is True

    async def taken_back() -> bool:
        return await _stored(redis) == _claim("run-a")

    await _until(taken_back)
    assert outcomes[-1] in (RenewOutcome.REACQUIRED, RenewOutcome.RENEWED)
    await buffer.close("completed")
    await asyncio.wait_for(keeper, timeout=2)

    assert lifecycle.discarded is False
    records = _warnings(caplog, CLAIM_LOGGER)
    fields = {
        "event": "turn_claim_lost", "conversation_id": "c", "run_id": "run-a",
        "holder_run_id": "run-b", "holder_pod": "pod-b",
    }
    # Once for the loss, not once per renewal that found B holding it.
    assert [{name: getattr(r, name) for name in fields} for r in records] == [fields]
    assert_log_fields(logged.output(), **fields)


async def test_a_redis_outage_while_keeping_a_claim_does_not_lose_it(redis, quick_renewals, monkeypatch):
    buffer = RedisStreamBuffer(conversation_id="c", turn_id=1)
    lifecycle = TurnLifecycle()
    await turn_claim.acquire(claim=_claim("run-a"), conversation_id="c")
    outcomes: list[RenewOutcome] = []
    real_renew = turn_claim.renew
    down = True

    async def recorded_renew(*, claim, conversation_id):
        if down:
            monkeypatch.setattr(turn_claim, "get_redis", lambda: UnreachableRedis())
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
    lifecycle = TurnLifecycle(saved_question=True, question_id=uuid4())

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
    assert (entry.run_id, entry.saved_question) == (handle.run_id, True)
    assert entry.released_at is not None


async def test_a_release_redis_does_not_answer_at_the_end_of_a_turn_is_tried_again_until_it_lands(
    redis, monkeypatch,
):
    """Every replica's next send waits on the release a turn makes after its
    terminal record. One Redis does not answer is tried again, past more
    than one failure, rather than left to lapse: another replica would wait
    ENDING_HOLDER_WAIT_SECONDS before it cleared the claim."""
    monkeypatch.setattr(turn_claim, "REDIS_CALL_SECONDS", 0.05)
    monkeypatch.setattr(turn_claim, "RELEASE_RETRY_SECONDS", 0.01)
    real_release = turn_claim.release
    releases = 0

    async def release_after_two_stalls(**kwargs):
        nonlocal releases
        releases += 1
        if releases <= 2:
            await asyncio.sleep(10)
        return await real_release(**kwargs)

    async def answered(buffer):
        await buffer.append("sse", COMPLETED)
        await buffer.close("completed")

    monkeypatch.setattr(turn_claim, "release", release_after_two_stalls)
    handle = await RunRegistry().start(
        conversation_id="c", turn_id=1, open_buffer=lambda: backend_mod.new_buffer("c", 1), produce=answered,
    )
    await asyncio.wait_for(handle.task, timeout=2)
    assert await _stored(redis) == handle.claim

    async def released() -> bool:
        return await redis.exists(claim_key("c"), CLAIMS_KEY) == 0

    await _until(released)
    assert releases == 3


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


async def test_a_stop_that_lands_while_a_finished_turns_watchers_unwind_still_releases_its_claim(
    redis, monkeypatch,
):
    """A second Stop, or a shutdown, cancels the turn's task while it waits
    for its claim keeper to unwind, after its terminal record. That cancel
    used to skip the release, so the next send on every replica waited for a
    lingering claim."""
    unwinding = asyncio.Event()

    async def keeper_mid_renewal(**_kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            unwinding.set()
            await asyncio.sleep(0.05)
            raise

    async def answered(buffer):
        await buffer.append("sse", COMPLETED)
        await buffer.close("completed")

    monkeypatch.setattr(turn_claim, "keep_claim", keeper_mid_renewal)
    handle = await RunRegistry().start(
        conversation_id="c", turn_id=2, open_buffer=lambda: backend_mod.new_buffer("c", 2), produce=answered,
    )
    await asyncio.wait_for(unwinding.wait(), timeout=2)

    handle.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handle.task

    assert handle.buffer.is_closed
    assert await redis.exists(claim_key("c"), CLAIMS_KEY) == 0


@pytest.mark.parametrize("last_record", ["streaming", "finishing"])
async def test_a_turn_that_ended_with_its_stream_open_holds_off_only_other_replicas(
    redis, monkeypatch, last_record,
):
    """Its terminal record and the seal after it both failed, so its task
    ended with its stream open and it kept its claim. The replica that ran it
    used to refuse the next send too, until the claim lapsed, where it had
    admitted it before claims existed. That replica knows nothing of the run
    writes any more and admits at once; another one cannot tell, and still
    refuses."""
    from cowork.api.v1.endpoints import responses as responses_ep
    from cowork.db.scoped import TenantScope

    monkeypatch.setattr(turn_claim, "ENDING_HOLDER_WAIT_SECONDS", 0.2)
    replica_a, replica_b = RunRegistry(), RunRegistry()
    monkeypatch.setattr(responses_ep, "registry", replica_a)
    conversation_id = str(uuid4())
    record = {"sse": "partial"} if last_record == "streaming" else COMPLETED

    async def ends_open(buffer):
        await buffer.append("sse", record)
        raise RuntimeError("the terminal record could not be written")

    async def answered(buffer):
        await buffer.append("sse", COMPLETED)
        await buffer.close("completed")

    ended = await replica_a.start(
        conversation_id=conversation_id, turn_id=1,
        open_buffer=lambda: backend_mod.new_buffer(conversation_id, 1), produce=ends_open,
    )
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(ended.task, timeout=2)
    assert not ended.buffer.is_closed
    assert await _stored(redis, conversation_id) == ended.claim

    with pytest.raises(TurnInProgress):
        await replica_b.start(
            conversation_id=conversation_id, turn_id=3,
            open_buffer=lambda: backend_mod.new_buffer(conversation_id, 3), produce=answered,
        )
    await responses_ep._refuse_while_answering(conversation=conversation_id, scope=TenantScope(org_mode=False))
    admitted = await replica_a.start(
        conversation_id=conversation_id, turn_id=3,
        open_buffer=lambda: backend_mod.new_buffer(conversation_id, 3), produce=answered,
    )
    await asyncio.wait_for(admitted.task, timeout=2)
    assert await redis.exists(claim_key(conversation_id)) == 0


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
    assert (await turn_index.get_turn(conversation_id)).saved_question is True
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


@pytest.mark.parametrize("deleted_on", ["a", "b"], ids=["deleted-where-it-ran", "deleted-on-the-other-replica"])
async def test_a_delete_whose_index_forget_fails_refuses_the_resend_only_within_the_stale_window(
    redis, monkeypatch, deleted_on,
):
    """Turn 3 saved its question and ended on replica A, and the user deleted
    it, so the next send is turn 3 again. The delete could not forget the
    turn's index entry. Its stamp used to refuse that send on every replica
    until the entry expired, an hour after the turn started; it now refuses
    it only until STALE_START_WINDOW_SECONDS after the turn released its
    claim. A delete that lands on replica B leaves A its finished handle, and
    that handle used to go on refusing the send on A after the window.

    The window's end is set rather than waited for: makes_stale reads it on
    each call, and a wait on the clock fails whenever the first logged
    exception (the failed forget's) imports the provider SDKs inside it."""
    registry_mod = importlib.import_module("cowork.streaming.registry")
    replica_a, replica_b = RunRegistry(), RunRegistry()
    monkeypatch.setattr(registry_mod, "registry", replica_a if deleted_on == "a" else replica_b)
    await _saved_turn(replica_a, conversation_id="d", turn_id=3)
    monkeypatch.setattr(turn_index, "get_sync_redis", lambda: UnreachableRedis())

    await asyncio.to_thread(registry_mod.discard_conversation, "d")

    async def answered(buffer):
        await buffer.append("sse", COMPLETED)
        await buffer.close("completed")

    assert (replica_a.get("d") is not None) is (deleted_on == "b")
    for runs in (replica_a, replica_b):
        with pytest.raises(TurnInProgress):
            await runs.start(
                conversation_id="d", turn_id=3, open_buffer=lambda: backend_mod.new_buffer("d", 3),
                produce=answered,
            )
    monkeypatch.setattr(turn_index, "STALE_START_WINDOW_SECONDS", 0)
    for runs in (replica_a, replica_b):
        admitted = await runs.start(
            conversation_id="d", turn_id=3, open_buffer=lambda: backend_mod.new_buffer("d", 3), produce=answered,
        )
        await asyncio.wait_for(admitted.task, timeout=2)


async def test_a_turn_whose_stamps_failed_is_refused_on_its_replica_only_within_the_stale_window(
    redis, monkeypatch,
):
    """Turn 3 saved its question, and both its stamps failed, so its index
    entry names its run with no stamp, and only replica A's handle knows the
    question was saved. A refuses a start at 3 as the stamp would have, and
    for as long: until STALE_START_WINDOW_SECONDS after the turn ended. The
    handle used to refuse it for as long as the process lived."""
    registry_mod = importlib.import_module("cowork.streaming.registry")

    async def failing_stamp(**_kwargs):
        raise RedisConnectionError("Redis is unreachable")

    monkeypatch.setattr(turn_index, "stamp_question", failing_stamp)
    replica_a = RunRegistry()
    lifecycle = TurnLifecycle(question_id=uuid4())

    async def saves_then_answers(buffer):
        await registry_mod.mark_question_saved(lifecycle=lifecycle, conversation_id="e")
        await buffer.append("sse", COMPLETED)
        await buffer.close("completed")

    async def answered(buffer):
        await buffer.append("sse", COMPLETED)
        await buffer.close("completed")

    ended = await replica_a.start(
        conversation_id="e", turn_id=3, open_buffer=lambda: backend_mod.new_buffer("e", 3),
        produce=saves_then_answers, lifecycle=lifecycle,
    )
    await asyncio.wait_for(ended.task, timeout=2)
    entry = await turn_index.get_turn("e")
    assert (entry.run_id, entry.saved_question) == (ended.run_id, False)

    with pytest.raises(TurnInProgress) as refused:
        await replica_a.start(
            conversation_id="e", turn_id=3, open_buffer=lambda: backend_mod.new_buffer("e", 3), produce=answered,
        )
    assert refused.value.turn_id == 3
    monkeypatch.setattr(turn_index, "STALE_START_WINDOW_SECONDS", 0)
    admitted = await replica_a.start(
        conversation_id="e", turn_id=3, open_buffer=lambda: backend_mod.new_buffer("e", 3), produce=answered,
    )
    await asyncio.wait_for(admitted.task, timeout=2)


async def test_a_delete_whose_index_forget_fails_logs_it_and_still_releases(
    redis, monkeypatch, caplog, owned_logger,
):
    """A forget that cannot reach Redis leaves a stamp that refuses sends at
    or below the deleted turn's number until the stale window after its
    turn's release ends, so it is logged with an event an operator can find.
    The buffers and the claim are still removed."""
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
    monkeypatch.setattr(turn_index, "get_sync_redis", lambda: UnreachableRedis())

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


async def test_the_boot_pass_stops_the_sandbox_turns_of_this_pods_claims_only(redis, monkeypatch):
    """The process that relayed these turns is gone, so nothing reads their
    replies, and the controller would run each one on while the
    conversation's next turn waits behind it. Each of this pod's listings,
    held or lapsed, gets the cancel flag the controller reads; another pod's
    claim is left running."""
    from cowork import server

    monkeypatch.setenv("POD_NAME", "cowork-server-abc")
    await turn_claim.acquire(claim=_claim("run-held", pod="cowork-server-abc"), conversation_id="c1")
    await turn_claim.acquire(claim=_claim("run-lapsed", pod="cowork-server-abc"), conversation_id="c2")
    await redis.delete(claim_key("c2"))
    await turn_claim.acquire(claim=_claim("run-elsewhere", pod="cowork-server-xyz"), conversation_id="c3")

    await server._release_own_turn_claims()

    flagged = {
        run_id: await redis.exists(cancel_flag_key(f"corr-{run_id}"))
        for run_id in ("run-held", "run-lapsed", "run-elsewhere")
    }
    assert flagged == {"run-held": 1, "run-lapsed": 1, "run-elsewhere": 0}
    assert await redis.hkeys(CLAIMS_KEY) == ["c3"]


def test_the_lifespan_releases_this_pods_claims_before_it_admits_a_turn():
    """The boot pass's own tests call it directly, so they pass with the
    lifespan's call gone. The call has to run, and before the scheduler,
    channels and the port start turns: a turn of this process would hold a
    claim the pass then releases."""
    from cowork import server

    source = inspect.getsource(server.lifespan)
    calls = ["await _release_own_turn_claims()", "start_scheduler()", "await _start_channels(app)", "        yield\n"]
    positions = [source.find(call) for call in calls]
    assert -1 not in positions, dict(zip(calls, positions))
    assert positions == sorted(positions), dict(zip(calls, positions))


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
    reach no Redis there."""
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
    for module in (turn_claim, turn_index, backend_mod):
        monkeypatch.setattr(module, "get_sync_redis", no_redis(f"{module.__name__}.get_sync_redis"))
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


async def test_the_file_backend_logs_a_stale_refusal_at_info(monkeypatch, tmp_path, caplog, owned_logger):
    """Desktop logs a second send that read the history before its turn's
    save at INFO, as it did before the claim. Only the Redis backend, where
    another replica's turn can be what refuses it, logs it at WARNING."""
    registry_mod = importlib.import_module("cowork.streaming.registry")
    owned_logger(registry_mod.logger.name, level=logging.INFO)
    monkeypatch.setenv("COWORK_STREAM_BACKEND", "file")
    runs = RunRegistry()

    async def answered(buffer):
        await buffer.append("sse", COMPLETED)
        await buffer.close("completed")

    ended = await runs.start(
        conversation_id="c", turn_id=3, open_buffer=opens(FileStreamBuffer(tmp_path / "turn3.jsonl")),
        produce=answered, lifecycle=TurnLifecycle(saved_question=True),
    )
    await asyncio.wait_for(ended.task, timeout=2)

    with pytest.raises(TurnInProgress):
        await runs.start(
            conversation_id="c", turn_id=3, open_buffer=opens(FileStreamBuffer(tmp_path / "again.jsonl")),
            produce=answered,
        )

    records = [
        r for r in caplog.records
        if r.name == registry_mod.logger.name and getattr(r, "event", None) == "turn_stale_refused"
    ]
    assert [r.levelno for r in records] == [logging.INFO]


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


async def test_a_reader_ends_a_quiet_stream_once_its_runs_claim_lapses(redis, monkeypatch):
    """A relay died holding its claim and left its stream open. A reader that
    finds the claim held keeps waiting, and checks again after each idle
    window, so once the claim lapses the reader ends the stream with an
    Interrupted record rather than waiting on it forever."""
    monkeypatch.setattr(buffer_mod, "REDIS_TAIL_IDLE_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr(RedisStreamBuffer, "_BLOCK_MS", 20)
    await RedisStreamBuffer(conversation_id="c", turn_id=1).append("sse", {"sse": "partial"})
    await redis.set(claim_key("c"), _claim("run-dead").model_dump_json(), px=300)
    checks: list[bool] = []
    real_check = RedisStreamBuffer._writer_holds_claim

    async def recorded_check(self, **kwargs):
        checks.append(await real_check(self, **kwargs))
        return checks[-1]

    monkeypatch.setattr(RedisStreamBuffer, "_writer_holds_claim", recorded_check)

    records = await asyncio.wait_for(
        _drain(RedisStreamBuffer(conversation_id="c", turn_id=1).tail(from_seq=0)), timeout=3,
    )

    assert checks[0] is True and checks[-1] is False
    assert [r.type for r in records][-1] == "Interrupted"


async def test_a_reader_whose_turn_finishes_at_its_idle_check_yields_the_turns_last_frames(
    redis, monkeypatch,
):
    """The turn writes its last frame and its terminal record, then releases
    its claim, between a reader's empty read and that reader's claim check.
    The reader yields every record after the last one it read, not only the
    terminal record, so the answer's last frame still reaches the user."""
    monkeypatch.setattr(buffer_mod, "REDIS_TAIL_IDLE_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr(RedisStreamBuffer, "_BLOCK_MS", 20)
    writer = RedisStreamBuffer(conversation_id="c", turn_id=1)
    await writer.append("sse", {"sse": "partial"})
    real_xread = buffer_mod.xread_blocking
    reads = 0

    async def xread_that_ran_out_as_the_turn_finished(**kwargs):
        nonlocal reads
        reads += 1
        if reads == 1:
            await writer.append("sse", COMPLETED)
            await writer.close("completed")
            return []
        return await real_xread(**kwargs)

    monkeypatch.setattr(buffer_mod, "xread_blocking", xread_that_ran_out_as_the_turn_finished)

    records = await asyncio.wait_for(
        _drain(RedisStreamBuffer(conversation_id="c", turn_id=1).tail(from_seq=0)), timeout=2,
    )

    assert [(r.type, r.data) for r in records] == [
        ("sse", {"sse": "partial"}), ("sse", COMPLETED), ("Done", {"reason": "completed"}),
    ]
    entries = await redis.xrange(stream_key("c", 1))
    assert [fields["type"] for _id, fields in entries] == ["sse", "sse", "Done"]


async def _drain(records):
    return [record async for record in records]
