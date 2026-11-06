"""Which run answers a conversation now, agreed by every replica through Redis.

Hosted Cowork runs several replicas, and a question can reach any of them.
RunRegistry refuses a second question while a turn is still answering, but its
memory knows only the turns its own process started. The claim is the part
every replica can read: a lease per conversation, taken by the run that starts
a turn, renewed while that run lives, and released once the turn's stream has a
terminal record. A start that finds another live run holding it is refused
with the same 409 a second question on the same replica gets.

Only the Redis stream backend uses it (``backend.get_backend() == "redis"``).
The file backend serves one process, whose registry already sees every turn.

The key holds the holder's TurnClaim as JSON, written with a TTL, so the claim
of a replica that died without releasing it lapses on its own. The run id is
the owner token: a new one per start, so a restarted process never takes a
dead run's claim for its own. A hash lists every claim by conversation, so a
process can find the claims stamped with its pod name without scanning keys.

Compare-and-set runs as WATCH/MULTI rather than a Lua script: the fakeredis
the tests use runs no EVAL without lupa, which this repository does not
install. Same choice as cowork.channels.ingress_lease, whose renew never takes
a lapsed lease back and reads a Redis error as a lost lease; a turn needs both
the other way round, so this module does not reuse it.

The claim does not order database writes. A save that outlives its producer
at shutdown (handlers.responses._saves_past_their_deadline) can still be
writing after its turn's claim is released.

Keys:
    cowork:turn_claim:{conversation_id}   STRING, the holder's TurnClaim JSON, PX CLAIM_TTL_SECONDS
    cowork:turn_claims                    HASH conversation_id -> TurnClaim JSON
"""
from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from collections.abc import Awaitable
from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, ValidationError
from redis.exceptions import WatchError

from cowork.common.logger import log_context
from cowork.streaming.buffer import RedisStreamBuffer, _is_final_client_frame, stream_key
from cowork.turnqueue.redis_client import get_redis, get_sync_redis

if TYPE_CHECKING:
    from cowork.streaming.buffer import StreamBuffer
    from cowork.streaming.registry import TurnLifecycle

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Longer than the stall the hosted liveness probe forgives (failureThreshold 6
# times periodSeconds 10, deployment/cowork-server/values.yaml), so a process
# whose event loop stalls and recovers still holds its claim afterwards.
# tests/test_turn_claim.py pins the two against each other.
CLAIM_TTL_SECONDS = 70
RENEW_INTERVAL_SECONDS = 10
# Bound on each Redis call here: get_redis() sets no socket timeout.
REDIS_CALL_SECONDS = 5
# How long admission waits for a holder that is writing its final client
# frame. The web UI sends a queued question on that frame, and the frame's
# terminal record follows within one Redis round trip.
FINISHING_HOLDER_WAIT_SECONDS = 10
_FINISHING_HOLDER_POLL_SECONDS = 0.1
# Tries of one compare-and-set after another client changed a watched key.
_WATCH_ATTEMPTS = 3

CLAIMS_KEY = "cowork:turn_claims"


def claim_key(conversation_id: str) -> str:
    return f"cowork:turn_claim:{conversation_id}"


def pod_name() -> str:
    """This process's pod: POD_NAME, which the hosted chart's deployment
    subchart sets from the pod's metadata, else the host name, which
    Kubernetes also sets to the pod's name."""
    return os.environ.get("POD_NAME") or socket.gethostname()


class TurnClaim(BaseModel):
    """The run holding a conversation's claim, as its key stores it.

    ``question_id`` is the turn's saved question, filled in by a renewal once
    the producer has saved it. ``pod`` names the process that took it, for a
    boot pass that releases the claims its own pod left behind.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    turn_id: int
    correlation_id: str | None = None
    question_id: UUID | None = None
    pod: str
    started_at: float


def new_claim(
    *, run_id: str, turn_id: int, correlation_id: str | None, question_id: UUID | None,
) -> TurnClaim:
    """A claim for a run this process is starting now."""
    return TurnClaim(
        run_id=run_id, turn_id=turn_id, correlation_id=correlation_id,
        question_id=question_id, pod=pod_name(), started_at=time.time(),
    )


class TurnClaimUnavailable(Exception):
    """Redis could not say whether another run holds the conversation, so the
    turn is refused rather than admitted on a guess.

    The app answers it with 503 and Retry-After (cowork.server), which reads
    ``status_code``.
    """

    status_code = 503

    def __init__(self, *, conversation_id: str) -> None:
        super().__init__(f"could not read the turn claim of conversation {conversation_id}")
        self.conversation_id = conversation_id


class AcquireResult(NamedTuple):
    """``holder`` is the claim on the conversation after the call: the
    caller's own when ``acquired``, another run's otherwise."""

    acquired: bool
    holder: TurnClaim


class RenewOutcome(StrEnum):
    # The run still held its claim, and its TTL starts again.
    RENEWED = "renewed"
    # The key was gone and the run's stream has no terminal record, so the
    # run took it back.
    REACQUIRED = "reacquired"
    # Another run holds the claim, or someone else ended the run's stream
    # while its key was gone.
    LOST = "lost"
    # Redis did not answer. Says nothing about who holds the claim.
    ERROR = "error"


class RenewResult(NamedTuple):
    outcome: RenewOutcome
    # The other run, when LOST because one holds the key.
    holder: TurnClaim | None = None


class StreamState(StrEnum):
    """A turn stream's state, read from its last record."""

    # No record yet: the turn has just started, or its remote worker is
    # still starting.
    EMPTY = "empty"
    LIVE = "live"
    # The last record is a final client frame; the terminal record follows.
    FINISHING = "finishing"
    ENDED = "ended"


def _parse(raw: str | None) -> TurnClaim | None:
    """The claim a stored value holds, or None for no value. A value no
    TurnClaim parses raises ValidationError: nothing but this module writes
    these keys, so it is not read as either answer."""
    return None if raw is None else TurnClaim.model_validate_json(raw)


def _held_by(*, raw: str | None, run_id: str) -> bool:
    """Whether a stored value is ``run_id``'s claim. A value no TurnClaim
    parses was not written for this run, so it is not this run's."""
    try:
        claim = _parse(raw)
    except ValidationError:
        return False
    return claim is not None and claim.run_id == run_id


def _state_of(last_entries: list) -> StreamState:
    """The state XREVRANGE ... COUNT 1 shows for a turn stream."""
    if not last_entries:
        return StreamState.EMPTY
    record = RedisStreamBuffer._record(last_entries[0][1])
    if record.is_terminal:
        return StreamState.ENDED
    if _is_final_client_frame(record.type, record.data):
        return StreamState.FINISHING
    return StreamState.LIVE


async def _bounded(call: Awaitable[T]) -> T:
    async with asyncio.timeout(REDIS_CALL_SECONDS):
        return await call


async def stream_state(*, conversation_id: str, turn_id: int, client=None) -> StreamState:
    """The state of a turn's stream. Unbounded; the caller bounds it."""
    r = client if client is not None else get_redis()
    return _state_of(await r.xrevrange(stream_key(conversation_id, turn_id), count=1))


async def read_claim(*, conversation_id: str, client=None) -> TurnClaim | None:
    """The conversation's claim, or None when no run holds it. Unbounded;
    the caller bounds it."""
    r = client if client is not None else get_redis()
    return _parse(await r.get(claim_key(conversation_id)))


async def acquire(*, claim: TurnClaim, conversation_id: str) -> AcquireResult:
    """Take the conversation's claim for ``claim`` unless another run holds it.

    The same transaction deletes the stream at this turn number when that
    stream has ended, which happens when the turn before at this number saved
    no message (backend.new_buffer clears it the same way). Deleted with the
    claim rather than when the buffer opens, so another replica that finds
    this claim never reads the earlier turn's terminal record as this run's.

    Raises on a Redis error. Unbounded; the caller bounds it, and removes this
    run's claim by compare-and-delete when a call is cut short, since Redis
    may have applied it.
    """
    r = get_redis()
    key = claim_key(conversation_id)
    stream = stream_key(conversation_id, claim.turn_id)
    value = claim.model_dump_json()
    for _attempt in range(_WATCH_ATTEMPTS):
        try:
            async with r.pipeline(transaction=True) as pipe:
                await pipe.watch(key, stream)
                holder = _parse(await pipe.get(key))
                if holder is not None:
                    return AcquireResult(acquired=holder.run_id == claim.run_id, holder=holder)
                ended = _state_of(await pipe.xrevrange(stream, count=1)) is StreamState.ENDED
                pipe.multi()
                pipe.set(key, value, px=CLAIM_TTL_SECONDS * 1000)
                pipe.hset(CLAIMS_KEY, conversation_id, value)
                if ended:
                    pipe.delete(stream)
                await pipe.execute()
                return AcquireResult(acquired=True, holder=claim)
        except WatchError:
            continue
    raise WatchError(f"the turn claim of {conversation_id} kept changing while it was taken")


async def renew(*, claim: TurnClaim, conversation_id: str) -> RenewResult:
    """Start the claim's TTL again while ``claim``'s run holds it, writing
    ``claim`` as its value.

    A key that is gone (it lapsed in a stall, or Redis lost it to an eviction
    or a failover) is taken back while the run's stream has no terminal
    record. A terminal record there means another reader ended the turn while
    the key was gone, which counts as lost.

    Never raises. A Redis error, a timeout or a value no TurnClaim parses is
    logged and answered ERROR, which never means lost.
    """
    try:
        return await _bounded(_renew(claim=claim, conversation_id=conversation_id))
    except Exception:
        logger.warning(
            "Could not renew a turn claim; the next renewal tries again.",
            exc_info=True,
            extra={
                "event": "turn_claim_renew_failed", "run_id": claim.run_id,
                **log_context(conversation_id=conversation_id),
            },
        )
        return RenewResult(RenewOutcome.ERROR)


async def _renew(*, claim: TurnClaim, conversation_id: str) -> RenewResult:
    r = get_redis()
    key = claim_key(conversation_id)
    value = claim.model_dump_json()
    for _attempt in range(_WATCH_ATTEMPTS):
        try:
            async with r.pipeline(transaction=True) as pipe:
                await pipe.watch(key)
                holder = _parse(await pipe.get(key))
                if holder is not None and holder.run_id != claim.run_id:
                    return RenewResult(RenewOutcome.LOST, holder)
                if holder is None:
                    # Watched only on this path: the run's own appends change
                    # the stream many times a second while it holds the key.
                    stream = stream_key(conversation_id, claim.turn_id)
                    await pipe.watch(stream)
                    if _state_of(await pipe.xrevrange(stream, count=1)) is StreamState.ENDED:
                        return RenewResult(RenewOutcome.LOST)
                pipe.multi()
                pipe.set(key, value, px=CLAIM_TTL_SECONDS * 1000)
                pipe.hset(CLAIMS_KEY, conversation_id, value)
                await pipe.execute()
                return RenewResult(RenewOutcome.RENEWED if holder is not None else RenewOutcome.REACQUIRED)
        except WatchError:
            continue
    raise WatchError(f"the turn claim of {conversation_id} kept changing while it was renewed")


async def release(*, conversation_id: str, run_id: str) -> bool:
    """Delete the conversation's claim if ``run_id`` holds it, and the claims
    hash's entry if that entry is ``run_id``'s. True when it deleted the
    claim.

    Raises on a Redis error. Unbounded; the caller bounds it.
    """
    r = get_redis()
    key = claim_key(conversation_id)
    for _attempt in range(_WATCH_ATTEMPTS):
        try:
            async with r.pipeline(transaction=True) as pipe:
                # The hash entry is not watched: no other run can write it
                # while this run's key exists, and a run that takes the key
                # changes the key, which is watched.
                await pipe.watch(key)
                holds = _held_by(raw=await pipe.get(key), run_id=run_id)
                listed = _held_by(raw=await pipe.hget(CLAIMS_KEY, conversation_id), run_id=run_id)
                if not (holds or listed):
                    return False
                pipe.multi()
                if holds:
                    pipe.delete(key)
                if listed:
                    pipe.hdel(CLAIMS_KEY, conversation_id)
                await pipe.execute()
                return holds
        except WatchError:
            continue
    raise WatchError(f"the turn claim of {conversation_id} kept changing while it was released")


def release_sync(*, conversation_id: str, run_id: str) -> bool:
    """Blocking ``release``, for conversation delete, which runs in a
    threadpool thread with no event loop."""
    r = get_sync_redis()
    key = claim_key(conversation_id)
    for _attempt in range(_WATCH_ATTEMPTS):
        try:
            with r.pipeline(transaction=True) as pipe:
                pipe.watch(key)
                holds = _held_by(raw=pipe.get(key), run_id=run_id)
                listed = _held_by(raw=pipe.hget(CLAIMS_KEY, conversation_id), run_id=run_id)
                if not (holds or listed):
                    return False
                pipe.multi()
                if holds:
                    pipe.delete(key)
                if listed:
                    pipe.hdel(CLAIMS_KEY, conversation_id)
                pipe.execute()
                return holds
        except WatchError:
            continue
    raise WatchError(f"the turn claim of {conversation_id} kept changing while it was released")


async def release_quietly(*, conversation_id: str, run_id: str) -> None:
    """``release``, bounded, for a caller that is already ending its turn or
    its admission. A failure is logged as a WARNING; the claim then lapses
    with its TTL."""
    try:
        await _bounded(release(conversation_id=conversation_id, run_id=run_id))
    except Exception:
        logger.warning(
            "Could not release a turn claim; it lapses with its TTL.",
            exc_info=True,
            extra={
                "event": "turn_claim_release_failed", "run_id": run_id,
                **log_context(conversation_id=conversation_id),
            },
        )


async def claim_turn(*, claim: TurnClaim, conversation_id: str) -> AcquireResult:
    """Admission's claim: take it, or find whether the run holding it still
    answers.

    A holder whose stream has no record yet or is streaming keeps the claim,
    and the result says so. A holder writing its final client frame is waited
    for, up to FINISHING_HOLDER_WAIT_SECONDS: the web UI sends a queued
    question on that frame, as RunRegistry.start waits for a turn of its own
    process. A holder whose stream has ended while its claim lingers (its
    release failed, or it is still unwinding) has that claim deleted by
    compare-and-delete, and this run tries once more. Nothing here writes to
    another run's stream.

    Raises TurnClaimUnavailable when Redis does not answer. An acquire cut
    short first has this run's claim removed, since Redis may have applied it,
    so a client's retry does not meet the claim of its own refused attempt.
    """
    result = await _acquire_or_undo(claim=claim, conversation_id=conversation_id)
    if result.acquired:
        return result
    try:
        ended = await _bounded(_holder_has_ended(holder=result.holder, conversation_id=conversation_id))
        if not ended:
            ended = await _wait_for_finishing_holder(holder=result.holder, conversation_id=conversation_id)
        if not ended:
            return result
        await _bounded(release(conversation_id=conversation_id, run_id=result.holder.run_id))
    except Exception as exc:
        raise TurnClaimUnavailable(conversation_id=conversation_id) from exc
    return await _acquire_or_undo(claim=claim, conversation_id=conversation_id)


async def _acquire_or_undo(*, claim: TurnClaim, conversation_id: str) -> AcquireResult:
    from cowork.db.units import run_to_completion

    try:
        # A cancel of the admission waits for the call, so a claim Redis
        # applied is known to be there when it is removed below.
        return await run_to_completion(_bounded(acquire(claim=claim, conversation_id=conversation_id)))
    except BaseException as exc:
        await release_quietly(conversation_id=conversation_id, run_id=claim.run_id)
        if isinstance(exc, Exception):
            raise TurnClaimUnavailable(conversation_id=conversation_id) from exc
        raise


async def _holder_has_ended(*, holder: TurnClaim, conversation_id: str) -> bool:
    """Whether ``holder`` no longer answers: its claim is gone or names
    another run, or its stream has a terminal record."""
    r = get_redis()
    current = await read_claim(conversation_id=conversation_id, client=r)
    if current is None or current.run_id != holder.run_id:
        return True
    state = await stream_state(conversation_id=conversation_id, turn_id=holder.turn_id, client=r)
    return state is StreamState.ENDED


async def _wait_for_finishing_holder(*, holder: TurnClaim, conversation_id: str) -> bool:
    """Wait for a holder writing its final client frame to end, polling until
    FINISHING_HOLDER_WAIT_SECONDS pass. False at once for a holder that is
    not finishing, and when the wait runs out."""
    r = get_redis()
    state = await _bounded(stream_state(conversation_id=conversation_id, turn_id=holder.turn_id, client=r))
    if state is not StreamState.FINISHING:
        return False
    loop = asyncio.get_running_loop()
    deadline = loop.time() + FINISHING_HOLDER_WAIT_SECONDS
    while loop.time() < deadline:
        await asyncio.sleep(_FINISHING_HOLDER_POLL_SECONDS)
        if await _bounded(_holder_has_ended(holder=holder, conversation_id=conversation_id)):
            return True
    return False


async def live_holder(*, conversation_id: str) -> TurnClaim | None:
    """The run holding the conversation while its stream has no record yet
    or is streaming. None when no run holds it, and when the holder is
    writing its final frames or has ended, which claim_turn settles.

    Raises TurnClaimUnavailable when Redis does not answer.
    """
    try:
        async with asyncio.timeout(REDIS_CALL_SECONDS):
            r = get_redis()
            holder = await read_claim(conversation_id=conversation_id, client=r)
            if holder is None:
                return None
            state = await stream_state(conversation_id=conversation_id, turn_id=holder.turn_id, client=r)
    except Exception as exc:
        raise TurnClaimUnavailable(conversation_id=conversation_id) from exc
    return holder if state in (StreamState.EMPTY, StreamState.LIVE) else None


async def claim_held_for_turn(*, conversation_id: str, turn_id: int, client=None) -> bool:
    """Whether a run holds the conversation's claim for this turn number.
    Raises on a Redis error. Unbounded; the caller bounds it."""
    holder = await read_claim(conversation_id=conversation_id, client=client)
    return holder is not None and holder.turn_id == turn_id


async def keep_claim(
    *, claim: TurnClaim, conversation_id: str, buffer: StreamBuffer, lifecycle: TurnLifecycle,
) -> None:
    """Renew a running turn's claim every RENEW_INTERVAL_SECONDS until its
    stream is ending.

    Stops once the turn's buffer is closed or writing its final client frame,
    as cancellation.watch_cancel does; the claim then stays until the registry
    releases it after the terminal record. Stops for a discarded turn too,
    whose stream discard_conversation deleted. A renewal that finds another
    run holding the claim, or the stream ended by someone else, sets
    ``lifecycle.claim_lost``, logs turn_claim_lost and stops. A Redis error is
    tried again at the next interval.

    Once the producer has saved the question, each renewal writes the claim
    with its id.
    """
    current = claim
    while True:
        await asyncio.sleep(RENEW_INTERVAL_SECONDS)
        if buffer.is_closed or buffer.is_finishing or lifecycle.discarded:
            return
        if lifecycle.question_id is not None and current.question_id != lifecycle.question_id:
            current = current.model_copy(update={"question_id": lifecycle.question_id})
        result = await renew(claim=current, conversation_id=conversation_id)
        if result.outcome is not RenewOutcome.LOST:
            continue
        if buffer.is_closed or buffer.is_finishing:
            # The run's own terminal record landed while the renewal read it.
            return
        lifecycle.claim_lost = True
        logger.warning(
            "A running turn lost its claim on the conversation to another run or reader.",
            extra={
                "event": "turn_claim_lost",
                "run_id": current.run_id,
                "holder_run_id": result.holder.run_id if result.holder is not None else None,
                "holder_pod": result.holder.pod if result.holder is not None else None,
                **log_context(
                    conversation_id=conversation_id, request_id=current.correlation_id,
                ),
            },
        )
        return


async def release_pod_claims(*, pod: str) -> int:
    """Release every claim stamped with ``pod``, and return how many it
    deleted.

    For a process that has just started: it holds no turn yet, so a claim
    carrying its pod's name was left by an earlier process in the same pod,
    such as one a liveness restart replaced. That holds only while a pod runs
    one server process, which tests/test_turn_claim.py pins for the hosted
    command. Unbounded; the caller bounds it.
    """
    released = 0
    for conversation_id, raw in (await get_redis().hgetall(CLAIMS_KEY)).items():
        try:
            claim = TurnClaim.model_validate_json(raw)
        except ValidationError:
            logger.warning(
                "A turn claims entry holds no readable claim; the boot pass leaves it.",
                extra={"event": "turn_claim_unreadable", **log_context(conversation_id=conversation_id)},
            )
            continue
        if claim.pod == pod and await release(conversation_id=conversation_id, run_id=claim.run_id):
            released += 1
    return released
