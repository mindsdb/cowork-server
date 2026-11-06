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
A claim that lapses instead of being released (its pod was killed) keeps its
listing until a boot pass (release_pod_claims) on any pod removes it.

Taking the claim also reads the conversation's turn index entry in the same
transaction, and refuses a start that entry shows read the history before an
earlier run saved its question (turn_index.TurnIndexEntry.makes_stale).

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
    cowork:turn_claims                    HASH conversation_id -> TurnClaim JSON, including
                                          lapsed claims until a boot pass removes them
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

# Longer than the longest stall the hosted liveness probe forgives plus the
# most a claim can have run down when the stall starts, so a process whose
# event loop stalls and recovers still holds its claim afterwards. The probe
# (deployment/cowork-server/values.yaml) forgives up to failureThreshold 6
# times periodSeconds 10, plus timeoutSeconds 5 for the probe that answers
# late: 65 s. A claim has run down by up to one renewal interval plus one
# renewal's Redis bound when the stall starts: 15 s. tests/test_turn_claim.py
# pins the sum against this. The cost: a send into a conversation whose pod
# was force-killed is refused for up to this long.
CLAIM_TTL_SECONDS = 90
RENEW_INTERVAL_SECONDS = 10
# Bound on each Redis call here: get_redis() sets no socket timeout.
REDIS_CALL_SECONDS = 5
# How long admission waits for a holder that is writing its final client
# frame, or whose stream has ended while it still holds the claim, to release
# it. The web UI sends a queued question on that frame. The terminal record
# follows within one Redis round trip, and the release a moment after.
ENDING_HOLDER_WAIT_SECONDS = 10
_ENDING_HOLDER_POLL_SECONDS = 0.1
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


class AcquireOutcome(StrEnum):
    # The caller's run holds the claim.
    ACQUIRED = "acquired"
    # Another run holds it.
    HELD = "held"
    # No run holds it, and the conversation's turn index entry shows that a
    # run saved its question at this turn number or a later one: the start
    # read the history before that save.
    STALE = "stale"


class AcquireResult(NamedTuple):
    """``holder`` is the claim on the conversation after the call: the
    caller's own when ACQUIRED, another run's when HELD, None when STALE.
    ``saved_turn_id`` is the turn whose saved question makes a STALE start
    stale."""

    outcome: AcquireOutcome
    holder: TurnClaim | None = None
    saved_turn_id: int | None = None

    @property
    def acquired(self) -> bool:
        return self.outcome is AcquireOutcome.ACQUIRED


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
    """Take the conversation's claim for ``claim`` unless another run holds it
    or the start is stale.

    Stale means the conversation's turn index entry shows that an earlier run
    saved its question at this turn number or a later one
    (turn_index.TurnIndexEntry.makes_stale), so the start read the history
    before that save. Read in the same transaction that takes the claim, so
    no start is admitted past that stamp, however it reached here.

    The same transaction deletes the stream at this turn number when that
    stream has ended, which happens when the turn before at this number saved
    no message (backend.new_buffer clears it the same way). Deleted with the
    claim rather than when the buffer opens, so another replica that finds
    this claim never reads the earlier turn's terminal record as this run's.
    A stale start deletes nothing.

    Raises on a Redis error. Unbounded; the caller bounds it, and removes this
    run's claim by compare-and-delete when a call is cut short, since Redis
    may have applied it.
    """
    # turn_index imports this module.
    from cowork.streaming.turn_index import TurnIndexEntry, turn_key

    r = get_redis()
    key = claim_key(conversation_id)
    stream = stream_key(conversation_id, claim.turn_id)
    index = turn_key(conversation_id)
    value = claim.model_dump_json()
    for _attempt in range(_WATCH_ATTEMPTS):
        try:
            async with r.pipeline(transaction=True) as pipe:
                await pipe.watch(key, stream, index)
                holder = _parse(await pipe.get(key))
                if holder is not None:
                    held = AcquireOutcome.ACQUIRED if holder.run_id == claim.run_id else AcquireOutcome.HELD
                    return AcquireResult(held, holder)
                turn = await pipe.hgetall(index)
                entry = TurnIndexEntry.from_index(turn) if turn else None
                if entry is not None and entry.makes_stale(turn_id=claim.turn_id):
                    return AcquireResult(AcquireOutcome.STALE, saved_turn_id=entry.turn_id)
                ended = _state_of(await pipe.xrevrange(stream, count=1)) is StreamState.ENDED
                pipe.multi()
                pipe.set(key, value, px=CLAIM_TTL_SECONDS * 1000)
                pipe.hset(CLAIMS_KEY, conversation_id, value)
                if ended:
                    pipe.delete(stream)
                await pipe.execute()
                return AcquireResult(AcquireOutcome.ACQUIRED, claim)
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
    and the result says HELD. A holder writing its final client frame, or
    whose stream has ended while it still holds the claim, is waited for
    until it releases the claim, up to ENDING_HOLDER_WAIT_SECONDS. The web UI
    sends a queued question on that frame, as RunRegistry.start waits for a
    turn of its own process. A holder's release stamps its saved question in
    the turn index first (registry._end_claim), so the acquire after the wait
    refuses a stale start as STALE. A claim still held over an ended stream
    when the wait runs out (its release failed, or its process died) is
    deleted by compare-and-delete, and this run tries once more. Nothing here
    writes to another run's stream.

    Raises TurnClaimUnavailable when Redis does not answer. An acquire cut
    short first has this run's claim removed, since Redis may have applied it,
    so a client's retry does not meet the claim of its own refused attempt.
    """
    result = await _acquire_or_undo(claim=claim, conversation_id=conversation_id)
    if result.outcome is not AcquireOutcome.HELD:
        return result
    try:
        waited = await _wait_for_ending_holder(holder=result.holder, conversation_id=conversation_id)
        if waited is _HolderWait.ANSWERING:
            return result
        if waited is _HolderWait.LINGERING:
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


class _HolderWait(StrEnum):
    """How admission's wait for another run's claim ended."""

    # The holder's stream has no record yet or is streaming, or it was still
    # writing its final client frame when the wait ran out.
    ANSWERING = "answering"
    # The holder's claim is gone, or names another run now.
    RELEASED = "released"
    # The holder's stream has ended, and its claim was still there when the
    # wait ran out.
    LINGERING = "lingering"


async def _holder_released(*, holder: TurnClaim, conversation_id: str, client) -> bool:
    """Whether ``holder``'s claim is gone or names another run. Unbounded;
    the caller bounds it."""
    current = await read_claim(conversation_id=conversation_id, client=client)
    return current is None or current.run_id != holder.run_id


async def _wait_for_ending_holder(*, holder: TurnClaim, conversation_id: str) -> _HolderWait:
    """Wait for a holder writing its final client frame, or whose stream has
    ended, to release its claim, polling until ENDING_HOLDER_WAIT_SECONDS
    pass. ANSWERING at once for a holder whose stream has no record yet or is
    streaming."""
    r = get_redis()
    if await _bounded(_holder_released(holder=holder, conversation_id=conversation_id, client=r)):
        return _HolderWait.RELEASED
    state = await _bounded(stream_state(conversation_id=conversation_id, turn_id=holder.turn_id, client=r))
    if state in (StreamState.EMPTY, StreamState.LIVE):
        return _HolderWait.ANSWERING
    loop = asyncio.get_running_loop()
    deadline = loop.time() + ENDING_HOLDER_WAIT_SECONDS
    while loop.time() < deadline:
        await asyncio.sleep(_ENDING_HOLDER_POLL_SECONDS)
        if await _bounded(_holder_released(holder=holder, conversation_id=conversation_id, client=r)):
            return _HolderWait.RELEASED
    state = await _bounded(stream_state(conversation_id=conversation_id, turn_id=holder.turn_id, client=r))
    return _HolderWait.LINGERING if state is StreamState.ENDED else _HolderWait.ANSWERING


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
    whose stream discard_conversation deleted. A renewal that was already
    running when the delete landed can find the key gone and the stream
    gone with it, and take the claim back; the keeper then releases it
    again. A renewal that finds another run holding the claim, or the stream
    ended by someone else, sets ``lifecycle.claim_lost``, logs
    turn_claim_lost and stops. A Redis error is tried again at the next
    interval.

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
        if lifecycle.discarded:
            if result.outcome in (RenewOutcome.RENEWED, RenewOutcome.REACQUIRED):
                await release_quietly(conversation_id=conversation_id, run_id=current.run_id)
            return
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


class BootPassResult(NamedTuple):
    # This pod's claims the pass deleted.
    released: int
    # Listings of other pods' claims that had lapsed, which the pass removed.
    pruned: int


async def release_pod_claims(*, pod: str) -> BootPassResult:
    """Release every claim stamped with ``pod``, and remove the listings of
    claims that lapsed without a release.

    For a process that has just started: it holds no turn yet, so a claim
    carrying its pod's name was left by an earlier process in the same pod,
    such as one a liveness restart replaced. That holds only while a pod runs
    one server process, which tests/test_turn_claim.py pins for the hosted
    command.

    Only a release removes a listing, so a pod killed mid-turn (a SIGKILL, an
    OOM kill, a lost node) leaves the listings of its claims behind once they
    lapse, under a pod name no later process carries. Every boot pass removes
    the listings whose claim key is gone or names another run. A claim that
    has not lapsed yet is left to the next boot pass. Unbounded; the caller
    bounds it.
    """
    r = get_redis()
    listed: dict[str, TurnClaim] = {}
    for conversation_id, raw in (await r.hgetall(CLAIMS_KEY)).items():
        try:
            listed[conversation_id] = TurnClaim.model_validate_json(raw)
        except ValidationError:
            logger.warning(
                "A turn claims entry holds no readable claim; the boot pass leaves it.",
                extra={"event": "turn_claim_unreadable", **log_context(conversation_id=conversation_id)},
            )
    if not listed:
        return BootPassResult(released=0, pruned=0)
    keys = await r.mget([claim_key(conversation_id) for conversation_id in listed])
    released = pruned = 0
    for (conversation_id, claim), raw_key in zip(listed.items(), keys, strict=True):
        if claim.pod == pod:
            if await release(conversation_id=conversation_id, run_id=claim.run_id):
                released += 1
        elif not _held_by(raw=raw_key, run_id=claim.run_id):
            if await _unlist_lapsed(conversation_id=conversation_id, run_id=claim.run_id):
                pruned += 1
    return BootPassResult(released=released, pruned=pruned)


async def _unlist_lapsed(*, conversation_id: str, run_id: str) -> bool:
    """Remove ``run_id``'s listing while its claim key is gone or names
    another run. True when it removed it.

    The key is watched: every write of a listing writes the key in the same
    transaction (acquire, renew), so a run that takes the claim meanwhile,
    or this run taking its own back, aborts the removal. A key that keeps
    changing through every attempt is being written, so the listing stays
    for the next boot pass and the answer is False. A Redis error raises.
    """
    r = get_redis()
    key = claim_key(conversation_id)
    for _attempt in range(_WATCH_ATTEMPTS):
        try:
            async with r.pipeline(transaction=True) as pipe:
                await pipe.watch(key)
                if _held_by(raw=await pipe.get(key), run_id=run_id):
                    return False
                if not _held_by(raw=await pipe.hget(CLAIMS_KEY, conversation_id), run_id=run_id):
                    return False
                pipe.multi()
                pipe.hdel(CLAIMS_KEY, conversation_id)
                await pipe.execute()
                return True
        except WatchError:
            continue
    return False
