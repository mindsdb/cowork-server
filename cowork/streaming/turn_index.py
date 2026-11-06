"""Which turn is current for each conversation, readable from any replica.

A lookup table, not a lock. Nothing here decides who may run a turn: the
conversation's turn claim does (turn_claim), and RunRegistry.start writes a
run's entry only once that run holds the claim. Whether a turn is still going
is answered by its buffer, whose terminal record any replica can read.

This exists so a replica that did not start a turn can still find its turn_id,
to open the right buffer, its correlation_id, to cancel it, and its org, to
authorize the caller. ``saved_question`` and ``question_id`` say the turn saved
its question, and which row. The run stamps them as soon as it saves the
question (registry.mark_question_saved) and again before it releases its
claim. RunRegistry.start and turn_claim.acquire read the stamp to refuse a
start whose history read predates that save (TurnIndexEntry.makes_stale).

Keys:
    cowork:turn:{conversation_id}   HASH turn_id, run_id, correlation_id, org_id, user_id, started_at,
                                         saved_question, question_id
    cowork:turns                    SET of conversation ids with a recorded turn
"""
from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from uuid import UUID

from pydantic import BaseModel, ConfigDict
from redis.asyncio import Redis
from redis.asyncio.client import Pipeline
from redis.exceptions import WatchError

from cowork.streaming.turn_claim import claim_key
from cowork.turnqueue.redis_client import get_redis, get_sync_redis

logger = logging.getLogger(__name__)

# Outlives the buffer TTL, so a conversation whose buffer expired reads as
# "no current turn" rather than as a turn with a missing buffer.
TURN_INDEX_TTL_SECONDS = 3600

_TURNS_SET = "cowork:turns"


def turn_key(conversation_id: str) -> str:
    return f"cowork:turn:{conversation_id}"


class TurnOwner(BaseModel):
    """Who started a recorded turn: its org (a person's partition on a
    trusted-identity deployment) and its user, None where the turn had none.

    Built from a turn's hash in one place (``from_index``), so the owner check
    never reads the hash's keys itself.
    """

    model_config = ConfigDict(frozen=True)

    org_id: str | None
    user_id: str | None

    @classmethod
    def from_index(cls, turn: Mapping[str, str]) -> TurnOwner:
        # record_turn writes "" for a missing id.
        return cls(org_id=turn.get("org_id") or None, user_id=turn.get("user_id") or None)


class TurnIndexEntry(BaseModel):
    """What admission reads from a turn's hash. A channel turn's entry has
    no ``run_id``: no registry handle covers it."""

    model_config = ConfigDict(frozen=True)

    turn_id: int
    run_id: str | None
    saved_question: bool
    question_id: UUID | None

    @classmethod
    def from_index(cls, turn: Mapping[str, str]) -> TurnIndexEntry:
        return cls(
            turn_id=int(turn["turn_id"]),
            run_id=turn.get("run_id") or None,
            saved_question=turn.get("saved_question") == "1",
            question_id=turn.get("question_id") or None,
        )

    def makes_stale(self, *, turn_id: int) -> bool:
        """Whether a start at ``turn_id`` read the history before this
        entry's run saved its question: the run saved it at this entry's
        number, and the start names that number or an earlier one. A channel
        turn's entry names no run, so it never does."""
        return self.run_id is not None and self.saved_question and turn_id <= self.turn_id


def _queue_record(
    *,
    pipe: Pipeline,
    conversation_id: str,
    turn_id: int,
    correlation_id: str,
    org_id: str | None,
    user_id: str | None,
    run_id: str | None,
) -> None:
    """Queue the writes that replace a conversation's entry on a MULTI.

    One MULTI, so no reader finds the hash empty between its DELETE and its
    HSET, which list_turns would read as an expired turn."""
    key = turn_key(conversation_id)
    pipe.delete(key)
    pipe.hset(key, mapping={
        "turn_id": str(turn_id),
        "correlation_id": correlation_id,
        "org_id": org_id or "",
        "user_id": user_id or "",
        "started_at": str(time.time()),
        **({"run_id": run_id} if run_id is not None else {}),
    })
    pipe.expire(key, TURN_INDEX_TTL_SECONDS)
    pipe.sadd(_TURNS_SET, conversation_id)


async def record_turn(
    conversation_id: str,
    *,
    turn_id: int,
    correlation_id: str,
    org_id: str | None,
    user_id: str | None,
    run_id: str | None = None,
    client=None,
) -> None:
    """Note this turn as the conversation's current one, replacing any previous.

    For a run that holds the conversation's turn claim (RunRegistry.start). A
    turn no registry handle covers uses record_channel_turn.

    ``client`` lets a caller that already holds a Redis client reuse it, rather
    than this module reaching for a second one.
    """
    r = client or get_redis()
    async with r.pipeline(transaction=True) as pipe:
        _queue_record(
            pipe=pipe, conversation_id=conversation_id, turn_id=turn_id, correlation_id=correlation_id,
            org_id=org_id, user_id=user_id, run_id=run_id,
        )
        await pipe.execute()


async def record_channel_turn(
    *,
    conversation_id: str,
    turn_id: int,
    correlation_id: str,
    org_id: str | None,
    user_id: str | None,
    client=None,
) -> bool:
    """record_turn for a turn no registry handle covers (a channel turn),
    unless a run holds the conversation's turn claim. That run's entry is
    what cancel and answer resolve through, so it is left as it is. True when
    it wrote the entry."""
    r = client or get_redis()
    key = turn_key(conversation_id)
    claim = claim_key(conversation_id)
    for _attempt in range(3):
        try:
            async with r.pipeline(transaction=True) as pipe:
                await pipe.watch(claim, key)
                if await pipe.exists(claim):
                    return False
                pipe.multi()
                _queue_record(
                    pipe=pipe, conversation_id=conversation_id, turn_id=turn_id, correlation_id=correlation_id,
                    org_id=org_id, user_id=user_id, run_id=None,
                )
                await pipe.execute()
                return True
        except WatchError:
            continue
    raise WatchError(f"the turn index entry of {conversation_id} kept changing while it was recorded")


async def stamp_question(
    *, conversation_id: str, run_id: str, question_id: UUID | None, client=None,
) -> bool:
    """Note in ``run_id``'s entry that its turn saved its question, and which
    row when known. Leaves another run's entry alone. True when it wrote."""
    r = client or get_redis()
    key = turn_key(conversation_id)
    fields = {"saved_question": "1", **({"question_id": str(question_id)} if question_id is not None else {})}
    for _attempt in range(3):
        try:
            async with r.pipeline(transaction=True) as pipe:
                await pipe.watch(key)
                if await pipe.hget(key, "run_id") != run_id:
                    return False
                pipe.multi()
                pipe.hset(key, mapping=fields)
                await pipe.execute()
                return True
        except WatchError:
            continue
    raise WatchError(f"the turn index entry of {conversation_id} kept changing while it was stamped")


async def get_turn(conversation_id: str) -> dict | None:
    turn = await get_redis().hgetall(turn_key(conversation_id))
    return dict(turn) if turn else None


async def forget_turn(conversation_id: str) -> None:
    r = get_redis()
    await r.delete(turn_key(conversation_id))
    await r.srem(_TURNS_SET, conversation_id)


async def forget_run(conversation_id: str, run_id: str) -> None:
    """Remove abandoned admission without deleting a successor's index entry."""
    r = get_redis()
    key = turn_key(conversation_id)
    try:
        async with r.pipeline(transaction=True) as pipe:
            await pipe.watch(key)
            if await pipe.hget(key, "run_id") != run_id:
                return
            pipe.multi()
            pipe.delete(key)
            pipe.srem(_TURNS_SET, conversation_id)
            await pipe.execute()
    except WatchError:
        return  # replaced while cleanup was pending; that entry belongs to its successor


def forget_turn_sync(conversation_id: str) -> None:
    """Blocking ``forget_turn``, for callers with no event loop.

    Conversation delete runs in a threadpool thread (the endpoint is a sync
    ``def``), and leaving the entry behind would have /in-flight keep naming a
    turn whose buffers were just deleted.
    """
    try:
        r = get_sync_redis()
        r.delete(turn_key(conversation_id))
        r.srem(_TURNS_SET, conversation_id)
    except Exception:
        # The entry stays until its TTL. One its turn stamped with a saved
        # question refuses sends at or below its turn number until then
        # (TurnIndexEntry.makes_stale); deleting the key clears it.
        logger.warning(
            "Could not forget a deleted conversation's turn index entry.",
            exc_info=True,
            extra={"event": "turn_index_forget_failed", "conversation_id": conversation_id},
        )


async def list_turns() -> list[dict]:
    """Every recorded turn, pruning set members whose hash has expired.

    Every member's hash is read in one pipelined round trip rather than one
    round trip each."""
    r = get_redis()
    # Materialized once: the pipeline's replies pair with members by position.
    members = list(await r.smembers(_TURNS_SET))
    if not members:
        return []
    async with r.pipeline(transaction=False) as pipe:
        for conversation_id in members:
            pipe.hgetall(turn_key(conversation_id))
        turns = await pipe.execute()
    out: list[dict] = []
    expired: list[str] = []
    for conversation_id, turn in zip(members, turns, strict=True):
        if not turn:
            expired.append(conversation_id)
            continue
        out.append({"conversation_id": conversation_id, **turn})
    if expired:
        await _prune(r, expired)
    return out


async def _prune(r: Redis, conversation_ids: list[str]) -> None:
    """Remove these members from the set if each one's hash is still gone.

    A turn can be recorded after list_turns read its conversation's hash as
    expired. Its SADD is then a no-op, so a plain SREM landing after it would
    hide a running turn from /in-flight-list until the conversation's next
    turn. So the SREM runs only while every hash is still missing, under a
    WATCH that aborts it if one is written before it lands. A prune that does
    not run leaves its members to the next list."""
    keys = [turn_key(conversation_id) for conversation_id in conversation_ids]
    try:
        async with r.pipeline(transaction=True) as pipe:
            await pipe.watch(*keys)
            if await pipe.exists(*keys):
                await pipe.reset()
                return
            pipe.multi()
            pipe.srem(_TURNS_SET, *conversation_ids)
            await pipe.execute()
    except WatchError:
        return
