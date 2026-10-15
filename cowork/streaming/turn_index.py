"""Which turn is current for each conversation, readable from any replica.

A lookup table, not a lock. Nothing here decides who may run a turn:
scratchpad-controller already serialises execution per conversation, and
whether a turn is still going is answered by its buffer, whose terminal record
any replica can read.

This exists so a replica that did not start a turn can still find its turn_id,
to open the right buffer, its correlation_id, to cancel it, and its org, to
authorize the caller.

Keys:
    cowork:turn:{conversation_id}   HASH turn_id, correlation_id, org_id, user_id, started_at
    cowork:turns                    SET of conversation ids with a recorded turn
"""
from __future__ import annotations

import logging
import time

from redis.asyncio import Redis
from redis.exceptions import WatchError

from cowork.turnqueue.redis_client import get_redis, get_sync_redis

logger = logging.getLogger(__name__)

# Outlives the buffer TTL, so a conversation whose buffer expired reads as
# "no current turn" rather than as a turn with a missing buffer.
TURN_INDEX_TTL_SECONDS = 3600

_TURNS_SET = "cowork:turns"


def _turn_key(conversation_id: str) -> str:
    return f"cowork:turn:{conversation_id}"


async def record_turn(
    conversation_id: str,
    *,
    turn_id: int,
    correlation_id: str,
    org_id: str | None,
    user_id: str | None,
    client=None,
) -> None:
    """Note this turn as the conversation's current one, replacing any previous.

    ``client`` lets a caller that already holds a Redis client reuse it, rather
    than this module reaching for a second one.
    """
    r = client or get_redis()
    key = _turn_key(conversation_id)
    # One MULTI, so no reader finds the hash empty between its DELETE and its
    # HSET, which list_turns would read as an expired turn.
    async with r.pipeline(transaction=True) as pipe:
        pipe.delete(key)
        pipe.hset(key, mapping={
            "turn_id": str(turn_id),
            "correlation_id": correlation_id,
            "org_id": org_id or "",
            "user_id": user_id or "",
            "started_at": str(time.time()),
        })
        pipe.expire(key, TURN_INDEX_TTL_SECONDS)
        pipe.sadd(_TURNS_SET, conversation_id)
        await pipe.execute()


async def get_turn(conversation_id: str) -> dict | None:
    turn = await get_redis().hgetall(_turn_key(conversation_id))
    return dict(turn) if turn else None


async def forget_turn(conversation_id: str) -> None:
    r = get_redis()
    await r.delete(_turn_key(conversation_id))
    await r.srem(_TURNS_SET, conversation_id)


def forget_turn_sync(conversation_id: str) -> None:
    """Blocking ``forget_turn``, for callers with no event loop.

    Conversation delete runs in a threadpool thread (the endpoint is a sync
    ``def``), and leaving the entry behind would have /in-flight keep naming a
    turn whose buffers were just deleted.
    """
    try:
        r = get_sync_redis()
        r.delete(_turn_key(conversation_id))
        r.srem(_TURNS_SET, conversation_id)
    except Exception:
        logger.warning("Could not forget the turn index entry for %s", conversation_id, exc_info=True)


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
            pipe.hgetall(_turn_key(conversation_id))
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
    keys = [_turn_key(conversation_id) for conversation_id in conversation_ids]
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
