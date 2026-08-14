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

from cowork.turnqueue.redis_client import get_redis

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
    await r.delete(key)
    await r.hset(key, mapping={
        "turn_id": str(turn_id),
        "correlation_id": correlation_id,
        "org_id": org_id or "",
        "user_id": user_id or "",
        "started_at": str(time.time()),
    })
    await r.expire(key, TURN_INDEX_TTL_SECONDS)
    await r.sadd(_TURNS_SET, conversation_id)


async def get_turn(conversation_id: str) -> dict | None:
    turn = await get_redis().hgetall(_turn_key(conversation_id))
    return dict(turn) if turn else None


async def forget_turn(conversation_id: str) -> None:
    r = get_redis()
    await r.delete(_turn_key(conversation_id))
    await r.srem(_TURNS_SET, conversation_id)


async def list_turns() -> list[dict]:
    """Every recorded turn, pruning set members whose hash has expired."""
    r = get_redis()
    out: list[dict] = []
    for conversation_id in await r.smembers(_TURNS_SET):
        turn = await r.hgetall(_turn_key(conversation_id))
        if not turn:
            await r.srem(_TURNS_SET, conversation_id)
            continue
        out.append({"conversation_id": conversation_id, **turn})
    return out
