"""When a cloud memory tier is due for compaction, shared by every replica.

A cloud turn builds a new anton session, so the desktop's "every 10 turns"
counter never gets past one. Instead a tier is due once it holds more than
anton's threshold of rules + lessons and has grown by `_GROWTH_BEFORE_NEXT`
since the last attempt, which Redis remembers.

Keys:
    cowork:memory-compaction:count:{org}:{kind}:{owner}   entry count after the last finished attempt
    cowork:memory-compaction:lock:{org}:{kind}:{owner}    a token while one replica compacts the tier
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from anton.core.memory.cortex import COMPACTION_THRESHOLD
from redis.exceptions import WatchError

from cowork.turnqueue.redis_client import get_redis

logger = logging.getLogger(__name__)

_GROWTH_BEFORE_NEXT = 10

# Long enough to outlive a quiet spell, short enough that a tier nobody
# writes any more does not keep its key forever.
_COUNT_TTL_SECONDS = 90 * 24 * 3600

# The whole compaction (key mint and model calls) must end before the turn
# key it runs on expires; the lock outlives it, so it never frees mid-run.
COMPACTION_DEADLINE_SECONDS = 600
_LOCK_TTL_SECONDS = COMPACTION_DEADLINE_SECONDS + 300


@dataclass(frozen=True)
class MemoryTier:
    """One compactable tier: a person's own memory, or a project's."""

    org_id: str | None
    kind: Literal["person", "project"]
    owner_id: str

    def _key(self, what: str) -> str:
        return f"cowork:memory-compaction:{what}:{self.org_id or '-'}:{self.kind}:{self.owner_id}"


async def is_due(tier: MemoryTier, entry_count: int) -> bool:
    """Whether `tier`, now holding `entry_count` rules + lessons, should be compacted."""
    client = get_redis()
    key = tier._key("count")
    stored = await client.get(key)
    last = int(stored) if stored is not None else 0
    if last > entry_count:
        # Pruned by hand since the last attempt: count growth from here.
        await client.set(key, entry_count, ex=_COUNT_TTL_SECONDS)
        last = entry_count
    return entry_count > COMPACTION_THRESHOLD and entry_count >= last + _GROWTH_BEFORE_NEXT


async def record_attempt(tier: MemoryTier, entry_count: int) -> None:
    """Remember the count after an attempt, succeeded or failed, so a tier that
    keeps failing costs one model call per `_GROWTH_BEFORE_NEXT` new entries."""
    await get_redis().set(tier._key("count"), entry_count, ex=_COUNT_TTL_SECONDS)


@asynccontextmanager
async def tier_lock(tier: MemoryTier) -> AsyncIterator[bool]:
    """Yield whether this caller holds the tier's compaction lock.

    False means another replica is compacting it now. The release deletes the
    key only while it still holds this attempt's token, so an attempt that
    outran the TTL cannot free the lock of the next one.
    """
    client = get_redis()
    key = tier._key("lock")
    token = uuid4().hex
    acquired = bool(await client.set(key, token, nx=True, ex=_LOCK_TTL_SECONDS))
    try:
        yield acquired
    finally:
        if acquired:
            await _release(key, token)


async def _release(key: str, token: str) -> None:
    # WATCH/MULTI rather than a Lua compare-and-delete: fakeredis, which the
    # tests run on, has no EVAL. Never raises: it runs in a `finally`.
    try:
        async with get_redis().pipeline(transaction=True) as pipe:
            await pipe.watch(key)
            if await pipe.get(key) != token:
                await pipe.reset()
                return
            pipe.multi()
            pipe.delete(key)
            await pipe.execute()
    except WatchError:
        return
    except Exception as exc:
        # The TTL frees the lock anyway.
        logger.warning("memory-compaction: lock release failed (%s)", type(exc).__name__)
