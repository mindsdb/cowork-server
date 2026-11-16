"""Who is watching a turn live: the signal that holds back its push.

A person reading a turn's stream (the POST that started it, or a reconnect
through GET /responses/tail) sees it finish, so a "task finished" push to
their phone would only repeat it. ``sse_from_buffer`` holds a ``watch`` for
as long as it streams a turn, and ``is_watched`` answers whether one was held
when the turn ended.

A reader that streamed the turn to its end stays counted for
``WATCH_GRACE_SECONDS`` after it stops: the reader sees the terminal record
and leaves at the same moment the producer finishes, and the push decision
comes after both. A reader that left before the end (a closed tab, a phone
that went to sleep) stops counting at once.

On the Redis stream backend a reconnect can land on any replica, so each
watch is also a member of a sorted set keyed by the turn, scored by when it
lapses: renewed while the reader streams and given its grace when it ends.
A replica that dies mid-stream leaves a member that lapses on its own within
``WATCH_LEASE_SECONDS``. Redis failures only ever make a turn read as
unwatched, so a broken signal costs a redundant push, never a missing one.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from uuid import uuid4

logger = logging.getLogger(__name__)

WATCH_GRACE_SECONDS = 15.0
WATCH_LEASE_SECONDS = 60.0
_RENEW_SECONDS = 20.0
_REDIS_TIMEOUT_SECONDS = 2.0
_KEY_TTL_SECONDS = 3600


@dataclass
class _TurnWatch:
    active: int = 0
    # time.monotonic() until which a reader that saw the end still counts.
    grace_until: float = 0.0


@dataclass
class Watch:
    """One reader's hold on a turn. ``saw_end`` is set by the reader when
    it streamed the terminal record."""

    conversation_id: str
    turn_id: int
    member: str = field(default_factory=lambda: uuid4().hex)
    saw_end: bool = False


_local: dict[tuple[str, int], _TurnWatch] = {}


def _prune() -> None:
    """Drop the turns no one watches any more, so the map holds only turns
    being read or just finished."""
    now = time.monotonic()
    for turn, entry in list(_local.items()):
        if entry.active <= 0 and entry.grace_until <= now:
            del _local[turn]


def _key(conversation_id: str, turn_id: int) -> str:
    return f"cowork:turn_watchers:{conversation_id}:{turn_id}"


def _redis_backend() -> bool:
    from cowork.streaming.backend import get_backend

    return get_backend() == "redis"


async def _redis_mark(watch: Watch, *, lapses_in: float) -> None:
    from cowork.turnqueue.redis_client import get_redis

    key = _key(watch.conversation_id, watch.turn_id)
    try:
        async with asyncio.timeout(_REDIS_TIMEOUT_SECONDS):
            async with get_redis().pipeline(transaction=False) as pipe:
                pipe.zadd(key, {watch.member: time.time() + lapses_in})
                pipe.expire(key, _KEY_TTL_SECONDS)
                await pipe.execute()
    except Exception:
        logger.debug("Could not record a turn watcher", exc_info=True)


async def _redis_unmark(watch: Watch) -> None:
    from cowork.turnqueue.redis_client import get_redis

    try:
        async with asyncio.timeout(_REDIS_TIMEOUT_SECONDS):
            await get_redis().zrem(_key(watch.conversation_id, watch.turn_id), watch.member)
    except Exception:
        logger.debug("Could not remove a turn watcher", exc_info=True)


async def _renew(watch: Watch) -> None:
    while True:
        await asyncio.sleep(_RENEW_SECONDS)
        await _redis_mark(watch, lapses_in=WATCH_LEASE_SECONDS)


@asynccontextmanager
async def watch(conversation_id: str, turn_id: int) -> AsyncIterator[Watch]:
    """Count the caller as watching the turn while the block runs. Never
    raises on its own account."""
    held = Watch(conversation_id=conversation_id, turn_id=turn_id)
    entry = _local.setdefault((conversation_id, turn_id), _TurnWatch())
    entry.active += 1
    redis = _redis_backend()
    renew: asyncio.Task | None = None
    if redis:
        await _redis_mark(held, lapses_in=WATCH_LEASE_SECONDS)
        renew = asyncio.create_task(_renew(held))
    try:
        yield held
    finally:
        entry.active -= 1
        if held.saw_end:
            entry.grace_until = max(entry.grace_until, time.monotonic() + WATCH_GRACE_SECONDS)
        _prune()
        if renew is not None:
            renew.cancel()
            await asyncio.gather(renew, return_exceptions=True)
            # Shielded: a second cancel of the reader (a disconnect) must not
            # leave the member half settled.
            await asyncio.shield(
                _redis_mark(held, lapses_in=WATCH_GRACE_SECONDS) if held.saw_end else _redis_unmark(held)
            )


async def is_watched(conversation_id: str, turn_id: int) -> bool:
    """Whether someone was reading the turn when it ended (see module doc)."""
    entry = _local.get((conversation_id, turn_id))
    if entry is not None:
        if entry.active > 0 or entry.grace_until > time.monotonic():
            return True
    if not _redis_backend():
        return False
    from cowork.turnqueue.redis_client import get_redis

    try:
        async with asyncio.timeout(_REDIS_TIMEOUT_SECONDS):
            count = await get_redis().zcount(_key(conversation_id, turn_id), time.time(), "+inf")
    except Exception:
        logger.debug("Could not read turn watchers", exc_info=True)
        return False
    return bool(count)


def reset() -> None:
    """Forget every local watch (tests)."""
    _local.clear()
