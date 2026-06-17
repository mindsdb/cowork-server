"""In-memory per-conversation turn buffers for stream reconnect.

The responses handler appends every formatted SSE event here while a
turn runs — for streaming turns AND non-streaming ones (scheduled runs
execute with ``stream=False``). ``GET /responses/tail`` replays a
buffer from a client-supplied sequence number and then follows the live
turn; ``GET /responses/in-flight`` reports ``has_buffer``/``latest_seq``
so the client can decide whether opening a tail is worthwhile.

This is the lean port of the legacy bundled server's file-backed
stream registry: the server is single-process, and buffers only need to
outlive the reconnect race (client attaches moments after the producer
finished), not a restart. Finished buffers are pruned after a TTL; a
new turn on the same conversation replaces the previous buffer.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

# Keep finished buffers long enough for "the producer wrote Done 50ms
# ago but the client still wants the replay" — and for the Task view to
# attach shortly after a scheduled run completes.
FINISHED_TTL_SECONDS = 600.0


class TurnBuffer:
    """Append-only event log for one turn, with async followers.

    ``append``/``finish`` are synchronous — they're called from the
    event loop thread (inside the handler's formatter loop), so a plain
    list append is safe; followers are woken via per-follower events to
    avoid shared-Event clear races between concurrent tails.
    """

    def __init__(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id
        self.events: list[dict] = []
        self.done = False
        self.finished_at: float | None = None
        self._waiters: set[asyncio.Event] = set()

    @property
    def latest_seq(self) -> int:
        return len(self.events)

    def _wake_followers(self) -> None:
        for waiter in list(self._waiters):
            waiter.set()

    def append(self, data: dict) -> None:
        self.events.append(data)
        self._wake_followers()

    def finish(self) -> None:
        self.done = True
        self.finished_at = time.monotonic()
        self._wake_followers()

    async def follow(self, from_seq: int = 0) -> AsyncIterator[dict]:
        """Yield events starting at ``from_seq``, then live ones until
        the turn finishes. Safe for multiple concurrent followers."""
        seq = max(0, from_seq)
        while True:
            while seq < len(self.events):
                event = self.events[seq]
                seq += 1
                yield event
            if self.done:
                return
            waiter = asyncio.Event()
            self._waiters.add(waiter)
            try:
                # Re-check after registering so an append/finish that
                # landed in between can't be missed.
                if seq < len(self.events) or self.done:
                    continue
                await waiter.wait()
            finally:
                self._waiters.discard(waiter)


_buffers: dict[str, TurnBuffer] = {}


def _prune() -> None:
    now = time.monotonic()
    for cid, buffer in list(_buffers.items()):
        if buffer.done and buffer.finished_at is not None and now - buffer.finished_at > FINISHED_TTL_SECONDS:
            del _buffers[cid]


def begin_turn(conversation_id: str) -> TurnBuffer:
    """Create (and register) the buffer for a new turn. Replaces any
    previous turn's buffer for the conversation."""
    _prune()
    buffer = TurnBuffer(str(conversation_id))
    _buffers[str(conversation_id)] = buffer
    return buffer


def ensure_buffer(conversation_id: str) -> TurnBuffer:
    """Return the existing buffer for *conversation_id*, or create one.

    Use this when the buffer may have been pre-created (e.g. by
    ``run_schedule_now``) so that early ``/tail`` followers are not
    orphaned by a replacement."""
    cid = str(conversation_id)
    existing = _buffers.get(cid)
    if existing is not None and not existing.done:
        return existing
    return begin_turn(cid)


def get_buffer(conversation_id: str) -> TurnBuffer | None:
    _prune()
    return _buffers.get(str(conversation_id))
