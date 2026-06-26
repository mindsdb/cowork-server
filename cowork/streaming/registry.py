"""Process-global registry of in-flight turns (in-process backend).

One ``RunHandle`` per ``conversation_id`` — owns the detached producer
``asyncio.Task`` and the ``StreamBuffer`` it writes to. Lookup is by
``conversation_id`` (a conversation has at most one in-flight turn).

This is the **in-process** dispatch model: the run executes as a task in
this server process, decoupled from the HTTP request that started it
(closing the request never cancels the task — only an explicit
``/cancel`` does). Good for desktop + the single-instance cloud
container.

WIP — multi-instance cloud: the run moves to a separate worker pool fed
by a queue (SQS / Redis), and "the registry" becomes a shared run-status
store (Redis HSET) + a cancel channel (Redis PUBLISH). The web tier then
only enqueues + tails the shared buffer. The endpoint contract stays the
same; only this dispatch layer is swapped. See buffer.RedisStreamBuffer.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from cowork.streaming.buffer import StreamBuffer

logger = logging.getLogger(__name__)


@dataclass
class RunHandle:
    """One in-flight (or recently-finished) turn. Kept after the task
    completes so a returning client can still tail to the terminal
    record; GC sweeps stale handles after a grace period."""

    conversation_id: str
    turn_id: int
    buffer: StreamBuffer
    task: asyncio.Task
    created_at_monotonic: float = field(default_factory=lambda: 0.0)

    @property
    def is_running(self) -> bool:
        return not self.task.done()

    async def cancel(self) -> bool:
        """Request cancellation of the producer task. Returns True if a
        cancel was issued (task still running), False if already done."""
        if self.task.done():
            return False
        self.task.cancel()

        try:
            await self.task
        except asyncio.CancelledError:
            return True
        except Exception:
            return False


class RunRegistry:
    """Process-wide map of in-flight turns. Single-threaded (asyncio loop)."""

    def __init__(self) -> None:
        self._by_cid: dict[str, RunHandle] = {}
        self._lock = asyncio.Lock()

    async def start(
        self,
        *,
        conversation_id: str,
        turn_id: int,
        buffer: StreamBuffer,
        producer_coro,
    ) -> RunHandle:
        """Spawn the producer as a detached task and register it. A
        duplicate start for an already-in-flight conversation returns the
        existing handle (the renderer's queue should prevent dupes)."""
        loop = asyncio.get_running_loop()
        async with self._lock:
            existing = self._by_cid.get(conversation_id)
            if existing is not None and existing.is_running:
                logger.info(
                    "Duplicate turn start for conversation %s; returning existing handle (turn %d).",
                    conversation_id, existing.turn_id,
                )
                return existing
            task = asyncio.create_task(producer_coro, name=f"turn[{conversation_id}/{turn_id}]")
            handle = RunHandle(
                conversation_id=conversation_id,
                turn_id=turn_id,
                buffer=buffer,
                task=task,
                created_at_monotonic=loop.time(),
            )
            self._by_cid[conversation_id] = handle
            return handle

    def get(self, conversation_id: str) -> Optional[RunHandle]:
        """Current handle (incl. recently-finished, useful for replay)."""
        return self._by_cid.get(conversation_id)

    async def cancel(self, conversation_id: str) -> bool:
        handle = self._by_cid.get(conversation_id)
        if handle is None:
            return False
        return await handle.cancel()

    def in_flight(self) -> list[RunHandle]:
        return [h for h in self._by_cid.values() if h.is_running]

    async def gc_finished(self, max_age_seconds: float = 300.0) -> int:
        """Drop handles whose producer finished > max_age ago. The buffer
        file stays on disk — only the in-memory handle is freed."""
        loop = asyncio.get_running_loop()
        now = loop.time()
        async with self._lock:
            stale = [
                cid for cid, h in self._by_cid.items()
                if not h.is_running and now - h.created_at_monotonic > max_age_seconds
            ]
            for cid in stale:
                self._by_cid.pop(cid, None)
        return len(stale)


# Single global instance per server process.
registry: RunRegistry = RunRegistry()
