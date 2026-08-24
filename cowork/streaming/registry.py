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
import os
from dataclasses import dataclass, field
from typing import Optional

from cowork.streaming.buffer import StreamBuffer

logger = logging.getLogger(__name__)

# Reap a turn only after it makes NO progress (writes no buffer record) for this
# long. A hung producer otherwise runs forever, holding the client's shared
# stream slot so sends wedge in every conversation; boot recovery only covers a
# real restart, not a re-adopted crash-orphan sidecar, so the bound is enforced
# at runtime.
#
# IDLE, not total duration: `buffer.latest_seq` advances on every frame, so an
# actively-streaming turn — or one blocked on an `ask_user` card (up to 300s,
# within this window) — resets the window and is never reaped. A total-duration
# cap would kill a long deliberate turn mid-conversation.
#
# COWORK_MAX_TURN_IDLE_SECONDS widens the window for deployments with long silent
# tool calls; a non-positive or unparseable value falls back to the default.
def _idle_bound_seconds() -> int:
    raw = os.environ.get("COWORK_MAX_TURN_IDLE_SECONDS")
    if raw is None:
        return 600
    try:
        val = int(raw)
    except ValueError:
        logger.warning("COWORK_MAX_TURN_IDLE_SECONDS=%r is not an int; using default 600s", raw)
        return 600
    if val <= 0:
        logger.warning("COWORK_MAX_TURN_IDLE_SECONDS=%d is not positive; using default 600s", val)
        return 600
    return val


_MAX_TURN_IDLE_SECONDS = _idle_bound_seconds()

# How often the idle watchdog samples buffer progress. Detection latency for a
# fully wedged producer is at most _MAX_TURN_IDLE_SECONDS + this.
_IDLE_POLL_SECONDS = 15


@dataclass
class TurnLifecycle:
    """One bit of shared state between a producer coroutine and its handle.

    Exists because of an ordering problem: ``registry.start()`` is handed an
    already-constructed producer coroutine, so the coroutine cannot reach the
    ``RunHandle`` that will own it. The caller creates one of these, passes it
    into the producer AND into ``start()``, and both sides then look at the
    same object — no registry lookup, which would be ambiguous (a discarded
    handle and a handle replaced by the next turn are both "not the one I
    registered", and ``reset()`` empties the map too).

    ``discarded`` means "the conversation this turn belongs to was truncated
    under it". Set from ``RunRegistry.discard``, which runs in a threadpool
    thread (the delete endpoint is a sync ``def``); a plain attribute write is
    safe there, unlike ``Task.cancel()``.
    """

    discarded: bool = False


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
    # Owning org, captured from the trusted scope that started the turn (None
    # in local mode). The tail/cancel/in-flight endpoints check it so one org
    # can't tail or cancel another org's in-flight turn; a conversation_id is
    # not an authorization token.
    org_id: str | None = None
    # Owner of the turn AND an authorization gate: conversations are personal,
    # so the tail/cancel/in-flight endpoints require BOTH org_id and user_id to
    # match (_owner_matches). Matching org alone previously let any member tail
    # or cancel a teammate's live turn and read its prompts/output (audit P0).
    user_id: str | None = None
    # Shared with the producer coroutine, see TurnLifecycle.
    lifecycle: TurnLifecycle = field(default_factory=TurnLifecycle)

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
        # The task finished without surfacing the cancellation — it either
        # absorbed it (every producer catches CancelledError to persist and
        # close its buffer) or was already on its last step. A cancel was
        # still issued, which is what this returns; falling off the end here
        # returned None from a `-> bool`.
        return True


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
        org_id: str | None = None,
        user_id: str | None = None,
        lifecycle: TurnLifecycle | None = None,
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
            task = asyncio.create_task(
                self._run_bounded(producer_coro, buffer, conversation_id, turn_id),
                name=f"turn[{conversation_id}/{turn_id}]",
            )
            handle = RunHandle(
                conversation_id=conversation_id,
                turn_id=turn_id,
                buffer=buffer,
                task=task,
                created_at_monotonic=loop.time(),
                org_id=org_id,
                user_id=user_id,
                # The SAME object the producer coroutine closed over, so
                # discard() can tell it to drop the turn on the floor.
                lifecycle=lifecycle if lifecycle is not None else TurnLifecycle(),
            )
            self._by_cid[conversation_id] = handle
            return handle

    async def _run_bounded(
        self, producer_coro, buffer: StreamBuffer, conversation_id: str, turn_id: int,
    ) -> None:
        """Run a producer under the idle bound (see module comment).

        On reap the producer is cancelled; its CancelledError handler seals the
        buffer with a terminal record (the user-Stop path), so the tail ends and
        the client releases its slot. An external cancel/discard cancels this
        wrapper and ``await task`` forwards it into the producer, so
        ``RunHandle.cancel``/``discard`` still work; a producer that raises on
        its own propagates unchanged.
        """
        loop = asyncio.get_running_loop()
        task = asyncio.ensure_future(producer_coro)
        reaped = False

        async def _watchdog() -> None:
            nonlocal reaped
            # The watchdog is a safety net, never a source of failure: an
            # unexpected error here must not touch the turn, so it fails open
            # (turn runs unbounded) and is logged rather than propagated.
            try:
                last_seq = buffer.latest_seq
                last_progress = loop.time()
                while True:
                    await asyncio.sleep(_IDLE_POLL_SECONDS)
                    seq = buffer.latest_seq
                    if seq != last_seq:
                        last_seq, last_progress = seq, loop.time()
                    elif loop.time() - last_progress >= _MAX_TURN_IDLE_SECONDS:
                        reaped = True
                        logger.warning(
                            "Turn for conversation %s (turn %d) made no progress for %ss; "
                            "producer cancelled and its buffer sealed.",
                            conversation_id, turn_id, _MAX_TURN_IDLE_SECONDS,
                        )
                        task.cancel()
                        return
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Idle watchdog for conversation %s (turn %d) failed; "
                    "turn runs unbounded.", conversation_id, turn_id,
                )

        watchdog = asyncio.ensure_future(_watchdog())
        try:
            await task
        except asyncio.CancelledError:
            # A reap surfaces as the producer's cancellation. Unlike an external
            # cancel (which must propagate for RunHandle.cancel to observe), a
            # reap is already terminal — its buffer is sealed — so swallow it.
            if not reaped:
                raise
        finally:
            # Reap the watchdog without propagating its outcome — it must never
            # overwrite the turn's own result (including a cancellation
            # RunHandle.cancel is waiting to observe).
            watchdog.cancel()
            await asyncio.gather(watchdog, return_exceptions=True)

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

    def discard(self, conversation_id: str) -> None:
        """Drop a conversation's handle AND stop its producer (sync, best-effort).

        Called after a turn delete: the buffered turn is stale (turn_id ==
        message count, so truncation makes the next turn reuse it), and the
        handle must not be tailed or reused. The dict pop is atomic; the async
        lock is skipped because a delete never races a *start* for the same id.

        A delete CAN race a running turn, though — since ask_user a turn may
        sit blocked on a question for up to 300 s, looking idle, which is
        exactly when a user edits or deletes an earlier message. Popping alone
        left that producer running against truncated history: its answer
        endpoint 404s (it gates on a registered handle) with no retry that can
        ever succeed, and it blocks to the full timeout before resuming. So
        flag the turn as discarded and cancel it.

        Order matters: ``discarded`` is set BEFORE the cancel is scheduled, so
        the producer's ``except CancelledError`` handler can never observe
        ``False`` and persist rows into a history that no longer has room for
        them.

        Callers reach this from a threadpool thread (the delete endpoint is a
        sync ``def``, so there is no running loop here) — hence
        ``call_soon_threadsafe`` on the task's own loop rather than a bare
        ``cancel()``, which is not thread-safe.
        """
        handle = self._by_cid.pop(conversation_id, None)
        if handle is None:
            return
        handle.lifecycle.discarded = True
        if handle.task.done():
            return
        try:
            handle.task.get_loop().call_soon_threadsafe(handle.task.cancel)
        except RuntimeError:
            # Loop already closed (shutdown, or a test loop that outlived its
            # task): nothing is going to run this producer again anyway.
            logger.warning(
                "Could not cancel the discarded producer for conversation %s",
                conversation_id, exc_info=True,
            )

    def reset(self) -> None:
        """Forget every handle without touching the producer tasks.

        For tests: this object is a process global, so a handle registered by
        one test stays visible to the next. Deliberately does NOT cancel the
        tasks — the caller owns those, and cancelling here would make a
        cleanup helper a scheduling side effect.
        """
        self._by_cid.clear()

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


def discard_conversation(conversation_id: str) -> None:
    """Drop a conversation's in-memory handle AND its on-disk buffers.

    After a turn delete the buffered turns are stale — `turn_id == message count`
    is reused once the history is truncated, so a stale buffer would be tailed,
    reused, or replayed on the next send. Streaming owns these files; callers just
    ask for the conversation to be discarded.

    Deliberately drops ALL of the conversation's buffers, not only the truncated
    turns: buffers are just reconnect cache (the DB transcript is the source of
    truth on reload), so a whole-conversation wipe is both safe and simplest.

    This used to justify itself with "a delete never races an in-flight stream
    for the same conversation". That held only while a turn was always
    streaming: a user does not delete a message while the agent is visibly
    typing. Since `ask_user` a turn can be alive but idle for up to 300 s,
    which is precisely when a delete looks safe to the user — so the race is
    now normal, and `registry.discard` handles it by cancelling the producer
    and marking it discarded (see its docstring) before these files go away.
    """
    from cowork.streaming.backend import remove_conversation_buffers
    from cowork.streaming.turn_index import forget_turn_sync

    cid = str(conversation_id)
    registry.discard(cid)
    remove_conversation_buffers(cid)
    # The index outlives the buffers by an hour otherwise, so /in-flight would
    # keep naming a turn whose buffer has just been deleted.
    forget_turn_sync(cid)
