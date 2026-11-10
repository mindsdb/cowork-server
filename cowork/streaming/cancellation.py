"""Cancel a captured run, locally and on the worker that owns its correlation id."""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from cowork.streaming.backend import get_backend
from cowork.turnqueue.redis_client import (
    DISCARDED_CANCEL_REASON,
    STALLED_CANCEL_REASON,
    cancel_cause_key,
    cancel_flag_key,
    get_redis,
    get_sync_redis,
)

if TYPE_CHECKING:
    from cowork.streaming.buffer import StreamBuffer
    from cowork.streaming.registry import RunHandle, TurnLifecycle

logger = logging.getLogger(__name__)
CANCEL_FLAG_TTL_SECONDS = 300
CANCEL_POLL_SECONDS = 0.5
CANCEL_REQUEST_SECONDS = 5.0
#: The reasons a cancel stores as its cause. Any other cancel deletes the
#: cause, so the latest cancel decides.
_STORED_CAUSES = frozenset({STALLED_CANCEL_REASON, "client_gone", DISCARDED_CANCEL_REASON})


def _queue_cancel(pipe, *, correlation_id: str, reason: str | None) -> None:
    """Queue a cancel's writes on a transaction pipeline: its cause, or the
    cause's removal, and the flag."""
    if reason in _STORED_CAUSES:
        pipe.set(cancel_cause_key(correlation_id), reason, ex=CANCEL_FLAG_TTL_SECONDS)
    else:
        pipe.delete(cancel_cause_key(correlation_id))
    pipe.set(cancel_flag_key(correlation_id), "1", ex=CANCEL_FLAG_TTL_SECONDS)


async def request_cancel(correlation_id: str, *, reason: str | None = None, client=None) -> None:
    """One transaction keeps the cause alive after the controller removes its flag."""
    r = client if client is not None else get_redis()
    async with asyncio.timeout(CANCEL_REQUEST_SECONDS):
        async with r.pipeline(transaction=True) as pipe:
            _queue_cancel(pipe, correlation_id=correlation_id, reason=reason)
            await pipe.execute()


def request_cancel_sync(correlation_id: str, *, reason: str | None = None, client=None) -> None:
    """Blocking ``request_cancel``, for conversation delete, which runs in a
    threadpool thread with no event loop. The caller's ``redis_deadline``
    bounds it."""
    r = client if client is not None else get_sync_redis()
    with r.pipeline(transaction=True) as pipe:
        _queue_cancel(pipe, correlation_id=correlation_id, reason=reason)
        pipe.execute()


async def cancel_run(handle: RunHandle, *, reason: str | None = None) -> bool:
    """Request cancellation without looking up a conversation again or waiting on teardown.

    Completion wins over automatic cancellation. The task and the worker flag
    always identify the same captured instance, even if another run starts
    while the Redis write is in flight.
    """
    if not handle.is_answering or handle.lifecycle.shutting_down:
        return False
    if get_backend() == "redis" or handle.lifecycle.identity.remote:
        await request_cancel(handle.lifecycle.identity.correlation_id, reason=reason)
    if not handle.is_answering or handle.lifecycle.shutting_down:
        return False
    if handle.lifecycle.cancel_requested:
        return True  # the owner observed our Redis flag while its write was settling
    handle.lifecycle.stalled = reason == STALLED_CANCEL_REASON
    return handle.request_cancel()


async def cancel_disconnected_run(handle: RunHandle | None) -> None:
    """A lost transport cannot receive an error; log a failed cancellation explicitly."""
    if handle is None:
        return
    try:
        await cancel_run(handle, reason="client_gone")
    except Exception:
        logger.exception("Could not cancel disconnected run %s", handle.run_id)


async def stop_controller_turn(correlation_id: str, *, turn_id: str, reason: str) -> None:
    """Write the sandbox controller's cancel flag for a turn this server is
    ending without its user's Stop, so the sandbox turn stops too.

    A failure is logged as a WARNING, never raised: the caller is already
    ending the turn, and the cancel it is unwinding must not turn into a
    Redis error. The line carries the correlation id as ``request_id``, as
    the turn's other lines do, plus ``turn_id`` and why the turn is ending
    (``reason``), as fields: the hosted log formatter drops a message's
    arguments.
    """
    try:
        await request_cancel(correlation_id)
    except Exception:
        logger.warning(
            "Could not ask the sandbox controller to stop a turn; the sandbox turn may run on.",
            exc_info=True,
            extra={
                "event": "controller_turn_stop_failed",
                "request_id": correlation_id,
                "turn_id": turn_id,
                "reason": reason,
            },
        )


async def watch_cancel(task: asyncio.Task, buffer: StreamBuffer, lifecycle: TurnLifecycle) -> None:
    """Let any replica cancel a direct/in-process producer using the worker flag protocol.

    Poll the captured correlation key, never the conversation's mutable index.
    Worker cancellation still needs its flag: stopping our tail alone cannot
    stop a pod. Failures leave the task running and are retried on the next tick.
    A cancel whose cause is ``discarded`` marks the turn discarded before it
    cancels it, as RunRegistry.discard does, since another replica deleted
    its conversation.
    """
    correlation_id = lifecycle.identity.correlation_id
    while not task.done():
        await asyncio.sleep(CANCEL_POLL_SECONDS)
        if buffer.is_closed or buffer.is_finishing or lifecycle.shutting_down:
            return
        try:
            r = get_redis()
            if not await r.exists(cancel_flag_key(correlation_id)):
                continue
            reason = await r.get(cancel_cause_key(correlation_id))
        except Exception:
            logger.warning("Could not check cancellation for run %s", lifecycle.identity.run_id, exc_info=True)
            await asyncio.sleep(CANCEL_REQUEST_SECONDS)
            continue
        if buffer.is_closed or buffer.is_finishing or lifecycle.shutting_down:
            return
        if lifecycle.cancel_requested:
            return
        if reason == DISCARDED_CANCEL_REASON:
            lifecycle.note_discarded_elsewhere()
        lifecycle.stalled = reason == STALLED_CANCEL_REASON
        lifecycle.cancel_requested = True
        task.cancel()
        return
