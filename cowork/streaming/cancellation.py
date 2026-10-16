"""Cancel a captured run, locally and on the worker that owns its correlation id."""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from cowork.streaming.backend import get_backend
from cowork.turnqueue.redis_client import STALLED_CANCEL_REASON, cancel_cause_key, cancel_flag_key, get_redis

if TYPE_CHECKING:
    from cowork.streaming.buffer import StreamBuffer
    from cowork.streaming.registry import RunHandle, TurnLifecycle

logger = logging.getLogger(__name__)
CANCEL_FLAG_TTL_SECONDS = 300
CANCEL_POLL_SECONDS = 0.5
CANCEL_REQUEST_SECONDS = 5.0


async def request_cancel(correlation_id: str, *, reason: str | None = None, client=None) -> None:
    """One transaction keeps the cause alive after the controller removes its flag."""
    r = client if client is not None else get_redis()
    async with asyncio.timeout(CANCEL_REQUEST_SECONDS):
        async with r.pipeline(transaction=True) as pipe:
            if reason in (STALLED_CANCEL_REASON, "client_gone"):
                pipe.set(cancel_cause_key(correlation_id), reason, ex=CANCEL_FLAG_TTL_SECONDS)
            else:
                pipe.delete(cancel_cause_key(correlation_id))
            pipe.set(cancel_flag_key(correlation_id), "1", ex=CANCEL_FLAG_TTL_SECONDS)
            await pipe.execute()


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


async def watch_cancel(task: asyncio.Task, buffer: StreamBuffer, lifecycle: TurnLifecycle) -> None:
    """Let any replica cancel a direct/in-process producer using the worker flag protocol.

    Poll the captured correlation key, never the conversation's mutable index.
    Worker cancellation still needs its flag: stopping our tail alone cannot
    stop a pod. Failures leave the task running and are retried on the next tick.
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
        lifecycle.stalled = reason == STALLED_CANCEL_REASON
        lifecycle.cancel_requested = True
        task.cancel()
        return
