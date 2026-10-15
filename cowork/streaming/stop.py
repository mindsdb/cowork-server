"""Stopping a scheduled run's turn: it takes the same cancellation path as
the Stop button, so a turn that runs in a pod is told to stop as well as the
local producer that tails it."""
from __future__ import annotations

from cowork.streaming.backend import get_backend
from cowork.streaming.cancellation import CANCEL_FLAG_TTL_SECONDS, request_cancel
from cowork.streaming.registry import RunHandle

__all__ = ["CANCEL_FLAG_TTL_SECONDS", "remote_backend", "stop_turn"]


async def stop_turn(handle: RunHandle, *, timed_out: bool = False) -> bool:
    """Stop the turn behind ``handle`` and wait for its producer to unwind.
    Returns whether a cancel was issued.

    ``timed_out`` saves the turn as interrupted rather than as the user's
    Stop. On the remote backend the pod is flagged too: cancelling the local
    producer alone only stops the tail, and the pod would run on.
    """
    if remote_backend() or handle.lifecycle.identity.remote:
        await request_cancel(handle.lifecycle.identity.correlation_id)
    # Before the cancel, so the producer's CancelledError handler cannot miss
    # it (see TurnLifecycle).
    handle.lifecycle.stalled = False
    if timed_out:
        handle.lifecycle.timed_out = True
    return await handle.cancel()


def remote_backend() -> bool:
    """Whether turns run in pods, where a local cancel cannot confirm the
    work stopped."""
    return get_backend() == "redis"
