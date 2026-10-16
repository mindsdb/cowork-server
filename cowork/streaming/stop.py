"""Stopping a running turn: the Stop button and the scheduler's duration bound
take the same path, so a turn that runs in a pod is told to stop as well as
the local producer that tails it."""
from __future__ import annotations

from cowork.streaming.backend import get_backend
from cowork.streaming.registry import RunHandle
from cowork.streaming.turn_index import get_turn
from cowork.turnqueue.redis_client import (
    STALLED_CANCEL_REASON,
    cancel_cause_key,
    cancel_flag_key,
    get_redis,
)

# Long enough for a controller mid-turn to notice, short enough that a stale
# flag cannot cancel a later turn that reuses the id.
CANCEL_FLAG_TTL_SECONDS = 300


async def request_remote_cancel(correlation_id: str, *, stalled: bool = False) -> None:
    """Ask whoever is running this turn to stop.

    The controller checks this key while it streams. Writing it rather than
    messaging a replica is what makes cancel work regardless of which replica
    the request landed on.

    The cause goes in the same transaction as the flag: a stall sets it, any
    other cancel clears it. scratchpad-controller's ``_clear_cancel`` deletes
    the flag right after it publishes the cancelled reply, so the replica that
    owns the turn reads the cause, not the flag, when that reply arrives. One
    transaction means the latest cancel decides: a failed write leaves no
    stray cause behind, and a Stop after a stall saves as a Stop.
    """
    async with get_redis().pipeline(transaction=True) as pipe:
        if stalled:
            pipe.set(
                cancel_cause_key(correlation_id), STALLED_CANCEL_REASON,
                ex=CANCEL_FLAG_TTL_SECONDS,
            )
        else:
            pipe.delete(cancel_cause_key(correlation_id))
        pipe.set(cancel_flag_key(correlation_id), "1", ex=CANCEL_FLAG_TTL_SECONDS)
        await pipe.execute()


async def cancel_conversation_turn(conversation_id: str) -> bool:
    """Ask the pod running this conversation's current turn to stop, from any
    replica and without its local producer. Returns whether a cancel was
    issued: never on the in-process backend, where no pod outlives the
    process that ran the turn, nor when no turn is recorded for it."""
    if not remote_backend():
        return False
    turn = await get_turn(conversation_id)
    if turn is None:
        return False
    await request_remote_cancel(turn["correlation_id"])
    return True


async def stop_turn(handle: RunHandle, *, stalled: bool = False, timed_out: bool = False) -> bool:
    """Stop the turn behind ``handle``. Returns whether a cancel was issued.

    ``timed_out`` saves the turn as interrupted rather than as the user's
    Stop. On the remote backend the pod is flagged too: cancelling the local
    producer alone only stops the tail, and the pod would run on.
    """
    turn = await get_turn(handle.conversation_id) if get_backend() == "redis" else None
    if turn is not None:
        await request_remote_cancel(turn["correlation_id"], stalled=stalled)
    # Before the cancel, so the producer's CancelledError handler cannot miss
    # it (see TurnLifecycle). Assigned, not only set: the latest cancel
    # decides, so a Stop after a stall cancel that the turn swallowed (a
    # running cell) saves as a Stop.
    handle.lifecycle.stalled = stalled
    if timed_out:
        handle.lifecycle.timed_out = True
    return await handle.cancel()


def remote_backend() -> bool:
    """Whether turns run in pods, where a local cancel cannot confirm the
    work stopped."""
    return get_backend() == "redis"
