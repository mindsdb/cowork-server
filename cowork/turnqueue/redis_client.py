"""Async Redis client helper for the turn queue.

Provides a lazily-created, module-level singleton ``redis.asyncio.Redis``
client. ``reset_redis`` clears the singleton (used by tests, and by anything
that needs to pick up a changed ``COWORK_TURN_REDIS_URL``).
"""
from __future__ import annotations

import os

import redis as syncredis
import redis.asyncio as aioredis

_client: aioredis.Redis | None = None
_sync_client: syncredis.Redis | None = None


def _url() -> str:
    return os.environ.get("COWORK_TURN_REDIS_URL", "redis://localhost:6379/0")


def cancel_flag_key(correlation_id: str) -> str:
    """The key a ``/cancel`` writes and whoever runs the turn polls.

    Three processes agree on this name: the endpoint sets it, the producer
    clears a stale one before each turn, and scratchpad-controller's
    ``_cancel_key`` rebuilds it. Kept in one place on this side so the local
    users cannot drift from each other.
    """
    return f"cowork:cancel:{correlation_id}"


# The one cancel reason that changes what a turn saves: the UI cancelled it
# after hearing nothing for its idle window, so it saves as a stall, not a Stop.
STALLED_CANCEL_REASON = "stalled"


def cancel_cause_key(correlation_id: str) -> str:
    """Why a ``/cancel`` asked this turn to stop, when the reason matters.

    Separate from ``cancel_flag_key`` because scratchpad-controller's
    ``_clear_cancel`` deletes the flag right after it publishes the cancelled
    reply, so the flag may be gone when the replica that owns the turn hears
    of the cancel. Only this server reads and writes it: the replica that
    takes the ``/cancel`` sets or clears it in one transaction with the flag
    (``_request_cancel``), the owner reads it when the cancel comes back, and
    the producer clears a stale one before each turn.
    """
    return f"cowork:cancel_cause:{correlation_id}"


def answer_queue_key(correlation_id: str) -> str:
    """The list /answer pushes a remote turn's ask_user answers onto.

    scratchpad-controller's ``answers.answer_key`` rebuilds the same name and
    pops it on the replica running the turn; kept here for the same reason as
    ``cancel_flag_key``.
    """
    return f"cowork:answer:{correlation_id}"


def reply_stream_key(conversation_id: str) -> str:
    """The stream scratchpad-controller relays a turn's replies onto.

    Both the producer (writing the job) and ``turnqueue/answers.py`` (reading
    the pod's verdict) need this name; kept here so the two cannot drift.
    """
    return f"scratchpad:reply:{conversation_id}"


def get_redis() -> aioredis.Redis:
    global _client
    if _client is None:
        _client = aioredis.from_url(_url(), decode_responses=True)
    return _client


def get_sync_redis() -> syncredis.Redis:
    """Blocking client on the same URL, for callers with no event loop.

    Conversation delete runs in a threadpool thread (the endpoint is a sync
    ``def``), so it cannot await the async client.
    """
    global _sync_client
    if _sync_client is None:
        _sync_client = syncredis.Redis.from_url(_url(), decode_responses=True)
    return _sync_client


def reset_redis() -> None:
    global _client, _sync_client
    _client = None
    _sync_client = None
