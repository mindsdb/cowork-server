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
