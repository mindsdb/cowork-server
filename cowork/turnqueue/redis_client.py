"""Async Redis client helper for the turn queue.

Provides a lazily-created, module-level singleton ``redis.asyncio.Redis``
client. ``reset_redis`` clears the singleton (used by tests, and by anything
that needs to pick up a changed ``COWORK_TURN_REDIS_URL``).
"""
from __future__ import annotations

import os

import redis.asyncio as aioredis

_client: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    global _client
    if _client is None:
        url = os.environ.get("COWORK_TURN_REDIS_URL", "redis://localhost:6379/0")
        _client = aioredis.from_url(url, decode_responses=True)
    return _client


def reset_redis() -> None:
    global _client
    _client = None
