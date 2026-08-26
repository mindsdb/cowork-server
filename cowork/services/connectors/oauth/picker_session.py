"""Short-lived, single-use Google Drive Picker session store.

Bridges the authenticated `POST .../picker/session` call (a `fetch()`, so it
carries the caller's real Bearer header) to the unauthenticated `GET
.../picker` popup navigation that actually renders the picker page — a
`window.open()` navigation cannot carry a custom header, so the live access
token has to be minted up front, at POST time, and handed to the GET route
some other way. See `oauth.py`'s `create_picker_session`/`oauth_picker`.

Backed by the same shared Redis the turn queue already uses
(`cowork.turnqueue.redis_client`), not an in-process dict, so this works
correctly behind multiple cowork-server replicas — the POST and the GET
popup navigation can land on different pods.
"""
from __future__ import annotations

import json
import secrets

from cowork.turnqueue.redis_client import get_redis

_KEY_PREFIX = "oauth:picker:session:"
_TTL_SECONDS = 300  # plenty of time for a user to notice the popup and act


def _key(session_id: str) -> str:
    return f"{_KEY_PREFIX}{session_id}"


async def create(payload: dict) -> str:
    session_id = secrets.token_urlsafe(24)
    redis = get_redis()
    await redis.set(_key(session_id), json.dumps(payload), ex=_TTL_SECONDS)
    return session_id


async def consume(session_id: str) -> dict | None:
    """Single-use: read-then-delete. A picker session is opened by exactly
    one popup that embeds the token directly into the page it renders, so
    there is no legitimate reason for the same session id to be looked up
    twice — treating it as single-use means a stale or replayed picker URL
    fails cleanly instead of re-minting a live token."""
    redis = get_redis()
    key = _key(session_id)
    raw = await redis.get(key)
    if raw is None:
        return None
    await redis.delete(key)
    return json.loads(raw)
