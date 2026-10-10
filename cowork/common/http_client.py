"""Shared httpx.AsyncClient singleton.

Used by the artifact preview proxy to forward iframe requests to
artifact-owned backends. A single long-lived client lets us reuse
TCP connections to loopback artifact servers across many requests
in the same UI session. Closed during FastAPI lifespan shutdown.
"""
from __future__ import annotations

import asyncio
from http.cookiejar import CookieJar, DefaultCookiePolicy
from typing import Optional

import httpx


_client: Optional[httpx.AsyncClient] = None
_lock = asyncio.Lock()


def get_proxy_client() -> httpx.AsyncClient:
    """Return the process-wide proxy client, creating it on first use."""
    global _client
    if _client is None:
        # Shared by every visitor: a cookie jar here would replay one visitor's
        # upstream cookies on another's requests, so it accepts nothing.
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0),
            cookies=CookieJar(policy=DefaultCookiePolicy(allowed_domains=[])),
        )
    return _client


async def close_proxy_client() -> None:
    """Close the proxy client during shutdown. Idempotent."""
    global _client
    async with _lock:
        if _client is not None:
            try:
                await _client.aclose()
            finally:
                _client = None
