"""Mint a short-TTL, org/user-scoped MindsHub 'turn' key for one turn.

The key never persists: it is used for a single turn, expires within minutes,
and keeps the long-lived tenant key out of the worker pod.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx


async def mint_turn_key(*, user_id: str, org_id: str, correlation_id: str,
                        ttl_seconds: int, settings) -> str:
    expiry = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()
    url = f"{settings.auth_internal_base_url.rstrip('/')}/v1/internal/turn-keys/"
    headers = {"X-Internal-Auth": settings.auth_internal_secret}
    body = {"user_id": user_id, "organization_id": org_id,
            "instance_id": correlation_id, "expiry_date": expiry, "rotate": False}
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()["key"]
