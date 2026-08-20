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
    # Cluster-only route: turn-key mint is secret-only (no Bearer factor), so
    # auth serves it under the top-level /internal/ prefix the public LB never
    # routes, NOT /v1/internal/. Reached here over ClusterIP (auth_internal_base_url).
    url = f"{settings.auth_internal_base_url.rstrip('/')}/internal/turn-keys/"
    headers = {"X-Internal-Auth": settings.auth_internal_secret}
    body = {"user_id": user_id, "organization_id": org_id,
            "instance_id": correlation_id, "expiry_date": expiry, "rotate": False}
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()["key"]


async def revoke_turn_key(*, instance_id: str, settings) -> None:
    """Revoke every active turn key for `instance_id`.

    Idempotent on the auth side: the endpoint answers 204 even when no key
    exists, so callers do not need to know whether a mint happened.
    """
    url = f"{settings.auth_internal_base_url.rstrip('/')}/v1/internal/turn-keys/{instance_id}/"
    headers = {"X-Internal-Auth": settings.auth_internal_secret}
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.delete(url, headers=headers)
        resp.raise_for_status()
