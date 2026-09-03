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


async def list_active_connections(*, org_id: str, user_id: str, settings) -> list[dict]:
    """Org's active OAuth-builtin connections, for the turn-key `oauth` block
    (Turn-Key Token Handoff). Internal/service-authenticated, same mechanism
    as mint_turn_key — not the caller's own Bearer credential: by the time
    the remote producer builds this block it only has the gateway-verified
    Principal (org_id/user_id), never the original request's raw
    Authorization header (ResponsesHandler is constructed from a Principal,
    not a Request). Returns each connection as {"engine": ..., "name": ...}.
    """
    url = f"{settings.auth_internal_base_url.rstrip('/')}/internal/oauth/connections/"
    headers = {"X-Internal-Auth": settings.auth_internal_secret}
    params = {"organization_id": org_id, "user_id": user_id}
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json().get("items", [])


async def revoke_turn_key(*, instance_id: str, settings) -> None:
    """Revoke every active turn key for `instance_id`.

    Idempotent on the auth side: the endpoint answers 204 even when no key
    exists, so callers do not need to know whether a mint happened. Like mint,
    revoke uses auth's ClusterIP-only top-level ``/internal/`` route; the public
    edge deliberately has no turn-key surface.
    """
    url = f"{settings.auth_internal_base_url.rstrip('/')}/internal/turn-keys/{instance_id}/"
    headers = {"X-Internal-Auth": settings.auth_internal_secret}
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.delete(url, headers=headers)
        resp.raise_for_status()
