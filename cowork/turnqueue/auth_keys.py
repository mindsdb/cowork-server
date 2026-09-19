"""Mint a short-TTL, org/user-scoped MindsHub 'turn' key for one turn.

The key never persists: it is used for a single turn, expires within minutes,
and keeps the long-lived tenant key out of the worker pod.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Literal

import httpx

from cowork.services.product_permissions import ProductPermissionDenied, ProductPermissionUnavailable


@dataclass(frozen=True)
class MintedTurnKey:
    key: str
    prefix: str


async def _request_turn_key(*, user_id: str, org_id: str, correlation_id: str,
                            ttl_seconds: int, settings,
                            purpose: Literal["execution", "artifact_publish"]) -> dict:
    expiry = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()
    # Cluster-only route: turn-key mint is secret-only (no Bearer factor), so
    # auth serves it under the top-level /internal/ prefix the public LB never
    # routes, NOT /v1/internal/. Reached here over ClusterIP (auth_internal_base_url).
    url = f"{settings.auth_internal_base_url.rstrip('/')}/internal/turn-keys/"
    headers = {"X-Internal-Auth": settings.auth_internal_secret}
    body = {"user_id": user_id, "organization_id": org_id,
            "instance_id": correlation_id, "expiry_date": expiry, "rotate": False, "purpose": purpose}
    if not settings.auth_internal_base_url or not settings.auth_internal_secret:
        raise ProductPermissionUnavailable()
    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
            resp = await client.post(url, json=body, headers=headers)
            if resp.status_code == 403:
                error = resp.json()
                if isinstance(error, dict) and error.get("code") == "permission_denied":
                    raise ProductPermissionDenied()
            resp.raise_for_status()
            result = resp.json()
            if not isinstance(result, dict) or not isinstance(result.get("key"), str) or not result["key"].strip():
                raise ProductPermissionUnavailable()
            return result
    except (httpx.HTTPError, TimeoutError, ValueError) as exc:
        raise ProductPermissionUnavailable() from exc


async def mint_turn_key(*, user_id: str, org_id: str, correlation_id: str,
                        ttl_seconds: int, settings, purpose: Literal["execution", "artifact_publish"] = "execution") -> str:
    result = await _request_turn_key(
        user_id=user_id,
        org_id=org_id,
        correlation_id=correlation_id,
        ttl_seconds=ttl_seconds,
        settings=settings,
        purpose=purpose,
    )
    return result["key"]


async def mint_turn_key_details(*, user_id: str, org_id: str, correlation_id: str,
                                ttl_seconds: int, settings,
                                purpose: Literal["execution", "artifact_publish"] = "execution") -> MintedTurnKey:
    result = await _request_turn_key(
        user_id=user_id,
        org_id=org_id,
        correlation_id=correlation_id,
        ttl_seconds=ttl_seconds,
        settings=settings,
        purpose=purpose,
    )
    prefix = result.get("prefix")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ProductPermissionUnavailable()
    return MintedTurnKey(key=result["key"], prefix=prefix)



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


async def list_verified_datasource_connections(*, org_id: str, user_id: str, turn_key_id: str, settings) -> list[dict]:
    """List auth-owned, verified datasource metadata for one user and org.

    Auth derives the identity from the live turn key named by ``turn_key_id``
    (the public prefix the mint returned); the ids are cross-checks only.
    """
    if not settings.datasource_producer_key_id or not settings.datasource_producer_key:
        raise ProductPermissionUnavailable()
    url = f"{settings.auth_internal_base_url.rstrip('/')}/internal/datasources/connections/"
    headers = {
        "X-Datasource-Service-Key-Id": settings.datasource_producer_key_id,
        "X-Datasource-Service-Key": settings.datasource_producer_key,
    }
    params = {"organization_id": org_id, "user_id": user_id, "turn_key_id": turn_key_id}
    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            result = resp.json()
    except (httpx.HTTPError, TimeoutError, ValueError) as exc:
        raise ProductPermissionUnavailable() from exc
    if not isinstance(result, dict) or not isinstance(result.get("items"), list):
        raise ProductPermissionUnavailable()
    return result["items"]


async def register_datasource_grants(
    *,
    user_id: str,
    org_id: str,
    correlation_id: str,
    turn_key_id: str,
    connection_ids: list[int],
    settings,
) -> dict:
    """Register an immutable grant set for the existing turn key."""
    if not settings.datasource_producer_key_id or not settings.datasource_producer_key:
        raise ProductPermissionUnavailable()
    url = f"{settings.auth_internal_base_url.rstrip('/')}/internal/datasources/turn-grants/"
    headers = {
        "X-Datasource-Service-Key-Id": settings.datasource_producer_key_id,
        "X-Datasource-Service-Key": settings.datasource_producer_key,
    }
    body = {
        "user_id": user_id,
        "organization_id": org_id,
        "correlation_id": correlation_id,
        "turn_key_id": turn_key_id,
        "connection_ids": connection_ids,
        "audience": "datasource-gateway",
        "purpose": "execute",
    }
    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            result = resp.json()
    except (httpx.HTTPError, TimeoutError, ValueError) as exc:
        raise ProductPermissionUnavailable() from exc
    if not isinstance(result, dict):
        raise ProductPermissionUnavailable()
    return result


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
