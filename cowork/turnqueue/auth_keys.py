"""Mint a short-TTL, org/user-scoped MindsHub 'turn' key for one turn.

The key never persists: it is used for a single turn, expires within minutes,
and keeps the long-lived tenant key out of the worker pod.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Literal

import httpx

from cowork.services.mindshub_workspaces import forget_stale_mindshub_workspace
from cowork.services.product_permissions import (
    ProductPermissionDenied,
    auth_refusal_reason,
    permission_unavailable,
)

KeyPurpose = Literal["execution", "artifact_publish", "connector_token"]

logger = logging.getLogger(__name__)

TURN_KEYS_ROUTE = "/internal/turn-keys/"
DATASOURCE_CONNECTIONS_ROUTE = "/internal/datasources/connections/"
DATASOURCE_GRANTS_ROUTE = "/internal/datasources/turn-grants/"
CUSTOM_CONNECTIONS_ROUTE = "/internal/custom-connections/"
CUSTOM_GRANTS_ROUTE = "/internal/custom-connections/turn-grants/"
CUSTOM_MCP_GRANTS_ROUTE = "/internal/custom-mcp/turn-grants/"

# auth's turn-key mint refuses an expiry past its cap with a 400 whose detail
# starts with this text, and with no error code. The text is the only sign of
# that refusal. The tests here pin this matcher to the text auth sends today;
# nothing on auth's side pins it, so a reworded refusal ends the fallback, and
# the turn answers permission_unavailable with auth's detail in its log line.
TTL_REFUSAL_PREFIX = "expiry_date exceeds the max turn-key TTL"


def _error_code(resp: httpx.Response) -> str | None:
    """The ``code`` field of an error body, or None if it has none."""
    try:
        error = resp.json()
    except (ValueError, RecursionError):
        return None
    return error.get("code") if isinstance(error, dict) else None


@dataclass(frozen=True)
class MintedTurnKey:
    """A key auth minted.

    ``prefix`` is the key's public prefix, None when auth's answer had none
    (``mint_turn_key_details`` refuses that). ``ttl_seconds`` is the lifetime
    auth minted the key for, which is shorter than the one asked for when auth
    refused that (``_request_turn_key``).
    """

    key: str
    prefix: str | None
    ttl_seconds: int


def _refuses_the_lifetime(*, resp: httpx.Response) -> bool:
    """Whether auth refused the mint because the key's expiry is past its cap."""
    if resp.status_code != 400:
        return False
    try:
        body = resp.json()
    except (ValueError, RecursionError):
        return False
    detail = body.get("detail") if isinstance(body, dict) else None
    return isinstance(detail, str) and detail.startswith(TTL_REFUSAL_PREFIX)


def _expiry(*, ttl_seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()


async def _request_turn_key(*, user_id: str, org_id: str, correlation_id: str,
                            ttl_seconds: int, settings,
                            purpose: KeyPurpose,
                            workspace_id: str | None = None) -> MintedTurnKey:
    """Mint one key, falling back once on each refusal that a changed request answers.

    A 404 ``workspace_not_found`` for the stored workspace mints on the
    organization's default. A 400 that refuses the key's lifetime, from an auth
    whose cap is below it, mints with the untimed lifetime
    (``settings.turn_key_ttl_seconds``) when the asked lifetime is longer.
    Each fallback changes the request, so neither repeats: three posts at most.
    A 403 ``permission_denied`` raises ProductPermissionDenied, and every
    other failure raises ProductPermissionUnavailable, each with no further post.
    """
    # Cluster-only route: turn-key mint is secret-only (no Bearer factor), so
    # auth serves it under the top-level /internal/ prefix the public LB never
    # routes, NOT /v1/internal/. Reached here over ClusterIP (auth_internal_base_url).
    url = f"{settings.auth_internal_base_url.rstrip('/')}{TURN_KEYS_ROUTE}"
    headers = {"X-Internal-Auth": settings.auth_internal_secret}
    body = {"user_id": user_id, "organization_id": org_id,
            "instance_id": correlation_id, "expiry_date": _expiry(ttl_seconds=ttl_seconds), "rotate": False,
            "purpose": purpose}
    # Omitted rather than sent empty: auth treats a missing/null workspace_id as
    # "use the organization's Default", which is also what an unselected
    # (empty-string) hub_workspace_id means here.
    if workspace_id:
        body["workspace_id"] = workspace_id
    if not settings.auth_internal_base_url or not settings.auth_internal_secret:
        raise permission_unavailable(route=TURN_KEYS_ROUTE, cause="not_configured")
    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
            resp = await client.post(url, json=body, headers=headers)
            for _fallback in range(2):
                if resp.status_code == 404 and "workspace_id" in body and _error_code(resp) == "workspace_not_found":
                    # The stored pick names a workspace this person can no longer
                    # use (grant removed, or deleted). The menu has already fallen
                    # back to the default; do the same here instead of failing
                    # every turn, and clear the pick so the next turn skips this.
                    logger.info(
                        "turn key mint: workspace %s refused for user %s; retrying on the default",
                        workspace_id, user_id,
                    )
                    forget_stale_mindshub_workspace(org_id=org_id, user_id=user_id, workspace_id=workspace_id)
                    del body["workspace_id"]
                elif _refuses_the_lifetime(resp=resp) and ttl_seconds > settings.turn_key_ttl_seconds:
                    # auth caps keys below the time-limited window. Every auth
                    # mints the untimed lifetime, and the job then runs without
                    # its time limit, since the key no longer covers that window
                    # (producer.stream_remote_replies).
                    logger.warning(
                        "turn key mint: auth refused the key's lifetime; minting the untimed lifetime, "
                        "so a time-limited turn runs untimed",
                        extra={"event": "turn_key_untimed_fallback", "purpose": purpose,
                               "requested_ttl_seconds": ttl_seconds, "ttl_seconds": settings.turn_key_ttl_seconds,
                               "reason": auth_refusal_reason(response=resp)},
                    )
                    ttl_seconds = settings.turn_key_ttl_seconds
                    body["expiry_date"] = _expiry(ttl_seconds=ttl_seconds)
                else:
                    break
                resp = await client.post(url, json=body, headers=headers)
            if resp.status_code == 403 and _error_code(resp) == "permission_denied":
                raise ProductPermissionDenied()
            resp.raise_for_status()
            result = resp.json()
    except (httpx.HTTPError, TimeoutError, ValueError, RecursionError) as exc:
        raise permission_unavailable(route=TURN_KEYS_ROUTE, cause=exc) from exc
    if not isinstance(result, dict) or not isinstance(result.get("key"), str) or not result["key"].strip():
        raise permission_unavailable(route=TURN_KEYS_ROUTE, cause="answer_without_key")
    prefix = result.get("prefix")
    return MintedTurnKey(
        key=result["key"],
        prefix=prefix if isinstance(prefix, str) and prefix.strip() else None,
        ttl_seconds=ttl_seconds,
    )


async def mint_turn_key(*, user_id: str, org_id: str, correlation_id: str,
                        ttl_seconds: int, settings, purpose: KeyPurpose = "execution",
                        workspace_id: str | None = None) -> str:
    minted = await _request_turn_key(
        user_id=user_id,
        org_id=org_id,
        correlation_id=correlation_id,
        ttl_seconds=ttl_seconds,
        settings=settings,
        purpose=purpose,
        workspace_id=workspace_id,
    )
    return minted.key


async def mint_turn_key_details(*, user_id: str, org_id: str, correlation_id: str,
                                ttl_seconds: int, settings,
                                purpose: KeyPurpose = "execution",
                                workspace_id: str | None = None) -> MintedTurnKey:
    """As mint_turn_key, returning the key with its public prefix and the lifetime auth minted it for."""
    minted = await _request_turn_key(
        user_id=user_id,
        org_id=org_id,
        correlation_id=correlation_id,
        ttl_seconds=ttl_seconds,
        settings=settings,
        purpose=purpose,
        workspace_id=workspace_id,
    )
    if minted.prefix is None:
        raise permission_unavailable(route=TURN_KEYS_ROUTE, cause="answer_without_prefix")
    return minted


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
        raise permission_unavailable(route=DATASOURCE_CONNECTIONS_ROUTE, cause="producer_key_not_configured")
    url = f"{settings.auth_internal_base_url.rstrip('/')}{DATASOURCE_CONNECTIONS_ROUTE}"
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
    except (httpx.HTTPError, TimeoutError, ValueError, RecursionError) as exc:
        raise permission_unavailable(route=DATASOURCE_CONNECTIONS_ROUTE, cause=exc) from exc
    if not isinstance(result, dict) or not isinstance(result.get("items"), list):
        raise permission_unavailable(route=DATASOURCE_CONNECTIONS_ROUTE, cause="answer_without_items")
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
        raise permission_unavailable(route=DATASOURCE_GRANTS_ROUTE, cause="producer_key_not_configured")
    url = f"{settings.auth_internal_base_url.rstrip('/')}{DATASOURCE_GRANTS_ROUTE}"
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
    except (httpx.HTTPError, TimeoutError, ValueError, RecursionError) as exc:
        raise permission_unavailable(route=DATASOURCE_GRANTS_ROUTE, cause=exc) from exc
    if not isinstance(result, dict):
        raise permission_unavailable(route=DATASOURCE_GRANTS_ROUTE, cause="answer_not_an_object")
    return result


def _producer_headers(settings) -> dict[str, str]:
    """The datasource producer role's credential, which auth's custom API routes also take."""
    return {
        "X-Datasource-Service-Key-Id": settings.datasource_producer_key_id,
        "X-Datasource-Service-Key": settings.datasource_producer_key,
    }


async def list_custom_api_connections(*, org_id: str, user_id: str, turn_key_id: str, settings) -> list[dict]:
    """List the turn key owner's verified custom API connections, metadata only.

    Auth derives the member from the live turn key named by ``turn_key_id``;
    the ids are cross-checks. Raises ``ProductPermissionUnavailable`` on any
    failure, an unconfigured producer key included.
    """
    if not settings.datasource_producer_key_id or not settings.datasource_producer_key:
        raise permission_unavailable(route=CUSTOM_CONNECTIONS_ROUTE, cause="producer_key_not_configured")
    url = f"{settings.auth_internal_base_url.rstrip('/')}{CUSTOM_CONNECTIONS_ROUTE}"
    params = {"organization_id": org_id, "user_id": user_id, "turn_key_id": turn_key_id}
    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
            resp = await client.get(url, params=params, headers=_producer_headers(settings))
            resp.raise_for_status()
            result = resp.json()
    except (httpx.HTTPError, TimeoutError, ValueError, RecursionError) as exc:
        raise permission_unavailable(route=CUSTOM_CONNECTIONS_ROUTE, cause=exc) from exc
    if not isinstance(result, dict) or not isinstance(result.get("items"), list):
        raise permission_unavailable(route=CUSTOM_CONNECTIONS_ROUTE, cause="answer_without_items")
    return result["items"]


async def register_custom_api_grants(
    *,
    user_id: str,
    org_id: str,
    correlation_id: str,
    turn_key_id: str,
    connection_ids: list[int],
    settings,
) -> dict:
    """Bind grants for these custom API connections to the existing turn key, for the gateway to resolve.

    Auth fixes the audience and purpose itself (``custom-api``, ``fetch``) and
    issues each grant at the connection's current credential version. Raises
    ``ProductPermissionUnavailable`` on any refusal or failure.
    """
    if not settings.datasource_producer_key_id or not settings.datasource_producer_key:
        raise permission_unavailable(route=CUSTOM_GRANTS_ROUTE, cause="producer_key_not_configured")
    url = f"{settings.auth_internal_base_url.rstrip('/')}{CUSTOM_GRANTS_ROUTE}"
    body = {
        "user_id": user_id,
        "organization_id": org_id,
        "correlation_id": correlation_id,
        "turn_key_id": turn_key_id,
        "connection_ids": connection_ids,
    }
    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
            resp = await client.post(url, json=body, headers=_producer_headers(settings))
            resp.raise_for_status()
            result = resp.json()
    except (httpx.HTTPError, TimeoutError, ValueError, RecursionError) as exc:
        raise permission_unavailable(route=CUSTOM_GRANTS_ROUTE, cause=exc) from exc
    if not isinstance(result, dict):
        raise permission_unavailable(route=CUSTOM_GRANTS_ROUTE, cause="answer_not_an_object")
    return result


#: Answers that mean auth or its proxy did not take the request; one retry is worth it.
_FAST_FAILURE_STATUSES = frozenset({502, 503, 504})


def auth_switched_off(response: httpx.Response) -> bool:
    """Whether ``response`` is auth's 503 ``disabled``: its custom MCP switch is off, a settled answer.

    Any other 503, a proxy's included, is not: its body is not auth's error object.
    """
    if response.status_code != 503:
        return False
    try:
        body = response.json()
    except (ValueError, RecursionError):
        return False
    return isinstance(body, dict) and body.get("code") == "disabled"


async def _send_retrying_once(client: httpx.AsyncClient, method: str, url: str, **kwargs) -> httpx.Response:
    """Send one request, and once more only when the first failed fast.

    A fast failure is no connection at all, or a 502, 503 or 504 answer. A
    timeout already spent the budget and a refusal would answer the same, so
    neither is retried; nor is auth's switched-off answer, which would too.
    Returns the last response; raises what httpx raises.
    """
    try:
        resp = await client.request(method, url, **kwargs)
    except httpx.ConnectError:
        return await client.request(method, url, **kwargs)
    if resp.status_code in _FAST_FAILURE_STATUSES and not auth_switched_off(resp):
        return await client.request(method, url, **kwargs)
    return resp


async def _custom_mcp_call(method: str, path: str, *, settings, params: dict | None = None,
                           json_body: dict | None = None) -> dict:
    """One producer-key call to auth's internal custom MCP routes, as a JSON object.

    Raises ``ProductPermissionUnavailable`` on any failure, refusal or timeout,
    an unconfigured producer key included; an HTTP refusal is its cause, so the
    caller can read the status. A body nested too deep to parse is a failure too.
    """
    if not settings.datasource_producer_key_id or not settings.datasource_producer_key:
        raise permission_unavailable(route=path, cause="producer_key_not_configured")
    url = f"{settings.auth_internal_base_url.rstrip('/')}{path}"
    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
            resp = await _send_retrying_once(
                client, method, url, params=params, json=json_body, headers=_producer_headers(settings),
            )
            resp.raise_for_status()
            result = resp.json()
    except (httpx.HTTPError, TimeoutError, ValueError, RecursionError) as exc:
        raise permission_unavailable(route=path, cause=exc) from exc
    if not isinstance(result, dict):
        raise permission_unavailable(route=path, cause="answer_not_an_object")
    return result


async def list_custom_mcp_connections(*, org_id: str, user_id: str, turn_key_id: str, settings) -> dict:
    """auth's answer listing the turn key owner's custom MCP servers with their switched-on definitions.

    Auth derives the member from the live turn key named by ``turn_key_id``;
    the ids are cross-checks. Returns the whole answer unread, so the caller
    tells a malformed listing from one that did not arrive. Raises
    ``ProductPermissionUnavailable`` when it did not arrive.
    """
    params = {"organization_id": org_id, "user_id": user_id, "turn_key_id": turn_key_id}
    return await _custom_mcp_call("GET", "/internal/custom-mcp/connections/", settings=settings, params=params)


async def register_custom_mcp_grants(
    *,
    user_id: str,
    org_id: str,
    correlation_id: str,
    turn_key_id: str,
    connection_ids: list[int],
    settings,
) -> dict:
    """Bind grants for these custom MCP connections to the existing turn key, for the gateway to resolve.

    Auth fixes the audience and purpose itself (``custom-mcp``, ``call``),
    issues each grant at the connection's current credential version and
    counts the turn in the shared hourly quota. Raises
    ``ProductPermissionUnavailable`` on any refusal or failure.
    """
    body = {
        "user_id": user_id,
        "organization_id": org_id,
        "correlation_id": correlation_id,
        "turn_key_id": turn_key_id,
        "connection_ids": connection_ids,
    }
    return await _custom_mcp_call("POST", CUSTOM_MCP_GRANTS_ROUTE, settings=settings, json_body=body)


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


async def extend_turn_key(*, instance_id: str, ttl_seconds: int, settings) -> bool:
    """Move the execution key of turn `instance_id` to expire `ttl_seconds` from now.

    The key string stays the same, so the running pod keeps using it. auth
    re-checks the key owner's permission and bounds the expiry (per window and
    over the key's whole life). Returns False, after logging, when auth refuses
    or can't be reached: the turn then runs on until its current expiry.
    """
    if not settings.auth_internal_base_url or not settings.auth_internal_secret:
        return False
    expiry = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()
    url = f"{settings.auth_internal_base_url.rstrip('/')}/internal/turn-keys/{instance_id}/"
    headers = {"X-Internal-Auth": settings.auth_internal_secret}
    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
            resp = await client.patch(url, json={"expiry_date": expiry}, headers=headers)
    except (httpx.HTTPError, TimeoutError):
        logger.warning("turn key extend failed: instance_id=%s", instance_id, exc_info=True)
        return False
    if resp.status_code != 200:
        logger.warning(
            "turn key extend refused: instance_id=%s status=%s code=%s",
            instance_id, resp.status_code, _error_code(resp),
        )
        return False
    return True
