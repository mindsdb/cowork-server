"""Org-mode relay: forward OAuth connector lifecycle + token calls to `auth`.

In org/cloud deployments cowork-server never talks to Google directly and
never touches the local vault for OAuth-builtin connectors — it's a
transparent relay to `auth`'s own `/v1/oauth/...` endpoints, authenticated
with the caller's own Bearer/session credential (the same one the gateway
already validated to populate `principal.py`'s trusted identity headers).
No separate service identity: `auth`'s public `/v1/...` routes accept that
credential directly via its generic JWT-or-API-key handling.

See the "OAuth Proxy + Data Vault" blueprint (cowork-server tab, OAuth
Connector Lifecycle + Google Drive File Picker) for the design this ports.
Local-mode desktop flow (`oauth/google.py: OAuthService`) is untouched —
this module is only ever reached from the org-mode branch of the
connectors/oauth endpoints.
"""
from __future__ import annotations

import httpx
from fastapi import HTTPException, Request, status

from cowork.common.settings.app_settings import OAuthSettings

_TIMEOUT_SECONDS = 10.0


def _auth_base_url(settings: OAuthSettings) -> str:
    base = settings.auth_service_base_url
    if not base:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OAuth is not configured for this deployment (auth_service_base_url unset).",
        )
    return base.rstrip("/")


def _forwarded_headers(request: Request) -> dict[str, str]:
    # Every request reaching here already passed cowork-server's own
    # identity enforcement (TrustedHeaderMiddleware), so a missing
    # Authorization header would be a caller bug, not something to paper
    # over here — let auth reject it with its own 401.
    auth_header = request.headers.get("authorization")
    return {"authorization": auth_header} if auth_header else {}


async def _relay(
    method: str,
    path: str,
    *,
    request: Request,
    settings: OAuthSettings,
    params: dict | None = None,
    json_body: dict | None = None,
) -> dict:
    url = f"{_auth_base_url(settings)}{path}"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            resp = await client.request(
                method, url, params=params, json=json_body, headers=_forwarded_headers(request),
            )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=f"auth service unreachable: {exc}",
        ) from exc
    if resp.status_code >= 400:
        # Relay auth's own status + detail verbatim (e.g. 401 on a bad/expired
        # credential, 404 on an unknown service, 403 on needs_reconnect) rather
        # than collapsing everything to a generic error.
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail=detail)
    return resp.json()


async def proxy_start(service: str, request: Request, settings: OAuthSettings, body: dict) -> dict:
    return await _relay(
        "POST", f"/v1/oauth/{service}/start", request=request, settings=settings, json_body=body,
    )


async def proxy_status(state: str, request: Request, settings: OAuthSettings) -> dict:
    return await _relay(
        "GET", "/v1/oauth/status", request=request, settings=settings, params={"state": state},
    )


async def proxy_catalogue(request: Request, settings: OAuthSettings) -> dict:
    # auth's counterpart to OAuthService.get_catalogue() — same shape
    # ({"items": [...]}), scoped server-side to the caller's own org/user
    # via the forwarded credential rather than a query param.
    return await _relay("GET", "/v1/oauth/catalogue", request=request, settings=settings)


async def proxy_picked_files(engine: str, name: str, files: list[dict], request: Request, settings: OAuthSettings) -> list[dict]:
    """Merge Google-Picker-granted files into a connection's persisted
    picked-files list, via auth's Data Vault (org mode has no durable local
    vault to write this to). Same shape/semantics as the local
    ConnectionsService.merge_picked_files this replaces — dedup by file id,
    union the `projects` list on conflict — just executed by auth."""
    result = await _relay(
        "PATCH", f"/v1/oauth/{engine}/{name}/picked-files",
        request=request, settings=settings, json_body={"files": files},
    )
    return result.get("files", [])


async def proxy_token(engine: str, request: Request, settings: OAuthSettings) -> dict:
    """Mint a live access token for `engine` via auth's turn-key endpoint,
    reused here (not just by anton) since its authorization is deliberately
    generic — "does an active connection exist for the resolved user/org."
    Used by the Google Drive Picker route."""
    return await _relay(
        "POST", f"/v1/oauth/{engine}/token", request=request, settings=settings, json_body={},
    )
