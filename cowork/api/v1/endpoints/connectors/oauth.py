from __future__ import annotations

import json
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from cowork.api.v1.permissions import AuthenticatedInOrgMode, LoopbackOnly, OpenByDesign, require
from cowork.common.settings.app_settings import ConnectorSettings, OAuthSettings
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.schemas.connectors import OAuthStartRequest, OAuthStartResponse, PickerTokenResponse
from cowork.services.connectors.oauth import auth_proxy
from cowork.services.connectors.oauth.config import OAUTH_SERVICES
from cowork.services.connectors.oauth.google import (
    _ENGINE_TO_SERVICE,
    _SERVICE_CREDENTIAL_ATTRS,
    _credentials_complete,
    oauth_service,
)

router = APIRouter()
_log = logging.getLogger("cowork.connectors.oauth")

# Engines whose identity can only be resolved via an MCP tool call, not a
# plain REST/GraphQL request Electron can make on its own — see
# get_mcp_identity below. Only HubSpot today; kept as a set (not a bare
# `engine == "hubspot"` check) since Stage 1 built this as general MCP
# infrastructure and Linear/PostHog both already run their own MCP servers.
_MCP_IDENTITY_ENGINES = {"hubspot"}

# Same alias as connections.py: the vault/relay choice is per-request tenancy
# context, not a bare settings flag — resolving it once here keeps this file
# from growing a second, independent way to answer "is this org mode" (a
# standalone _org_mode() used to live here, computing the same fact connections.py
# already resolved via DI, with nothing keeping the two in sync if TenantScope's
# derivation ever grows an extra condition).
ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


# AuthenticatedInOrgMode: in org mode this forwards to auth_proxy.proxy_start,
# which relies on the caller already being identified — there's no DB check
# of its own to fall back on here.
# Confirmed: auth's own /v1/oauth/{service}/start view requires authentication
# (IsAuthenticated) independently of anything cowork-server does.
@router.post(
    "/{service}/start",
    response_model=OAuthStartResponse,
    response_model_by_alias=True,
    dependencies=[Depends(require(AuthenticatedInOrgMode))],
)
async def start_oauth(service: str, request: Request, scope: ScopeDep,
                       body: OAuthStartRequest = Body(default_factory=OAuthStartRequest)):
    if service not in OAUTH_SERVICES:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown OAuth service: {service!r}")
    if scope.org_mode:
        # auth runs the actual PKCE handshake in org mode — no LocalDataVault/
        # OAuthService touched at all on this branch. See cowork-server's
        # OAuth Connector Lifecycle blueprint item.
        return await auth_proxy.proxy_start(
            service, request, OAuthSettings(),
            {"client_id": body.client_id, "client_secret": body.client_secret, "extra_fields": body.extra_fields},
        )
    return oauth_service.start(service, OAuthSettings(), client_id=body.client_id, client_secret=body.client_secret, extra_fields=body.extra_fields)


# LoopbackOnly: returns a raw client_secret, so the credential is the caller
# being on this machine — same restriction as settings reveal-key and /raw
# (ENG-868).
@router.get("/{engine}/credentials", dependencies=[Depends(require(LoopbackOnly))])
def get_oauth_credentials(engine: str):
    """Return client_id and client_secret for a builtin-OAuth engine.
    Called by Electron main process only — never exposed to the renderer."""
    service_id = _ENGINE_TO_SERVICE.get(engine)
    if service_id is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown OAuth engine: {engine!r}")
    id_attr, secret_attr = _SERVICE_CREDENTIAL_ATTRS[service_id]
    settings = OAuthSettings()
    client_id = getattr(settings, id_attr, "")
    # `secret_attr` is `None` for public, PKCE-only providers (PostHog) — no
    # client_secret exists, so an empty string here means "correctly has
    # none", not "not configured".
    client_secret = getattr(settings, secret_attr, "") if secret_attr else ""
    if not _credentials_complete(client_id, client_secret, secret_attr):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"OAuth credentials not configured for {engine!r}.")
    response = {"client_id": client_id, "client_secret": client_secret}
    if OAUTH_SERVICES[service_id].uses_picker and settings.google_picker_api_key:
        response["picker_api_key"] = settings.google_picker_api_key
    return response


class McpIdentityRequest(BaseModel):
    access_token: str


def _first_str(*values: Any) -> str:
    for v in values:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _parse_mcp_identity(user_details: Any, org_details: Any) -> tuple[str, str]:
    """Extraction of {account_email, account_name} out of
    get_user_details/get_organization_details' real JSON shapes.

    Live-verified 2026-09-15 against a real HubSpot MCP server (Stage 2
    testing) — both nest the useful fields one level down, not at the top
    level as originally guessed during Stage 1:

        get_user_details:         {"userInformation": {"email": ..., "firstName": ..., "lastName": ...}, ...}
        get_organization_details: {"accountInformation": {"portalName": ..., ...}, ...}

    `account_name` prefers the HubSpot portal name over the connecting
    person's own name — HubSpot connections are portal-scoped (a whole team
    can share one), so the portal is the more useful "account" identity for
    the connection tile, the same reasoning Supabase/Linear/PostHog already
    use their org/workspace name for.

    Tolerant of the tool result coming back as a JSON string (the common
    case — MCP text-content blocks) or an already-decoded dict.
    """

    def _as_dict(value: Any) -> dict:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except ValueError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    user = _as_dict(user_details)
    user_info = _as_dict(user.get("userInformation"))
    org = _as_dict(org_details)
    account_info = _as_dict(org.get("accountInformation"))

    email = _first_str(user_info.get("email"))
    person_name = _first_str(
        f"{user_info.get('firstName', '')} {user_info.get('lastName', '')}".strip(),
    )
    name = _first_str(account_info.get("portalName"), person_name)
    return email, name


# LoopbackOnly: same restriction as /credentials above — this hands back
# account identity derived from the caller-supplied access token, so the
# credential is the caller being on this machine. Called by Electron's
# oauth-identity.ts (fetchHubspotIdentity) — Electron can't speak MCP
# itself (it's TypeScript, the client is Python), so it hands the
# freshly-obtained access token here instead of calling the provider
# directly, unlike every other FETCHERS entry.
@router.post("/{engine}/mcp/identity", dependencies=[Depends(require(LoopbackOnly))])
async def get_mcp_identity(engine: str, body: McpIdentityRequest):
    """Resolve {account_email, account_name} for an MCP-based connector by
    calling its `get_user_details`/`get_organization_details` tools directly
    (anton's lightweight `call_mcp_tool` helper — cowork-server already
    depends on anton as a package, so this reuses its client/server-URL
    table rather than a second MCP implementation here)."""
    if engine not in _MCP_IDENTITY_ENGINES:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No MCP identity resolution for {engine!r}.")
    from anton.core.mcp.wiring import call_mcp_tool

    try:
        user_details = await call_mcp_tool(engine, body.access_token, "get_user_details")
    except Exception as exc:
        _log.warning("MCP identity resolution failed for %s: %s", engine, exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Could not resolve {engine} account identity.") from exc

    # Separate try/except from get_user_details above: account_name is only
    # ever a nice-to-have (see the account_email check below), so a portal
    # where get_organization_details errors — unsupported tool, insufficient
    # scope, tier-gating — must not throw away an already-successful email.
    try:
        org_details = await call_mcp_tool(engine, body.access_token, "get_organization_details")
    except Exception as exc:
        _log.warning("MCP organization lookup failed for %s (continuing with email only): %s", engine, exc)
        org_details = None

    account_email, account_name = _parse_mcp_identity(user_details, org_details)
    if not account_email:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Could not retrieve {engine} account email.")
    return {"account_email": account_email, "account_name": account_name}


# AuthenticatedInOrgMode: in org mode this forwards to auth_proxy.proxy_catalogue.
# Confirmed: auth's own /v1/oauth/catalogue view requires authentication
# (IsAuthenticated) independently of anything cowork-server does.
@router.get("/catalogue", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def oauth_catalogue(request: Request, scope: ScopeDep):
    if scope.org_mode:
        return await auth_proxy.proxy_catalogue(request, OAuthSettings())
    return {"items": oauth_service.get_catalogue(ConnectorSettings(), OAuthSettings(), scope=scope)}


# AuthenticatedInOrgMode: in org mode this forwards to auth_proxy.proxy_status.
# Confirmed: auth's own /v1/oauth/status view requires authentication
# (IsAuthenticated) independently of anything cowork-server does.
@router.get("/status", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def oauth_status(request: Request, scope: ScopeDep, state: str = Query(...)):
    settings = OAuthSettings()
    if scope.org_mode:
        return await auth_proxy.proxy_status(state, request, settings)
    outcome = oauth_service.get_outcome(state, settings)
    if outcome is None:
        return {"status": "expired"}
    if outcome.get("status") in {"success", "error"}:
        oauth_service.clear_outcome(state, settings)
    return outcome


# OpenByDesign, standalone reason: this is the OAuth provider's own redirect
# target (Google/GitHub send the user's browser here with code/state/error)
# — there is no principal to check by construction, same as auth's own
# OAuthCallbackView.
@router.get("/{service}/callback", response_class=HTMLResponse, dependencies=[Depends(require(OpenByDesign))])
def oauth_callback(service: str, code: str = "", state: str = "", error: str = ""):
    if service not in OAUTH_SERVICES:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown OAuth service: {service!r}")
    html = oauth_service.callback(service, code, state, error, OAuthSettings())
    return HTMLResponse(content=html)


def _require_picker_engine(engine: str, *, org_mode: bool) -> None:
    if not org_mode:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not available outside org deployments")
    service_id = _ENGINE_TO_SERVICE.get(engine)
    if service_id is None or not OAUTH_SERVICES[service_id].uses_picker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No file picker for engine {engine!r}")


# AuthenticatedInOrgMode: org-mode-only surface (_require_picker_engine 404s
# outside org mode), no DB check of its own.
@router.post("/{engine}/picker/session", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def create_picker_session(engine: str, scope: ScopeDep):
    """Gone: the picker is built in the SPA now, so there is no session to
    mint. Answers 410 rather than 404 so a tab still running the previous
    bundle tells the user to reload instead of reporting a picker failure."""
    _require_picker_engine(engine, org_mode=scope.org_mode)
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="Cowork has been updated — reload the page to add Google Drive files.",
    )


# AuthenticatedInOrgMode: forwards to auth_proxy.proxy_token, no DB check of
# its own.
# Confirmed: auth's own /v1/oauth/{engine}/token view requires authentication
# (IsAuthenticated) independently of anything cowork-server does.
@router.post(
    "/{engine}/picker/token",
    response_model=PickerTokenResponse,
    dependencies=[Depends(require(AuthenticatedInOrgMode))],
)
async def mint_picker_token(engine: str, request: Request, scope: ScopeDep, body: dict = Body(default_factory=dict)):
    """Org-mode only. Returns a live Drive access token to the caller's own
    authenticated fetch — safe because nothing here is reachable without the
    caller's Bearer, and the token never leaves that response."""
    _require_picker_engine(engine, org_mode=scope.org_mode)

    settings = OAuthSettings()
    token = await auth_proxy.proxy_token(engine, request, settings, name=body.get("name") or "")

    access_token = token.get("access_token")
    api_key = token.get("picker_api_key") or settings.google_picker_api_key
    app_id = token.get("app_id", "")
    if not access_token or not api_key or not app_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google Drive Picker is not fully configured for this deployment.",
        )

    return PickerTokenResponse(
        access_token=access_token,
        account_email=token.get("account_email", ""),
        api_key=api_key,
        app_id=app_id,
    )
