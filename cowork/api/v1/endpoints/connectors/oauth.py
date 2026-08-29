from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse

from cowork.api.v1.endpoints.guards import require_local
from cowork.common.settings.app_settings import ConnectorSettings, OAuthSettings
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.schemas.connectors import OAuthStartRequest, OAuthStartResponse
from cowork.services.connectors.oauth import auth_proxy
from cowork.services.connectors.oauth.config import OAUTH_SERVICES
from cowork.services.connectors.oauth.google import (
    _ENGINE_TO_SERVICE,
    _SERVICE_CREDENTIAL_ATTRS,
    _credentials_complete,
    oauth_service,
)

router = APIRouter()

# Same alias as connections.py: the vault/relay choice is per-request tenancy
# context, not a bare settings flag — resolving it once here keeps this file
# from growing a second, independent way to answer "is this org mode" (a
# standalone _org_mode() used to live here, computing the same fact connections.py
# already resolved via DI, with nothing keeping the two in sync if TenantScope's
# derivation ever grows an extra condition).
ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


@router.post("/{service}/start", response_model=OAuthStartResponse, response_model_by_alias=True)
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


@router.get("/{engine}/credentials")
def get_oauth_credentials(engine: str, request: Request):
    """Return client_id and client_secret for a builtin-OAuth engine.
    Called by Electron main process only — never exposed to the renderer."""
    # Returns a raw client_secret — same loopback restriction as the settings
    # reveal-key and /raw endpoints (ENG-868).
    require_local(request)
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


@router.get("/catalogue")
async def oauth_catalogue(request: Request, scope: ScopeDep):
    if scope.org_mode:
        return await auth_proxy.proxy_catalogue(request, OAuthSettings())
    return {"items": oauth_service.get_catalogue(ConnectorSettings(), OAuthSettings(), scope=scope)}


@router.get("/status")
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


@router.get("/{service}/callback", response_class=HTMLResponse)
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


@router.post("/{engine}/picker/token")
async def mint_picker_token(engine: str, request: Request, scope: ScopeDep, body: dict = Body(default_factory=dict)):
    """Org-mode only. Mints the live access token directly into the JSON
    response.

    Replaces the old two-step session handoff (mint at POST time, hand off
    via an opaque Redis-backed ticket to a header-less `GET .../picker`
    popup navigation, which could never carry a Bearer header). The Picker
    now renders in-page in the SPA itself (see `pickDriveFiles`/host.ts) —
    everything here is a normal authenticated `fetch()`, so there is no
    header-less hop this token needs to survive, and nothing about it is
    ever served to an unauthenticated request.

    `file_ids` (pre-navigating the picker to specific files) is no longer
    threaded through this endpoint at all: the caller already has that value
    itself and can hand it straight to the Picker widget's own JS builder,
    with no server round trip needed."""
    _require_picker_engine(engine, org_mode=scope.org_mode)

    settings = OAuthSettings()
    token = await auth_proxy.proxy_token(engine, request, settings, name=body.get("name") or "")

    access_token = token.get("access_token")
    account_email = token.get("account_email") or body.get("account_email", "")
    api_key = token.get("picker_api_key") or settings.google_picker_api_key
    app_id = token.get("app_id", "")
    if not access_token or not api_key or not app_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google Drive Picker is not fully configured for this deployment.",
        )

    return {
        "access_token": access_token,
        "account_email": account_email,
        "api_key": api_key,
        "app_id": app_id,
    }
