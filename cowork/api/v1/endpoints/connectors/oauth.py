from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse

from cowork.db.scoped import TenantScope, get_tenant_scope

from cowork.api.v1.endpoints.guards import require_local
from cowork.common.settings.app_settings import ConnectorSettings, OAuthSettings
from cowork.schemas.connectors import OAuthStartRequest, OAuthStartResponse
from cowork.services.connectors.oauth.config import OAUTH_SERVICES
from cowork.services.connectors.oauth.google import (
    _ENGINE_TO_SERVICE,
    _SERVICE_CREDENTIAL_ATTRS,
    _credentials_complete,
    oauth_service,
)

router = APIRouter()


@router.post("/{service}/start", response_model=OAuthStartResponse, response_model_by_alias=True)
def start_oauth(service: str, body: OAuthStartRequest = Body(default_factory=OAuthStartRequest)):
    if service not in OAUTH_SERVICES:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown OAuth service: {service!r}")
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
def oauth_catalogue(scope: TenantScope = Depends(get_tenant_scope)):
    return {"items": oauth_service.get_catalogue(ConnectorSettings(), OAuthSettings(), scope=scope)}


@router.get("/status")
def oauth_status(state: str = Query(...)):
    settings = OAuthSettings()
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
