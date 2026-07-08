from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Query, status
from fastapi.responses import HTMLResponse

from cowork.common.settings.app_settings import ConnectorSettings, OAuthSettings
from cowork.schemas.connectors import OAuthStartRequest, OAuthStartResponse
from cowork.services.connectors.oauth.config import GOOGLE_SERVICES
from cowork.services.connectors.oauth.google import _ENGINE_TO_SERVICE, _SERVICE_CREDENTIAL_ATTRS, google_service

router = APIRouter()


@router.post("/{service}/start", response_model=OAuthStartResponse, response_model_by_alias=True)
def start_oauth(service: str, body: OAuthStartRequest = Body(default_factory=OAuthStartRequest)):
    if service not in GOOGLE_SERVICES:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown OAuth service: {service!r}")
    return google_service.start(service, OAuthSettings(), client_id=body.client_id, client_secret=body.client_secret, extra_fields=body.extra_fields)


@router.get("/{engine}/credentials")
def get_oauth_credentials(engine: str):
    """Return client_id and client_secret for a builtin-OAuth engine.
    Called by Electron main process only — never exposed to the renderer."""
    service_id = _ENGINE_TO_SERVICE.get(engine)
    if service_id is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown OAuth engine: {engine!r}")
    id_attr, secret_attr = _SERVICE_CREDENTIAL_ATTRS[service_id]
    settings = OAuthSettings()
    client_id = getattr(settings, id_attr, "")
    client_secret = getattr(settings, secret_attr, "")
    if not client_id or not client_secret:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"OAuth credentials not configured for {engine!r}.")
    response = {"client_id": client_id, "client_secret": client_secret}
    if engine == "google_drive" and settings.google_picker_api_key:
        response["picker_api_key"] = settings.google_picker_api_key
    return response


@router.get("/catalogue")
def oauth_catalogue():
    return {"items": google_service.get_catalogue(ConnectorSettings(), OAuthSettings())}


@router.get("/status")
def oauth_status(state: str = Query(...)):
    settings = OAuthSettings()
    outcome = google_service.get_outcome(state, settings)
    if outcome is None:
        return {"status": "expired"}
    if outcome.get("status") in {"success", "error"}:
        google_service.clear_outcome(state, settings)
    return outcome


@router.get("/{service}/callback", response_class=HTMLResponse)
def oauth_callback(service: str, code: str = "", state: str = "", error: str = ""):
    if service not in GOOGLE_SERVICES:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown OAuth service: {service!r}")
    html = google_service.callback(service, code, state, error, OAuthSettings())
    return HTMLResponse(content=html)
