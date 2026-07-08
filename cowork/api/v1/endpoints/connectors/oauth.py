from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Query, status
from fastapi.responses import HTMLResponse

from cowork.common.settings.app_settings import ConnectorSettings, OAuthSettings
from cowork.schemas.connectors import OAuthStartRequest, OAuthStartResponse
from cowork.services.connectors.oauth.config import GOOGLE_SERVICES
from cowork.services.connectors.oauth.google import google_service

router = APIRouter()


@router.post("/{service}/start", response_model=OAuthStartResponse, response_model_by_alias=True)
def start_oauth(service: str, body: OAuthStartRequest = Body(default_factory=OAuthStartRequest)):
    if service not in GOOGLE_SERVICES:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown OAuth service: {service!r}")
    return google_service.start(service, OAuthSettings(), client_id=body.client_id, client_secret=body.client_secret, extra_fields=body.extra_fields)


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


@router.get("/{service}/status")
def oauth_status(service: str):
    if service not in GOOGLE_SERVICES:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown OAuth service: {service!r}")
    from cowork.services.connectors.oauth.state import OAuthStateStore
    store = OAuthStateStore(OAuthSettings().state_path)
    # Compare recency of lastSuccessAt vs lastError
    # We load raw data so we have lastErrorAt and lastSuccessAt
    data = store._load().get(service, {})
    pending = data.get("pending", {})
    last_success = data.get("lastSuccessAt", "")
    last_error_time = data.get("lastErrorAt", "")
    
    is_connected = False
    if last_success and (not last_error_time or last_success > last_error_time):
        is_connected = True
        
    return {
        "pending": bool(pending),
        "connected": is_connected,
        "connection_name": data.get("connectionName", ""),
        "error": data.get("lastError", "") if not is_connected else ""
    }
