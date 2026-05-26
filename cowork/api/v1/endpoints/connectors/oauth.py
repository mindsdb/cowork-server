from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import HTMLResponse

from cowork.common.settings.app_settings import OAuthSettings
from cowork.schemas.connectors import OAuthStartResponse
from cowork.services.connectors.oauth.config import GOOGLE_SERVICES
from cowork.services.connectors.oauth.google import google_service

router = APIRouter()


@router.post("/{service}/start", response_model=OAuthStartResponse)
def start_oauth(service: str):
    if service not in GOOGLE_SERVICES:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown OAuth service: {service!r}")
    return google_service.start(service, OAuthSettings())


@router.get("/{service}/callback", response_class=HTMLResponse)
def oauth_callback(service: str, code: str = "", state: str = "", error: str = ""):
    if service not in GOOGLE_SERVICES:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown OAuth service: {service!r}")
    html = google_service.callback(service, code, state, error, OAuthSettings())
    return HTMLResponse(content=html)
