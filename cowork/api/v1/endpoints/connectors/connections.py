from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from cowork.common.settings.app_settings import ConnectorSettings
from cowork.schemas.connectors import ConnectionDetailResponse, ConnectionSummaryResponse, DirectSaveRequest
from cowork.services.connectors.connections import service
from cowork.services.connectors.oauth.google import google_service
from cowork.services.connectors.specs._registry import registry

_log = logging.getLogger("cowork.connectors.connections")
router = APIRouter()


@router.get("/", response_model=list[ConnectionSummaryResponse])
def list_connections():
    return service.list()


@router.get("/{engine}/{name}", response_model=ConnectionDetailResponse)
def get_connection(engine: str, name: str):
    record = service.get(engine, name)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found.")
    return record


@router.post("/save")
def save_connection_direct(body: DirectSaveRequest):
    """Persist credentials to the vault without running a probe.
    Used after an OAuth PKCE flow (Electron main-process PKCE) where the
    token exchange already succeeded. Calls verify_connection before saving —
    token validation is only implemented for Google Drive so far; other Google
    services skip the check until their verify URLs are added to verify_connection."""
    if registry.get_connector(body.connector_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown connector: {body.connector_id}")
    from anton.core.datasources.data_vault import LocalDataVault
    access_token = body.values.get("access_token", "")
    if access_token:
        try:
            google_service.verify_connection(body.connector_id, access_token)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    slug = body.name.strip() or f"{body.connector_id}-{uuid.uuid4().hex[:6]}"
    payload: dict = {**body.values, "_connector_id": body.connector_id}
    if body.method:
        payload["_method"] = body.method
    if body.values.get("access_token") or body.values.get("refresh_token"):
        payload["auth_type"] = "oauth"
    try:
        LocalDataVault(Path(ConnectorSettings().vault_dir)).save(body.connector_id, slug, payload)
    except Exception:
        _log.exception("Failed to save connection %s/%s", body.connector_id, slug)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save connection.")
    return {"ok": True, "name": slug, "label": slug}


@router.delete("/{engine}/{name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_connection(engine: str, name: str):
    try:
        google_service.revoke(engine, name, ConnectorSettings())
    except Exception:
        _log.exception("Failed to revoke token for %s/%s", engine, name)
    if not service.delete(engine, name):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found.")
