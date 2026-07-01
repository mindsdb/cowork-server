from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from cowork.common.settings.app_settings import ConnectorSettings
from cowork.schemas.connectors import ConnectionDetailResponse, ConnectionSummaryResponse, DirectSaveRequest
from cowork.services.connectors.connections import service
from cowork.services.connectors.oauth.google import google_service
from cowork.services.connectors.persist import persist_connection
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
    token exchange already succeeded. Calls verify_connection before saving."""
    if registry.get_connector(body.connector_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown connector: {body.connector_id}")
    access_token = body.values.get("access_token", "")
    if access_token:
        try:
            google_service.verify_connection(body.connector_id, access_token)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    values = dict(body.values)
    # Best-effort: capture the authenticated account email so the connection is
    # identifiable (drives a readable, dedup-able slug). Absent when the email
    # scope wasn't granted — persist_connection then falls back to a random
    # slug, so this never blocks the save.
    if access_token and not values.get("account_email"):
        email = google_service.account_email(access_token)
        if email:
            values["account_email"] = email
    if values.get("access_token") or values.get("refresh_token"):
        values["auth_type"] = "oauth"
    try:
        slug = persist_connection(body.connector_id, body.method, body.name, values)
    except Exception:
        _log.exception("Failed to save connection %s", body.connector_id)
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


# Fields written to vault by this endpoint. refresh_token is intentionally
# excluded — it is stored in the OS keychain by Electron Main, never in vault.
_PATCH_TOKEN_VAULT_FIELDS = {"access_token", "expires_at", "status"}


class PatchTokenBody(BaseModel):
    access_token: str | None = None
    expires_at: str | None = None
    refresh_token: str | None = None  # accepted from Electron but not persisted to vault
    status: str | None = None


@router.patch("/{engine}/{name}/token")
def patch_connection_token(engine: str, name: str, body: PatchTokenBody):
    """Partially update token fields on a vault entry.

    Called by Electron Main after a successful token refresh (access_token +
    expires_at) or to mark a connection as needs_reconnect (status). refresh_token
    is accepted in the request body but never written to the vault — it is stored
    exclusively in the OS keychain by Electron Main.

    Returns 404 if the vault entry does not exist (connection was deleted while
    Electron was mid-refresh; caller should discard silently).
    """
    updates = {
        k: v for k, v in body.model_dump().items()
        if v is not None and k in _PATCH_TOKEN_VAULT_FIELDS
    }
    if not updates:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one of access_token, expires_at, or status is required.",
        )
    if not service.patch_token(engine, name, updates):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found.")
    return {"ok": True}
