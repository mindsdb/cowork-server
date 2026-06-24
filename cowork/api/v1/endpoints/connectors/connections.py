from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from cowork.common.settings.app_settings import ConnectorSettings, OAuthSettings
from cowork.schemas.connectors import (
    ConnectionDetailResponse,
    ConnectionHealthResponse,
    ConnectionSummaryResponse,
    DirectSaveRequest,
    ReconnectInfoResponse,
    TestConnectionResponse,
)
from cowork.services.connectors import health as health_mod
from cowork.services.connectors.connections import service
from cowork.services.connectors.oauth.config import GOOGLE_SERVICES
from cowork.services.connectors.oauth.google import google_service
from cowork.services.connectors.specs._registry import registry

_log = logging.getLogger("cowork.connectors.connections")
router = APIRouter()

# engine name (e.g. "google_drive") → OAuth service id (e.g. "google-drive").
# Mirrors the map in oauth.google; rebuilt here to avoid importing a private.
_ENGINE_TO_SERVICE = {cfg.engine: svc for svc, cfg in GOOGLE_SERVICES.items()}


@router.get("/", response_model=list[ConnectionSummaryResponse])
def list_connections():
    return service.list()


@router.get("/{engine}/{name}", response_model=ConnectionDetailResponse)
def get_connection(engine: str, name: str):
    record = service.get(engine, name)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found.")
    return record


@router.post("/{engine}/{name}/test", response_model=TestConnectionResponse)
async def test_connection(engine: str, name: str):
    """Re-run the connection probe against the saved credentials.

    Loads the stored (decrypted) credentials, runs the same headless Anton
    probe used at submit time, returns pass/fail + the real error, and stamps
    ``last_tested_at`` / ``last_test_result`` on the record. The credential
    ciphertext is never rewritten — only the plaintext test metadata.
    """
    from cowork.services.connectors.test_runner import run_test

    record = service.read_record(engine, name)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found.")

    credentials = dict(record.get("fields") or {})
    result = await run_test(engine, name, credentials)
    # Persist the verdict so the list view's health badge reflects it — but
    # ONLY for a real pass/fail. An empty result means "couldn't test" (no
    # live probe for this connector, or the probe couldn't start); stamping
    # that as a failure would wrongly mark the connection broken forever.
    if result.result in (health_mod.TEST_PASS, health_mod.TEST_FAIL):
        try:
            service.record_test_result(
                engine, name,
                result=result.result,
                error=result.error if not result.ok else None,
            )
            result.tested_at = (service.read_record(engine, name) or {}).get("last_tested_at")
        except Exception:
            _log.exception("Could not persist test result for %s/%s", engine, name)
    return result


@router.get("/{engine}/{name}/health", response_model=ConnectionHealthResponse)
def connection_health(engine: str, name: str):
    """Provider-agnostic health verdict for a saved connection (no network)."""
    record = service.read_record(engine, name)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found.")
    fields = dict(record.get("fields") or {})
    last_test_result = record.get("last_test_result")
    status_obj = health_mod.compute_health(fields, last_test_result=last_test_result)
    return ConnectionHealthResponse(
        engine=engine,
        name=name,
        health=status_obj.status,
        detail=status_obj.detail,
        reconnectable=status_obj.reconnectable,
        is_oauth=health_mod.is_oauth(fields),
        expires_at=status_obj.expires_at,
        last_tested_at=record.get("last_tested_at"),
        last_test_result=last_test_result,
    )


@router.post("/{engine}/{name}/reconnect", response_model=ReconnectInfoResponse)
def reconnect_connection(engine: str, name: str):
    """Reconnect entry point for a saved connection.

    For OAuth connections we first try a silent token refresh — if the
    connection still holds a valid refresh token that recovers it without any
    user interaction, and we report ``refreshed=True`` so the UI can skip the
    browser flow. Otherwise we return the reconnect ``method`` (``oauth`` vs
    ``credentials``) and, for OAuth, the ``service`` id the client uses to kick
    off the interactive flow (``POST /connectors/oauth/{service}/start``).

    Non-OAuth connections reconnect by re-entering credentials (the modify
    flow the client already drives), so we just report ``method=credentials``.
    """
    record = service.read_record(engine, name)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found.")

    fields = dict(record.get("fields") or {})
    if not health_mod.is_oauth(fields):
        return ReconnectInfoResponse(
            engine=engine, name=name, method="credentials",
            health=health_mod.compute_health(fields, last_test_result=record.get("last_test_result")).status,
            message="Re-enter this connection's credentials to reconnect.",
        )

    service_id = _ENGINE_TO_SERVICE.get(engine)
    refreshed = False
    if service_id:
        try:
            refreshed = google_service.refresh_one(engine, name, OAuthSettings())
        except Exception:
            _log.exception("Silent reconnect refresh failed for %s/%s", engine, name)

    # Recompute health off the (possibly refreshed) record.
    fresh = service.read_record(engine, name) or record
    health = health_mod.compute_health(
        dict(fresh.get("fields") or {}), last_test_result=fresh.get("last_test_result")
    ).status
    return ReconnectInfoResponse(
        engine=engine, name=name, method="oauth", service=service_id,
        refreshed=refreshed, health=health,
        message=(
            "Reconnected — the access token was refreshed."
            if refreshed
            else "Sign in again to reconnect this account."
        ),
    )


@router.post("/save")
def save_connection_direct(body: DirectSaveRequest):
    """Persist credentials to the vault without running a probe.
    Used after an OAuth PKCE flow (Electron main-process PKCE) where the
    token exchange already succeeded. Calls verify_connection before saving."""
    if registry.get_connector(body.connector_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown connector: {body.connector_id}")
    from cowork.services.connectors.encrypted_vault import build_vault
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
        build_vault(Path(ConnectorSettings().vault_dir)).save(body.connector_id, slug, payload)
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
