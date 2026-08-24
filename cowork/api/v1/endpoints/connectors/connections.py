from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from cowork.common.settings.app_settings import ConnectorSettings, OAuthSettings
from cowork.db.scoped import TenantScope, get_tenant_scope, scoped_storage_root
from cowork.schemas.connectors import (
    ConnectionDetailResponse,
    ConnectionSummaryResponse,
    DirectSaveRequest,
    DirectSaveResponse,
    PatchPickedFilesBody,
)
from cowork.services.connectors.connections import ConnectionsService
from cowork.services.connectors.developer_validation import (
    DeveloperCredentialError,
    DeveloperProviderUnavailable,
    validate_developer_connection,
)
from cowork.services.connectors.oauth.google import oauth_service
from cowork.services.connectors.persist import persist_connection
from cowork.services.connectors.specs._registry import registry

_log = logging.getLogger("cowork.connectors.connections")
router = APIRouter()

# Every route resolves the tenant scope: the vault is org-keyed, the same as
# SkillService/skills.py, and must never be read/written through the unscoped
# module-level `service` singleton on an org request.
ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


@router.get("/", response_model=list[ConnectionSummaryResponse])
def list_connections(scope: ScopeDep):
    return ConnectionsService(scope).list()


@router.get("/{engine}/{name}", response_model=ConnectionDetailResponse)
def get_connection(engine: str, name: str, scope: ScopeDep):
    record = ConnectionsService(scope).get(engine, name)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found.")
    return record


@router.post("/save", response_model=DirectSaveResponse)
def save_connection_direct(body: DirectSaveRequest, scope: ScopeDep):
    """Persist credentials to the vault without running a probe.
    Used after an OAuth PKCE flow (Electron main-process PKCE) where the
    token exchange already succeeded. Electron verifies the token and resolves
    account_email before calling this endpoint."""
    return _persist_direct_connection(body, scope, dict(body.values))


@router.post("/validate-and-save", response_model=DirectSaveResponse)
def validate_and_save_developer_connection(body: DirectSaveRequest, scope: ScopeDep):
    """Validate a Code developer-tool credential before storing it.

    Built-in OAuth has already been verified by its token exchange and keeps
    using ``/save``.  This route is for the personal-token fallback shown when
    a desktop build has no hosted OAuth client configured.
    """
    try:
        identity = validate_developer_connection(body.connector_id, body.method, body.values)
    except DeveloperCredentialError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except DeveloperProviderUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    values = {**body.values, **identity.as_fields()}
    return _persist_direct_connection(body, scope, values)


def _persist_direct_connection(
    body: DirectSaveRequest,
    scope: TenantScope,
    values: dict,
) -> dict[str, object]:
    if registry.get_connector(body.connector_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown connector: {body.connector_id}")
    if body.method in {"browser_oauth_builtin", "oauth"} or values.get("refresh_token"):
        values["auth_type"] = "oauth"
    from pathlib import Path
    from anton.core.datasources.data_vault import LocalDataVault
    vault = LocalDataVault(scoped_storage_root(Path(ConnectorSettings().vault_dir), scope, store="data-vault"))
    try:
        slug = persist_connection(
            body.connector_id,
            body.method,
            body.name,
            values,
            replace_existing=body.replace_existing,
            vault=vault,
        )
    except Exception:
        _log.exception("Failed to save connection %s", body.connector_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save connection.")
    saved = vault.read_record(body.connector_id, slug) or {}
    user_label = str(saved.get("fields", {}).get("_user_label", "")).strip() or None
    return {"ok": True, "name": slug, "label": slug, "user_label": user_label}


@router.delete("/{engine}/{name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_connection(engine: str, name: str, scope: ScopeDep):
    try:
        oauth_service.revoke(engine, name, ConnectorSettings(), OAuthSettings(), scope=scope)
    except Exception:
        _log.exception("Failed to revoke token for %s/%s", engine, name)
    if not ConnectionsService(scope).delete(engine, name):
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
def patch_connection_token(engine: str, name: str, body: PatchTokenBody, scope: ScopeDep):
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
    if not ConnectionsService(scope).patch_token(engine, name, updates):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found.")
    return {"ok": True}


@router.patch("/{engine}/{name}/picked-files")
def patch_picked_files(engine: str, name: str, body: PatchPickedFilesBody, scope: ScopeDep):
    """Merge Google-Picker-granted files into the connection's persisted
    `_picked_files` list. Called by Electron Main right after the user
    picks files — drive.file scope only covers files the app created, so
    this is the record of what else the user has explicitly granted."""
    files = [f.model_dump(by_alias=True, exclude_none=True) for f in body.files]
    merged = ConnectionsService(scope).merge_picked_files(engine, name, files)
    if merged is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found.")
    return {"ok": True, "files": merged}


@router.delete("/{engine}/{name}/picked-files/{file_id}")
def delete_picked_file(engine: str, name: str, file_id: str, project: str, scope: ScopeDep):
    """Untag one file from `project` — the "un-pick" counterpart to
    patch_picked_files, used by the Project files rail's delete action on
    a Drive reference row. Only removes the file from `project`'s rail;
    if the file is tagged to other projects too, it stays visible there
    (see remove_picked_file's docstring)."""
    remaining = ConnectionsService(scope).remove_picked_file(engine, name, file_id, project)
    if remaining is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found.")
    return {"ok": True, "files": remaining}
