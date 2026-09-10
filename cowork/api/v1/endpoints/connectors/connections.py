from __future__ import annotations

import json
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from cowork.api.v1.endpoints.guards import require_local
from cowork.api.v1.permissions import AuthenticatedInOrgMode, OpenByDesign, require
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
from cowork.services.connectors.oauth import auth_proxy
from cowork.services.connectors.oauth.google import oauth_service
from cowork.services.connectors.persist import persist_connection
from cowork.services.connectors.specs._registry import registry

_log = logging.getLogger("cowork.connectors.connections")
router = APIRouter()

# Every route resolves the tenant scope: the vault is org-keyed, the same as
# SkillService/skills.py, and must never be read/written through the unscoped
# module-level `service` singleton on an org request.
ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


# AuthenticatedInOrgMode: in org mode this forwards to auth_proxy.proxy_catalogue,
# no DB check of its own.
# Confirmed: auth's own /v1/oauth/catalogue view requires authentication
# (IsAuthenticated) independently of anything cowork-server does.
@router.get(
    "/", response_model=list[ConnectionSummaryResponse], dependencies=[Depends(require(AuthenticatedInOrgMode))]
)
async def list_connections(scope: ScopeDep, request: Request):
    if scope.org_mode:
        # No durable local vault to read in org mode — same forwarded-
        # credential proxy as the OAuth Connector Lifecycle endpoints.
        # auth's catalogue already carries every connection (it's the same
        # data ConnectWorkflowView's org-mode connector list reads), so no
        # separate "list connections" endpoint is needed on auth's side.
        catalogue = await auth_proxy.proxy_catalogue(request, OAuthSettings())
        return [
            ConnectionSummaryResponse(engine=c["engine"], name=c["name"], user_label=c.get("user_label"))
            for item in catalogue.get("items", [])
            for c in item.get("connections", [])
        ]
    return ConnectionsService(scope).list()


# Non-secret fields surfaced from auth's connection-detail response —
# mirrors anton's TurnKeyDataVault._TURNKEY_RESPONSE_FIELDS allowlist (minus
# access_token, which this read-only detail view never needs to hold).
# `status`/`_picked_files` are handled separately below, not through this
# passthrough allowlist: `status` because "" is a valid absent-value here
# but "active"/"needs_reconnect" are never falsy, so the shared `if
# detail.get(k)` guard is fine for it too; `_picked_files` because its shape
# has to change (see below) rather than being a straight passthrough.
_ORG_CONNECTION_DETAIL_FIELDS = ("account_email", "token_type", "scope", "expires_at", "status")


# AuthenticatedInOrgMode: in org mode this forwards to
# auth_proxy.proxy_connection_detail, no DB check of its own.
# Confirmed: auth's own /v1/oauth/{engine}/{name} view requires authentication
# (IsAuthenticated) independently of anything cowork-server does.
@router.get(
    "/{engine}/{name}", response_model=ConnectionDetailResponse, dependencies=[Depends(require(AuthenticatedInOrgMode))]
)
async def get_connection(engine: str, name: str, scope: ScopeDep, request: Request):
    if scope.org_mode:
        # ENG-2097: this used to call proxy_token, whose fixed response
        # shape never carried _picked_files — files persisted correctly via
        # patch_picked_files below, they just had no read path back to this
        # panel. proxy_connection_detail is a real read of the stored row
        # instead (no refresh/provider call as a side effect), and as a
        # bonus also surfaces a stuck connection's real status instead of
        # proxy_token's 403 — closing the "known limitation" this comment
        # used to describe.
        detail = await auth_proxy.proxy_connection_detail(engine, name, request, OAuthSettings())
        fields = {k: detail[k] for k in _ORG_CONNECTION_DETAIL_FIELDS if detail.get(k)}
        # Desktop's local vault stores this as a JSON-encoded string
        # (ConnectionsService._load_picked_files/merge_picked_files) —
        # CustomizeView.jsx's JSON.parse(fields._picked_files) expects that
        # same shape, not the native list auth's Data Vault holds it as.
        fields["_picked_files"] = json.dumps(detail.get("picked_files") or [])
        return ConnectionDetailResponse(engine=engine, name=name, fields=fields)
    record = ConnectionsService(scope).get(engine, name)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found.")
    return record


# OpenByDesign, standalone reason: what protects this route is require_local
# below — confirmed the only callers are Electron's main process and a
# renderer path gated behind !host.isWeb, both loopback-only.
@router.post(
    "/save",
    response_model=DirectSaveResponse,
    dependencies=[Depends(require_local), Depends(require(OpenByDesign))],
)
def save_connection_direct(body: DirectSaveRequest, scope: ScopeDep):
    """Persist credentials to the vault without running a probe.
    Used after an OAuth PKCE flow (Electron main-process PKCE) where the
    token exchange already succeeded. Electron verifies the token and resolves
    account_email (and, where the provider's identity fetcher in
    cowork/src/main/oauth-identity.ts supplies one, account_name) before
    calling this endpoint."""
    return _persist_direct_connection(body, scope, dict(body.values))


# OpenByDesign, defensive classification: an antontron caller audit found no
# caller anywhere (not the client, not cowork-server's own submissions flow,
# which persists developer credentials by calling the service functions
# directly rather than over HTTP to this route). Classified loopback-only to
# match its sibling /save on the theory that if this is reachable at all, it
# is by the same Electron-only flow — TODO: confirm whether this route is
# actually dead and can be removed, or find its real caller.
@router.post(
    "/validate-and-save",
    response_model=DirectSaveResponse,
    dependencies=[Depends(require_local), Depends(require(OpenByDesign))],
)
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
        # default_label gives a brand-new OAuth connection's tile a
        # meaningful title (the account/org/workspace name the provider
        # returned) instead of the generic engine-id default — but only for
        # a genuinely new connection; it can never clobber a label the user
        # already set on a reconnect (see persist_connection's default_label
        # docs). Mirrors the same wiring in oauth/google.py's callback(),
        # the analogous save path for a non-Electron (web) OAuth flow.
        slug = persist_connection(
            body.connector_id,
            body.method,
            body.name,
            values,
            replace_existing=body.replace_existing,
            default_label=str(values.get("account_name") or "").strip() or None,
            vault=vault,
        )
    except Exception:
        _log.exception("Failed to save connection %s", body.connector_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save connection.")
    saved = vault.read_record(body.connector_id, slug) or {}
    user_label = str(saved.get("fields", {}).get("_user_label", "")).strip() or None
    return {"ok": True, "name": slug, "label": slug, "user_label": user_label}


# AuthenticatedInOrgMode: in org mode this forwards to auth_proxy.proxy_delete,
# no DB check of its own.
# Confirmed: auth's own /v1/oauth/{engine}/{name} view requires authentication
# (IsAuthenticated) independently of anything cowork-server does.
@router.delete(
    "/{engine}/{name}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require(AuthenticatedInOrgMode))]
)
async def delete_connection(engine: str, name: str, scope: ScopeDep, request: Request):
    if scope.org_mode:
        # No local vault to delete from in org mode — same forwarded-
        # credential proxy as the other OAuth Connector Lifecycle endpoints.
        await auth_proxy.proxy_delete(engine, name, request, OAuthSettings())
        return
    try:
        # revoke() makes a blocking provider HTTP call (up to its own 10s
        # timeout) - off the event loop, since this handler is now async for
        # the org-mode branch above and FastAPI no longer threadpools it.
        await run_in_threadpool(oauth_service.revoke, engine, name, ConnectorSettings(), OAuthSettings(), scope=scope)
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


# OpenByDesign, standalone reason: what protects this route is require_local
# below — confirmed the only caller is Electron main's token-refresh.ts,
# always loopback (the in-body 501 for org mode, below, is a second,
# independent belt-and-suspenders check).
@router.patch(
    "/{engine}/{name}/token",
    dependencies=[Depends(require_local), Depends(require(OpenByDesign))],
)
def patch_connection_token(engine: str, name: str, body: PatchTokenBody, scope: ScopeDep):
    """Partially update token fields on a vault entry.

    Called by Electron Main after a successful token refresh (access_token +
    expires_at) or to mark a connection as needs_reconnect (status). refresh_token
    is accepted in the request body but never written to the vault — it is stored
    exclusively in the OS keychain by Electron Main.

    Returns 404 if the vault entry does not exist (connection was deleted while
    Electron was mid-refresh; caller should discard silently).

    Desktop-only: org mode's token lifecycle is auth's job — get_valid_access_token
    auto-refreshes on read, there is no client-side refresh loop to report back
    here. Electron main's own token-refresh.ts always targets the local desktop
    sidecar, never cowork-server, so no org-mode caller reaches this route today.
    Guarded explicitly so a future org-mode caller added here by mistake fails
    loudly instead of silently writing to the wrong (non-durable) vault.
    """
    if scope.org_mode:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Token refresh is not applicable in org mode; auth manages the token lifecycle server-side.",
        )
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


# AuthenticatedInOrgMode: in org mode this forwards to auth_proxy.proxy_picked_files,
# no DB check of its own.
# Confirmed: auth's own /v1/oauth/{engine}/{name}/picked-files view requires
# authentication (IsAuthenticated) independently of anything cowork-server does.
@router.patch("/{engine}/{name}/picked-files", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def patch_picked_files(engine: str, name: str, body: PatchPickedFilesBody, scope: ScopeDep, request: Request):
    """Merge Google-Picker-granted files into the connection's persisted
    `_picked_files` list. Called right after the user picks files —
    drive.file scope only covers files the app created, so this is the
    record of what else the user has explicitly granted.

    Org mode: the local vault file this otherwise writes to isn't durable
    or shared across replicas, so the merge happens in auth's Data Vault
    instead — same forwarded-credential proxy mechanism as the OAuth
    Connector Lifecycle endpoints. See cowork-server's Google Drive File
    Picker blueprint item."""
    files = [f.model_dump(by_alias=True, exclude_none=True) for f in body.files]
    if scope.org_mode:
        merged = await auth_proxy.proxy_picked_files(engine, name, files, request, OAuthSettings())
        return {"ok": True, "files": merged}
    merged = ConnectionsService(scope).merge_picked_files(engine, name, files)
    if merged is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found.")
    return {"ok": True, "files": merged}


# AuthenticatedInOrgMode, defensive classification: an antontron caller audit
# confirmed this is reachable from both desktop and hosted web (no isElectron/
# isWeb guard, see useGoogleDrivePicker.js). This declares the identity floor
# only — it does NOT fix the actual gap: unlike patch_picked_files above, this
# route has no org-mode branch and unconditionally hits the local vault via
# ConnectionsService, and auth_proxy has no "remove one picked file" call to
# forward to (only proxy_picked_files's merge/union semantics and proxy_delete
# for the whole connection). TODO: an org-mode caller here likely gets a 404
# against the wrong (local, non-durable) vault instead of actually un-picking
# the file — needs a real auth-service endpoint before it can work correctly
# in org mode, same class of gap as test_providers's stored-key issue.
@router.delete(
    "/{engine}/{name}/picked-files/{file_id}",
    dependencies=[Depends(require(AuthenticatedInOrgMode))],
)
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
