"""Routes the desktop app uses to add custom MCP servers and switch their tools.

These are the only writers of a custom MCP connection's held settings
(``services/connectors/custom_mcp.py`` says why). Every route needs the app
secret and protected mode, is loopback only, and refuses org mode: agent code
on the same machine can call loopback routes with the auth token, but cannot
send the secret, so it can never point a token at another server or switch
a tool on.
"""
from __future__ import annotations

import logging
from typing import Annotated, Any

import httpx2
from anton.core.mcp.egress import check_server_url
from anton.core.mcp.errors import McpEgressBlocked, McpHostUnresolved, McpTokenRejected
from fastapi import APIRouter, Depends, HTTPException, status

from cowork.api.v1.endpoints.connectors.connections import require_app_secret
from cowork.api.v1.permissions import LoopbackDesktopOnly, require
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.schemas.custom_mcp import AllowedToolsRequest, CheckUrlRequest, CreateCustomMcpRequest, CustomMcpSettings
from cowork.services.connectors import app_held
from cowork.services.connectors.custom_mcp import CUSTOM_MCP_ENGINE, is_connection_id
from cowork.services.connectors.persist import (
    ConnectionExistsError,
    own_vault_for_scope,
    own_vault_root,
    persist_connection,
)

_log = logging.getLogger("cowork.connectors.custom_mcp")

router = APIRouter(dependencies=[Depends(require(LoopbackDesktopOnly)), Depends(require_app_secret)])
ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]

MISSING = {"error": "custom_connection_missing"}


def _require_protected(scope: TenantScope) -> None:
    """409 unless this is a desktop server holding the app secret: only then
    does memory hold custom MCP settings and the secret guard these routes."""
    if scope.org_mode or not app_held.active():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="custom_mcp_unavailable")


def canonical_server_url(url: str) -> str:
    """The one form of a custom server URL the app and the server use: https,
    the ASCII host, the port only when not 443, and a path of at least ``/``.
    Raises HTTPException 422 for a URL anton's check refuses."""
    try:
        host, port = check_server_url(url)
    except McpEgressBlocked as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"error": "invalid_url", "message": str(exc)}) from exc
    path = httpx2.URL(url).raw_path.decode("ascii") or "/"
    shown_host = f"[{host}]" if ":" in host else host
    return f"https://{shown_host}{'' if port == 443 else f':{port}'}{path}"


def _record_file_present(scope: TenantScope, name: str) -> bool:
    """Whether a file stands where the vault keeps the record of ``name``.

    anton names it ``<engine>-<name>`` after keeping only word characters and
    dashes, which a checked connection id and ``custom_mcp`` already are. The
    vault folder's entries are compared by name, so no path is ever built
    from ``name``; any other name is no record. A broken link or a folder at
    that name counts as present."""
    if not is_connection_id(name):
        return False
    wanted = f"{CUSTOM_MCP_ENGINE}-{name}"
    try:
        return any(entry.name == wanted for entry in own_vault_root(scope).iterdir())
    except FileNotFoundError:
        return False


def _require_connection(scope: TenantScope, name: str) -> None:
    """404 with its own body when ``name`` is no custom MCP connection of this
    scope, dropping anything still held for it. A record file that exists but
    cannot be read is 503 instead: the app deletes the connection's keychain
    entries on the 404 body, and that must never happen for a connection
    whose record is only unreadable for now."""
    if not is_connection_id(name):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=MISSING)
    if own_vault_for_scope(scope).read_record(CUSTOM_MCP_ENGINE, name) is not None:
        return
    if _record_file_present(scope, name):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"error": "custom_connection_unreadable"})
    app_held.release(own_vault_root(scope), CUSTOM_MCP_ENGINE, name)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=MISSING)


def _held_values(settings: CustomMcpSettings) -> dict[str, Any]:
    """The settings as the held entry stores them, with the server URL in its
    canonical form (422 for one the check refuses)."""
    values = settings.model_dump(exclude={"id"})
    if values.get("server_url") is not None:
        values["server_url"] = canonical_server_url(values["server_url"])
    return values


@router.post("/check-url")
def check_url(body: CheckUrlRequest, scope: ScopeDep) -> dict[str, str]:
    """The canonical form of a server URL the user typed, or 422."""
    _require_protected(scope)
    return {"url": canonical_server_url(body.url)}


@router.post("", status_code=status.HTTP_201_CREATED)
def create_connection(body: CreateCustomMcpRequest, scope: ScopeDep) -> dict[str, str]:
    """Create a custom MCP connection under the id the app drew, and hold its
    settings. 409 when the id exists: ids are never reused."""
    _require_protected(scope)
    values = _held_values(body)
    try:
        # Checked and written under the record lock; stamps the creator.
        persist_connection(CUSTOM_MCP_ENGINE, "mcp", body.id, {}, create_only=True, scope=scope)
    except ConnectionExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="custom_connection_exists") from exc
    root = own_vault_root(scope)
    # A new connection starts from nothing, whatever an earlier one under
    # this id left in memory.
    app_held.release(root, CUSTOM_MCP_ENGINE, body.id)
    app_held.hold_custom(root, body.id, {**values, "allowed_tools": values.get("allowed_tools") or []})
    return {"id": body.id, "server_url": values["server_url"]}


@router.put("/{name}/held")
def put_held(name: str, body: CustomMcpSettings, scope: ScopeDep) -> dict[str, str]:
    """Hold settings and token for an existing connection: at every server
    start, on reconnect and on refresh. 404 with its own body when the
    connection is gone, so the app can tell a deletion from a missing route."""
    _require_protected(scope)
    _require_connection(scope, name)
    entry = app_held.hold_custom(own_vault_root(scope), name, _held_values(body))
    return {"server_url": entry.get("server_url", "")}


@router.put("/{name}/allowed-tools")
def put_allowed_tools(name: str, body: AllowedToolsRequest, scope: ScopeDep) -> dict[str, list[str]]:
    """Replace the tools Anton may use on this connection."""
    _require_protected(scope)
    _require_connection(scope, name)
    entry = app_held.hold_custom(own_vault_root(scope), name, {"allowed_tools": body.tools})
    return {"allowed_tools": entry["allowed_tools"]}


@router.get("/{name}/held-summary")
def held_summary(name: str, scope: ScopeDep) -> dict[str, str]:
    """The label and server URL the detail panel shows, from memory."""
    _require_protected(scope)
    _require_connection(scope, name)
    entry = app_held.held_custom(own_vault_root(scope), name)
    return {"label": entry.get("label", ""), "server_url": entry.get("server_url", "")}


@router.get("/{name}/tools")
async def list_tools(name: str, scope: ScopeDep) -> dict[str, list[dict[str, Any]]]:
    """The server's tools with their read-only label and whether each is on."""
    _require_protected(scope)
    _require_connection(scope, name)
    entry = app_held.held_custom(own_vault_root(scope), name)
    if not entry.get("server_url"):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"error": "not_handed_over"})
    # Loads the MCP SDK; kept out of server start, as oauth.py does for call_mcp_tool.
    from anton.core.mcp.wiring import list_custom_server_tools

    try:
        tools = await list_custom_server_tools(entry["server_url"], entry.get("access_token"), connection_name=name)
    except McpHostUnresolved as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail={"error": "unresolved"}) from exc
    except McpEgressBlocked as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"error": "refused_address"}) from exc
    except McpTokenRejected as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"error": "token_rejected"}) from exc
    except Exception as exc:
        # Any other failure means the server could not be reached; only the
        # error's type is logged, since its text can carry the server's URL.
        _log.warning("listing tools of a custom MCP connection failed: %s", type(exc).__name__)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail={"error": "unreachable"}) from exc
    # Read again after the listing: a switch made while it ran must show as made.
    allowed = set(app_held.held_custom(own_vault_root(scope), name).get("allowed_tools") or [])
    return {
        "tools": [
            {"name": tool.name, "description": tool.description, "read_only": tool.read_only, "allowed": tool.name in allowed}
            for tool in tools
        ]
    }
