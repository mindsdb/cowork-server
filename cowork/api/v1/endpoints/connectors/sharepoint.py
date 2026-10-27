"""SharePoint connections saved with an Entra app key, and browsing them.

A connection is created or edited here, never through the generic
submission: the key is checked against Microsoft (``SharePointClient.check``)
before anything is stored. Create and edit are separate routes: a create
refuses a name already in use, so it never replaces another key's
connection. Creates take one lock while they check the name again and save,
so a name or env prefix taken during the key check is refused, not
overwritten. Browse reads only the sites the
connection names. Refused in org mode: the hosted app keeps connections in
auth, not in this vault.
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, status

from anton.core.datasources.data_vault import LocalDataVault

from cowork.api.v1.permissions import DesktopOnly, require
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.schemas.sharepoint import (
    SharePointChildrenResponse,
    SharePointItem,
    SharePointLibrariesResponse,
    SharePointLibrary,
)
from cowork.services.connectors.connections import ConnectionsService
from cowork.services.connectors.identity import VAULT_KEEP_SENTINEL
from cowork.services.connectors.layered_vault import SHARED_CONNECTION_REFUSAL, LayeredDataVault
from cowork.services.connectors.persist import persist_connection, vault_for_scope
from cowork.services.connectors.sharepoint import (
    ENGINE,
    AppKey,
    SharePointClient,
    SharePointError,
    SharePointErrorKind,
    normalise_site_url,
    parse_site_urls,
    validate_tenant_id,
)

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require(DesktopOnly))])
ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]

_KEY_FIELDS = ("tenant_id", "client_id", "client_secret", "site_urls")
_CREATE_FIELDS = frozenset({"name", "user_label", *_KEY_FIELDS})
_EDIT_FIELDS = frozenset({"user_label", *_KEY_FIELDS})
# Letters and digits joined by single - or _, so anton's vault file name and
# DS_ env prefix keep the name intact and `__` (its field separator) never appears.
_NAME = re.compile(r"[A-Za-z0-9]+([-_][A-Za-z0-9]+)*")
_MAX_NAME = 64
_MAX_CURSOR = 4096
_KEY_NO_LONGER_WORKS = "This connection's key no longer works. Ask whoever set it up to update it."
_NAME_TAKEN = "A SharePoint connection with this name already exists."
# One server process serves the desktop and self-hosted; org mode is refused.
_CREATE_LOCK = threading.Lock()
_BROWSE_STATUS = {
    SharePointErrorKind.INVALID: status.HTTP_400_BAD_REQUEST,
    SharePointErrorKind.TENANT_NOT_FOUND: status.HTTP_409_CONFLICT,
    SharePointErrorKind.APP_NOT_FOUND: status.HTTP_409_CONFLICT,
    SharePointErrorKind.KEY_REJECTED: status.HTTP_409_CONFLICT,
    SharePointErrorKind.SITE_NOT_GRANTED: status.HTTP_404_NOT_FOUND,
    SharePointErrorKind.ITEM_NOT_FOUND: status.HTTP_404_NOT_FOUND,
    SharePointErrorKind.THROTTLED: status.HTTP_429_TOO_MANY_REQUESTS,
    SharePointErrorKind.UNREACHABLE: status.HTTP_502_BAD_GATEWAY,
    # Raised only by imports; listed so every kind maps.
    SharePointErrorKind.TOO_MANY_FILES: status.HTTP_400_BAD_REQUEST,
    SharePointErrorKind.TOO_LARGE: status.HTTP_400_BAD_REQUEST,
    SharePointErrorKind.BAD_NAME: status.HTTP_400_BAD_REQUEST,
    SharePointErrorKind.PLAN_TOO_SLOW: status.HTTP_400_BAD_REQUEST,
    SharePointErrorKind.IMPORT_TOO_SLOW: status.HTTP_400_BAD_REQUEST,
}


def _unprocessable(detail: str) -> HTTPException:
    """A 422 whose detail is written here, never an echo of the request body."""
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=detail)


def _body_fields(body: Any, allowed: frozenset[str], required: tuple[str, ...]) -> dict[str, str]:
    """The body's fields, all text, none outside ``allowed``, every ``required`` one present."""
    if not isinstance(body, dict):
        raise _unprocessable("Send the connection as a JSON object.")
    unknown = sorted(str(key) for key in body if key not in allowed)
    if unknown:
        raise _unprocessable(f"Unknown fields: {', '.join(unknown)}.")
    fields: dict[str, str] = {}
    for key, value in body.items():
        if not isinstance(value, str):
            raise _unprocessable(f"{key} must be text.")
        fields[key] = value.strip()
    missing = [key for key in required if not fields.get(key)]
    if missing:
        raise _unprocessable(f"Missing {', '.join(missing)}.")
    return fields


def _credentials(fields: dict[str, str]) -> dict[str, str]:
    """The stored fields, validated: site URLs normalised and kept one per line."""
    try:
        tenant_id = validate_tenant_id(fields["tenant_id"])
        site_urls = "\n".join(parse_site_urls(fields["site_urls"]))
    except SharePointError as exc:
        raise _unprocessable(str(exc)) from exc
    return {
        "tenant_id": tenant_id,
        "client_id": fields["client_id"],
        "client_secret": fields["client_secret"],
        "site_urls": site_urls,
    }


def _check(credentials: dict[str, str]) -> None:
    """Refuse with the check's own message unless the key opens every site."""
    key = AppKey(credentials["tenant_id"], credentials["client_id"], credentials["client_secret"])
    try:
        with SharePointClient(key) as client:
            client.check(credentials["site_urls"].split("\n"))
    except SharePointError as exc:
        raise _unprocessable(str(exc)) from exc


def _env_key(name: str) -> str:
    """The form anton's DS_ env prefix gives a name: two names with one key would overwrite each other's env."""
    return re.sub(r"[^\w]", "_", name).casefold()


def _name_taken(vault: LocalDataVault | LayeredDataVault, name: str) -> bool:
    """Whether a SharePoint connection already holds ``name`` or its env prefix."""
    taken = {_env_key(c["name"]) for c in vault.list_connections() if c.get("engine") == ENGINE}
    return _env_key(name) in taken


def _valid_name(name: str) -> bool:
    """Whether ``name`` fits the connection name rule."""
    return len(name) <= _MAX_NAME and bool(_NAME.fullmatch(name))


def _saved(vault: LocalDataVault | LayeredDataVault, name: str) -> dict[str, Any]:
    """The response for a saved connection: its name and the label the UI shows."""
    record = vault.read_record(ENGINE, name) or {}
    user_label = str(record.get("fields", {}).get("_user_label", "")).strip() or None
    return {"ok": True, "name": name, "user_label": user_label}


@router.post("/connections", status_code=status.HTTP_201_CREATED)
def create_connection(body: Annotated[Any, Body()], scope: ScopeDep) -> dict[str, Any]:
    """Create a SharePoint connection once its key opens every site; 409 if the name is taken."""
    fields = _body_fields(body, _CREATE_FIELDS, ("name", *_KEY_FIELDS))
    name = fields["name"]
    if not _valid_name(name):
        raise _unprocessable(
            "Use letters and digits for the name, joined by single - or _, at most 64 characters."
        )
    if VAULT_KEEP_SENTINEL in fields.values():
        raise _unprocessable("A new connection has no stored secret to keep.")
    vault = vault_for_scope(scope)
    if _name_taken(vault, name):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=_NAME_TAKEN)
    credentials = _credentials(fields)
    _check(credentials)
    # The key check can take a minute, so the name is checked again under the
    # lock every create holds while it saves.
    with _CREATE_LOCK:
        if _name_taken(vault, name):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=_NAME_TAKEN)
        slug = persist_connection(
            ENGINE, None, name, credentials, user_label=fields.get("user_label") or None, vault=vault, scope=scope,
        )
    if slug != name:
        # Only a writer outside these routes gets here. The record it saved
        # to may have existed before this request, so nothing is deleted.
        logger.warning("A SharePoint create for %r was saved as %r by another writer's race", name, slug)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=_NAME_TAKEN)
    return _saved(vault, slug)


@router.put("/connections/{name}")
def edit_connection(name: str, body: Annotated[Any, Body()], scope: ScopeDep) -> dict[str, Any]:
    """Replace a SharePoint connection's key and sites once the new values pass the check.

    Only ``client_secret`` may carry the keep sentinel, meaning the stored
    secret stays; the check runs with that stored secret.
    """
    fields = _body_fields(body, _EDIT_FIELDS, _KEY_FIELDS)
    vault = vault_for_scope(scope)
    record = vault.read_record(ENGINE, name) if _valid_name(name) else None
    if record is None or record.get("name") != name:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found.")
    if not ConnectionsService(scope).can_change(ENGINE, name):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=SHARED_CONNECTION_REFUSAL)
    if any(fields[key] == VAULT_KEEP_SENTINEL for key in ("tenant_id", "client_id", "site_urls")):
        raise _unprocessable("Only the client secret can be kept as stored.")
    credentials = _credentials(fields)
    checked = dict(credentials)
    if credentials["client_secret"] == VAULT_KEEP_SENTINEL:
        stored_secret = str(record.get("fields", {}).get("client_secret") or "")
        if not stored_secret:
            raise _unprocessable("This connection has no stored secret to keep. Enter the client secret.")
        checked["client_secret"] = stored_secret
    _check(checked)
    persist_connection(
        ENGINE,
        None,
        name,
        credentials,
        user_label=fields.get("user_label") or None,
        replace_existing=True,
        vault=vault,
        scope=scope,
    )
    return _saved(vault, name)


def _browse_error(exc: SharePointError) -> HTTPException:
    """The browse response for a SharePoint failure; a stored key that stopped working is one message."""
    detail = str(exc)
    if exc.kind in {
        SharePointErrorKind.TENANT_NOT_FOUND,
        SharePointErrorKind.APP_NOT_FOUND,
        SharePointErrorKind.KEY_REJECTED,
    }:
        detail = _KEY_NO_LONGER_WORKS
    return HTTPException(status_code=_BROWSE_STATUS[exc.kind], detail=detail)


def _stored_connection(scope: TenantScope, name: str) -> tuple[AppKey, list[str]]:
    """The saved connection's key and sites, validated again on every load.

    A record can be written without the check (an agent turn, an agent-built
    form), so nothing stored reaches a URL until it passes the same rules a
    create applies.
    """
    record = vault_for_scope(scope).read_record(ENGINE, name) if _valid_name(name) else None
    if record is None or record.get("name") != name:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found.")
    fields = record.get("fields") or {}
    client_id = str(fields.get("client_id") or "").strip()
    client_secret = str(fields.get("client_secret") or "")
    try:
        tenant_id = validate_tenant_id(str(fields.get("tenant_id") or ""))
        sites = parse_site_urls(str(fields.get("site_urls") or ""))
        if not client_id or not client_secret:
            raise SharePointError(SharePointErrorKind.INVALID, "The stored key is incomplete.")
    except SharePointError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="This connection needs to be saved again."
        ) from exc
    return AppKey(tenant_id, client_id, client_secret), sites


def _granted_site(site: str, sites: list[str]) -> str:
    """``site`` normalised, if the connection names it; 403 otherwise."""
    try:
        canonical = normalise_site_url(site)
    except SharePointError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if canonical not in sites:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This connection doesn't include that site.")
    return canonical


@router.get("/connections/{name}/drives", response_model=SharePointLibrariesResponse)
def list_libraries(name: str, site: str, scope: ScopeDep) -> SharePointLibrariesResponse:
    """List the document libraries of one site the connection names."""
    key, sites = _stored_connection(scope, name)
    url = _granted_site(site, sites)
    try:
        with SharePointClient(key) as client:
            libraries = client.list_drives(url)
    except SharePointError as exc:
        raise _browse_error(exc) from exc
    return SharePointLibrariesResponse(
        libraries=[SharePointLibrary(id=library.id, name=library.name) for library in libraries]
    )


@router.get("/connections/{name}/children", response_model=SharePointChildrenResponse)
def list_children(
    name: str,
    site: str,
    drive_id: str,
    scope: ScopeDep,
    item_id: str = "root",
    cursor: str | None = None,
) -> SharePointChildrenResponse:
    """List one page of a folder in one of the connection's sites; ``cursor`` pages on."""
    if cursor is not None and len(cursor) > _MAX_CURSOR:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="That page cursor isn't valid.")
    key, sites = _stored_connection(scope, name)
    url = _granted_site(site, sites)
    try:
        with SharePointClient(key) as client:
            page = client.list_children(url, drive_id, item_id, cursor)
    except SharePointError as exc:
        raise _browse_error(exc) from exc
    return SharePointChildrenResponse(
        items=[
            SharePointItem(id=item.id, name=item.name, size=item.size, kind=item.kind, modified=item.modified)
            for item in page.items
        ],
        cursor=page.cursor,
    )
