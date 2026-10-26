"""SharePoint connections saved with an Entra app key.

A connection is created or edited here, never through the generic
submission: the key is checked against Microsoft (``SharePointClient.check``)
before anything is stored. Create and edit are separate routes so a create
can never replace an existing connection. Refused in org mode: the hosted
app keeps connections in auth, not in this vault.
"""
from __future__ import annotations

import re
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, status

from cowork.api.v1.permissions import DesktopOnly, require
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.services.connectors.connections import ConnectionsService
from cowork.services.connectors.identity import VAULT_KEEP_SENTINEL
from cowork.services.connectors.layered_vault import SHARED_CONNECTION_REFUSAL
from cowork.services.connectors.persist import persist_connection, vault_for_scope
from cowork.services.connectors.sharepoint import (
    AppKey,
    SharePointClient,
    SharePointError,
    parse_site_urls,
    validate_tenant_id,
)

router = APIRouter(dependencies=[Depends(require(DesktopOnly))])
ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]

ENGINE = "sharepoint"
_KEY_FIELDS = ("tenant_id", "client_id", "client_secret", "site_urls")
_CREATE_FIELDS = frozenset({"name", "user_label", *_KEY_FIELDS})
_EDIT_FIELDS = frozenset({"user_label", *_KEY_FIELDS})
# Alphanumeric at both ends, so anton's vault file-name sanitizing leaves it
# unchanged and two names can't land on one record.
_NAME = re.compile(r"[A-Za-z0-9]([A-Za-z0-9_-]{0,62}[A-Za-z0-9])?")


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


def _saved(vault, name: str) -> dict[str, Any]:
    """The response for a saved connection: its name and the label the UI shows."""
    record = vault.read_record(ENGINE, name) or {}
    user_label = str(record.get("fields", {}).get("_user_label", "")).strip() or None
    return {"ok": True, "name": name, "user_label": user_label}


@router.post("/connections", status_code=status.HTTP_201_CREATED)
def create_connection(body: Annotated[Any, Body()], scope: ScopeDep) -> dict[str, Any]:
    """Create a SharePoint connection once its key opens every site; 409 if the name is taken."""
    fields = _body_fields(body, _CREATE_FIELDS, ("name", *_KEY_FIELDS))
    name = fields["name"]
    if not _NAME.fullmatch(name):
        raise _unprocessable("Use letters, digits, - and _ for the name, starting and ending with a letter or digit.")
    if VAULT_KEEP_SENTINEL in fields.values():
        raise _unprocessable("A new connection has no stored secret to keep.")
    vault = vault_for_scope(scope)
    taken = {c["name"].casefold() for c in vault.list_connections() if c.get("engine") == ENGINE}
    if name.casefold() in taken:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="A SharePoint connection with this name already exists.")
    credentials = _credentials(fields)
    _check(credentials)
    slug = persist_connection(
        ENGINE, None, name, credentials, user_label=fields.get("user_label") or None, vault=vault, scope=scope,
    )
    if slug != name:
        # Another create took this name while the key was being checked.
        ConnectionsService(scope).delete(ENGINE, slug)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="A SharePoint connection with this name already exists.")
    return _saved(vault, slug)


@router.put("/connections/{name}")
def edit_connection(name: str, body: Annotated[Any, Body()], scope: ScopeDep) -> dict[str, Any]:
    """Replace a SharePoint connection's key and sites once the new values pass the check.

    Only ``client_secret`` may carry the keep sentinel, meaning the stored
    secret stays; the check runs with that stored secret.
    """
    fields = _body_fields(body, _EDIT_FIELDS, _KEY_FIELDS)
    vault = vault_for_scope(scope)
    record = vault.read_record(ENGINE, name) if _NAME.fullmatch(name) else None
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
