"""Google Drive file search for the in-app file chooser."""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.common.settings.app_settings import OAuthSettings
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.services.connectors.connections import ConnectionsService
from cowork.services.connectors.google_drive_files import (
    DriveAccessDenied,
    DriveSearchError,
    search_files,
)
from cowork.services.connectors.oauth import auth_proxy

logger = logging.getLogger(__name__)

ENGINE = "google_drive"
ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]

# AuthenticatedInOrgMode: in org mode the token comes from auth_proxy.proxy_token,
# which forwards the caller's own Bearer; locally from the caller's vault.
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])

ACCESS_DENIED_DETAIL = (
    "Google Drive refused the search. Reconnect Google Drive and allow "
    "\"See and download all your Google Drive files\"."
)


class DriveSearchRequest(BaseModel):
    name: str
    query: str = ""
    page_token: str = ""


async def _access_token(name: str, request: Request, scope: TenantScope) -> str:
    if scope.org_mode:
        token = await auth_proxy.proxy_token(ENGINE, request, OAuthSettings(), name=name)
        return token.get("access_token") or ""
    fields = await run_in_threadpool(ConnectionsService(scope).runtime_fields, ENGINE, name)
    if fields is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Google Drive connection not found.")
    return fields.get("access_token") or ""


@router.post("/search")
async def search_drive_files(body: DriveSearchRequest, request: Request, scope: ScopeDep) -> dict:
    """The token stays on the server; the client only gets file metadata."""
    access_token = await _access_token(body.name, request, scope)
    if not access_token:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=ACCESS_DENIED_DETAIL)
    try:
        return await search_files(access_token, body.query, body.page_token)
    except DriveAccessDenied as exc:
        logger.info("drive search refused for %s: %s", body.name, exc)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=ACCESS_DENIED_DETAIL) from exc
    except DriveSearchError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
