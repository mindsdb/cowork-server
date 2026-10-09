"""Notion page search for the chat's "Add pages from Notion" picker."""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.common.settings.app_settings import OAuthSettings
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.services.connectors.connections import ConnectionsService
from cowork.services.connectors.notion_pages import (
    NotionReconnectRequired,
    NotionSearchUnavailable,
    search_pages,
)
from cowork.services.connectors.oauth import auth_proxy

_log = logging.getLogger("cowork.connectors.notion")

# AuthenticatedInOrgMode: in org mode the token comes from auth_proxy.proxy_token,
# which forwards the caller's own credential; auth checks it independently.
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])
ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


class NotionSearchRequest(BaseModel):
    name: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=200)


async def _access_token(name: str, scope: TenantScope, request: Request) -> str:
    if scope.org_mode:
        token = await auth_proxy.proxy_token("notion", request, OAuthSettings(), name=name)
        return token.get("access_token") or ""
    # Desktop: Electron's refresh loop keeps the vault's access token current.
    return ConnectionsService(scope).mcp_access_token("notion", name) or ""


@router.post("/search")
async def search_notion_pages(body: NotionSearchRequest, scope: ScopeDep, request: Request) -> dict:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Type something to search for.")
    access_token = await _access_token(body.name, scope, request)
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This Notion connection can't search pages. Connect Notion with In-Browser Connect.",
        )
    try:
        pages = await search_pages(access_token, query)
    except NotionSearchUnavailable as exc:
        _log.warning("Notion page search unavailable: %s", exc)
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Notion search isn't available in this build.") from exc
    except NotionReconnectRequired as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Notion needs to be reconnected.") from exc
    except Exception as exc:
        _log.warning("Notion page search failed: %r", exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Could not search Notion.") from exc
    return {"results": pages}
