"""/browse — the user's MindsHub browser (ENG-3298).

GET  /browse/status     is it provisioned, running, enabled
POST /browse/provision  create or wake it, and remember where it is
POST /browse/embed      a fresh viewer URL for the side pane

Replaces the compat stub that always answered ``{"available": false}``;
``available`` stays in the answer for the renderer that already reads it.

In ``standalone`` browser mode the same routes talk to the org's own browser
service with its API key instead: no MindsHub credential needed, nothing to
provision, and ``mode`` in the answer says which one the renderer is showing.
"""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlmodel import Session

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.common.settings.user_settings import get_user_settings
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.db.session import get_session
from cowork.principal import hub_credential
from cowork.services import browser as browser_service
from cowork.services.settings import SettingService

router = APIRouter()

ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]
SessionDep = Annotated[Session, Depends(get_session)]


class EmbedRequest(BaseModel):
    # None: the user's own profile for the mode, resolved server-side since on
    # a standalone service it depends on who is asking.
    session_id: Optional[str] = None


def _raise(exc: browser_service.BrowserServiceError) -> None:
    raise HTTPException(status_code=exc.status, detail=exc.detail) from exc


def _answer(view: dict, user) -> dict:
    enabled = bool(user.browser_enabled)
    return {
        **view,
        "mode": browser_service.browser_mode(user),
        "enabled": enabled,
        "available": enabled and view.get("status") == "running" and bool(view.get("endpoint")),
    }


def _remember_endpoint(session: Session, scope: TenantScope, endpoint: str) -> None:
    if endpoint and get_user_settings(scope).browser_url != endpoint:
        SettingService(session, scope).upsert_setting("browser_url", endpoint)


# AuthenticatedInOrgMode, same as /hub/usage: in mindshub mode every route
# here acts as the caller with their MindsHub credential, and MindsHub rejects
# a missing or invalid one. Local mode has no principal, and the desktop's
# credential is what the call carries. Standalone mode needs no MindsHub
# credential, but still only answers an authenticated caller.
@router.get("/status", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def browse_status(request: Request, scope: ScopeDep, session: SessionDep) -> dict:
    user = get_user_settings(scope)
    try:
        if browser_service.browser_mode(user) == browser_service.STANDALONE:
            return _answer(await browser_service.service_status(user), user)
        view = await browser_service.fetch_status(hub_credential(request))
    except browser_service.BrowserServiceError as exc:
        _raise(exc)
    _remember_endpoint(session, scope, view.get("endpoint", ""))
    return _answer(view, user)


@router.post("/provision", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def browse_provision(request: Request, scope: ScopeDep, session: SessionDep) -> dict:
    user = get_user_settings(scope)
    try:
        if browser_service.browser_mode(user) == browser_service.STANDALONE:
            # Nothing to create on someone's own service, and `browser_url`
            # stays MindsHub's: answer as status does.
            return _answer(await browser_service.service_status(user), user)
        view = await browser_service.provision(hub_credential(request))
    except browser_service.BrowserServiceError as exc:
        _raise(exc)
    _remember_endpoint(session, scope, view.get("endpoint", ""))
    return _answer(view, user)


@router.post("/embed", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def browse_embed(body: EmbedRequest, request: Request, scope: ScopeDep) -> dict:
    try:
        return await browser_service.embed_for(
            get_user_settings(scope), hub_credential(request), scope.user_id, body.session_id,
        )
    except browser_service.BrowserServiceError as exc:
        _raise(exc)
