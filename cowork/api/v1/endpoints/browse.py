"""/browse — the user's MindsHub browser (ENG-3298).

GET  /browse/status     is it provisioned, running, enabled
POST /browse/provision  create or wake it, and remember where it is
POST /browse/embed      a fresh viewer URL for the side pane
GET  /browse/locations  the router's cities and the caller's proxy preferences

Replaces the compat stub that always answered ``{"available": false}``;
``available`` stays in the answer for the renderer that already reads it.

In ``standalone`` browser mode the same routes talk to the org's own browser
service with its API key instead: no MindsHub credential needed, nothing to
provision, and ``mode`` in the answer says which one the renderer is showing.

With the shared browser router (``COWORK_BROWSER_ROUTER_URL``, ENG-3188) the
MindsHub mode's routes go to it instead of the per-user instance: status reads
the caller's pod (``starting`` while it comes up), provision warms it, and the
answer's ``endpoint`` is the router. Nothing is remembered in ``browser_url``
then: the router is the same for everyone. Provision also hands the router the
caller's proxy preferences (``browser_proxy``, ``browser_city``) and, when the
renderer sends them, its time zone and language; ``/browse/locations`` lists
the cities Settings can offer.
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
from cowork.principal import mindshub_credential
from cowork.services import browser as browser_service
from cowork.services.settings import SettingService

router = APIRouter()

ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]
SessionDep = Annotated[Session, Depends(get_session)]


class ProvisionRequest(BaseModel):
    # The renderer's own, so the browser presents the same time zone and
    # language as the person (and picks their city when none is chosen).
    timezone: Optional[str] = None
    locale: Optional[str] = None


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
    # The router is deployment config, not where this user's instance is.
    if browser_service.router_url():
        return
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
        view = await browser_service.status_for(mindshub_credential(request))
    except browser_service.BrowserServiceError as exc:
        _raise(exc)
    _remember_endpoint(session, scope, view.get("endpoint", ""))
    return _answer(view, user)


@router.post("/provision", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def browse_provision(
    request: Request, scope: ScopeDep, session: SessionDep, body: Optional[ProvisionRequest] = None,
) -> dict:
    user = get_user_settings(scope)
    body = body or ProvisionRequest()
    try:
        if browser_service.browser_mode(user) == browser_service.STANDALONE:
            # Nothing to create on someone's own service, and `browser_url`
            # stays MindsHub's: answer as status does.
            return _answer(await browser_service.service_status(user), user)
        prefs = browser_service.router_prefs(user, body.timezone, body.locale)
        view = await browser_service.provision_for(mindshub_credential(request), prefs)
    except browser_service.BrowserServiceError as exc:
        _raise(exc)
    _remember_endpoint(session, scope, view.get("endpoint", ""))
    return _answer(view, user)


@router.post("/embed", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def browse_embed(body: EmbedRequest, request: Request, scope: ScopeDep) -> dict:
    try:
        return await browser_service.embed_for(
            get_user_settings(scope), mindshub_credential(request), scope.user_id, body.session_id,
        )
    except browser_service.BrowserServiceError as exc:
        _raise(exc)


@router.get("/locations", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def browse_locations(request: Request, scope: ScopeDep) -> dict:
    # Only the router has cities; a standalone service or a per-user instance
    # browses from wherever it runs.
    if browser_service.browser_mode(get_user_settings(scope)) == browser_service.STANDALONE:
        return dict(browser_service.NO_LOCATIONS)
    try:
        return await browser_service.locations(mindshub_credential(request))
    except browser_service.BrowserServiceError as exc:
        _raise(exc)
