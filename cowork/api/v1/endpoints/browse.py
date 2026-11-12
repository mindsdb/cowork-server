"""/browse: the agent's browser, for Settings and the chat's browser pane.

GET    /browse/status     whether this deployment offers the caller a browser, and its state
POST   /browse/provision  web: start the caller's browser early, with their time zone and language
POST   /browse/embed      web: a viewer link for one of the caller's sessions
POST   /browse/approve    web: the person approves one step the browser held for them
POST   /browse/decline    web: the person skips one step the browser held for them
DELETE /browse/logins     web: sign the person out of every site in their browser

On the web the routes go to the browser router as the caller, with their own
MindsHub credential (cowork.services.browser). Status first reads the caller's
browser entitlement from MindsHub, and answers ``organization_off`` or
``not_entitled`` for an account that may not use the browser without asking
the router. A refusal from the router keeps its 401, 403 or 429 and its code in
``detail``; any other router failure is 424 with the router's code. The router
answers a turn key ``person_only`` on every route here.

Status also says what picks the browser's steps, its decisions model, and
whether the caller may set one up: without one the browser doesn't run, and
its switch reads as off.

Approve doesn't answer the chat card: the web app calls it first and answers
the card only once the browser recorded the approval. The card's Skip, and any
answer but Approve, sends decline, best effort: the step stays held either way.

On the desktop the browser is the app's own, registered through
``PUT /runtime-credential/browser``. Status says whether one is registered,
and a signed-in desktop whether its organization turned the browser off; the
other routes answer 409, as the web does without a router.
"""

from __future__ import annotations

from functools import partial
from typing import Annotated, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.common.settings.user_settings import get_user_settings
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.db.units import run_on_unit_threads
from cowork.handlers.jev_route import decisions_model
from cowork.principal import Principal, can_manage_org, get_principal, mindshub_credential
from cowork.services import browser as browser_service
from cowork.services.browser_contract import SESSION_ID

router = APIRouter()

ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]
PrincipalDep = Annotated[Principal | None, Depends(get_principal)]


class ProvisionRequest(BaseModel):
    # The app's own, so the browser presents the same time zone and language
    # as the person.
    timezone: str | None = None
    locale: str | None = None


class EmbedRequest(BaseModel):
    session_id: str = Field(pattern=SESSION_ID.pattern)


class OpenRequest(BaseModel):
    conversation_id: str = Field(pattern=SESSION_ID.pattern)
    # None opens the task's browser on a blank tab, where the pane shows its start page.
    url: str | None = Field(default=None, max_length=4096)


class TaskBrowsersRequest(BaseModel):
    # The project's tasks, from the app's own list; the server names their sessions.
    conversation_ids: list[Annotated[str, Field(pattern=SESSION_ID.pattern)]] = Field(
        max_length=browser_service.MAX_LISTED_TASKS,
    )


class TaskTabRequest(BaseModel):
    conversation_id: str = Field(pattern=SESSION_ID.pattern)
    tab_id: str = Field(pattern=r"^t[1-9]\d{0,8}$")
    # Closing: keep a note of it, to reopen when the task needs it.
    park: bool = True


class TaskBrowserRequest(BaseModel):
    conversation_id: str = Field(pattern=SESSION_ID.pattern)


def _raise(exc: browser_service.BrowserServiceError) -> NoReturn:
    raise HTTPException(status_code=exc.status, detail=exc.detail.model_dump()) from exc


# AuthenticatedInOrgMode, same as /hub/usage: on the web every route here acts
# as the caller with their MindsHub credential, and the router and MindsHub
# refuse a missing or invalid one. The desktop has no principal, and its
# routes call nothing.
@router.get(
    "/status", response_model=browser_service.BrowseStatus,
    dependencies=[Depends(require(AuthenticatedInOrgMode))],
)
async def browse_status(request: Request, scope: ScopeDep, principal: PrincipalDep) -> browser_service.BrowseStatus:
    try:
        return await browser_service.browse_status(
            await run_on_unit_threads(partial(get_user_settings, scope)), bearer=mindshub_credential(request),
            org_id=scope.org_id or "", user_id=scope.user_id or "",
            # Who may set the decision model, as the settings routes decide
            # who may write an organization setting.
            can_set_decisions_model=not scope.org_mode or can_manage_org(principal),
        )
    except browser_service.BrowserServiceError as exc:
        _raise(exc)


@router.post(
    "/provision", response_model=browser_service.BrowseStatus,
    dependencies=[Depends(require(AuthenticatedInOrgMode))],
)
async def browse_provision(
    request: Request, scope: ScopeDep, body: ProvisionRequest | None = None,
) -> browser_service.BrowseStatus:
    user = await run_on_unit_threads(partial(get_user_settings, scope))
    body = body or ProvisionRequest()
    prefs = browser_service.router_prefs(timezone=body.timezone, locale=body.locale)
    try:
        warmed = await browser_service.warm(mindshub_credential(request), prefs)
    except browser_service.BrowserServiceError as exc:
        _raise(exc)
    # The router started it for the caller, so it is theirs to use.
    return browser_service.BrowseStatus(
        enabled=browser_service.browser_runs(user), available=True, status=warmed.status,
        decisions_model=decisions_model(user),
    )


@router.post(
    "/embed", response_model=browser_service.BrowserEmbed,
    dependencies=[Depends(require(AuthenticatedInOrgMode))],
)
async def browse_embed(body: EmbedRequest, request: Request) -> browser_service.BrowserEmbed:
    try:
        return await browser_service.embed(mindshub_credential(request), body.session_id)
    except browser_service.BrowserServiceError as exc:
        _raise(exc)


@router.post(
    "/open", response_model=browser_service.OpenedSession,
    dependencies=[Depends(require(AuthenticatedInOrgMode))],
)
async def browse_open(body: OpenRequest, request: Request, scope: ScopeDep) -> browser_service.OpenedSession:
    """The person typed an address in a task's browser pane: open it there
    directly, without the agent (ENG-4138, prototype). No address opens the
    browser on a blank tab, or leaves an open one where it is."""
    url = browser_service.person_url(body.url) if body.url else None
    if body.url and url is None:
        raise HTTPException(status_code=400, detail={"code": "bad_url", "message": "That isn't a web address."})
    try:
        return await browser_service.person_open(
            mindshub_credential(request), user_id=scope.user_id, conversation_id=body.conversation_id, url=url,
        )
    except browser_service.BrowserServiceError as exc:
        _raise(exc)


@router.post(
    "/approve", response_model=browser_service.BrowserApproval,
    dependencies=[Depends(require(AuthenticatedInOrgMode))],
)
async def browse_approve(body: browser_service.ApproveRequest, request: Request) -> browser_service.BrowserApproval:
    try:
        return await browser_service.approve(mindshub_credential(request), body)
    except browser_service.BrowserServiceError as exc:
        _raise(exc)


@router.post(
    "/decline", response_model=browser_service.BrowserDecline,
    dependencies=[Depends(require(AuthenticatedInOrgMode))],
)
async def browse_decline(body: browser_service.DeclineRequest, request: Request) -> browser_service.BrowserDecline:
    try:
        return await browser_service.decline(mindshub_credential(request), body)
    except browser_service.BrowserServiceError as exc:
        _raise(exc)


@router.delete(
    "/logins", response_model=browser_service.LoginsCleared,
    dependencies=[Depends(require(AuthenticatedInOrgMode))],
)
async def browse_clear_logins(request: Request) -> browser_service.LoginsCleared:
    try:
        return await browser_service.clear_logins(mindshub_credential(request))
    except browser_service.BrowserServiceError as exc:
        _raise(exc)


# The project's browsers (ENG-4138, prototype): the open ones among a
# project's tasks, with their tabs, and the person's switch and close.
@router.post(
    "/browsers", response_model=browser_service.TaskBrowsers,
    dependencies=[Depends(require(AuthenticatedInOrgMode))],
)
async def browse_task_browsers(body: TaskBrowsersRequest, request: Request, scope: ScopeDep) -> browser_service.TaskBrowsers:
    try:
        return await browser_service.task_browsers(
            mindshub_credential(request), user_id=scope.user_id, conversation_ids=body.conversation_ids,
        )
    except browser_service.BrowserServiceError as exc:
        _raise(exc)


@router.post("/tabs/switch", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def browse_switch_tab(body: TaskTabRequest, request: Request, scope: ScopeDep) -> dict:
    try:
        await browser_service.switch_tab(
            mindshub_credential(request), user_id=scope.user_id, conversation_id=body.conversation_id, tab_id=body.tab_id,
        )
    except browser_service.BrowserServiceError as exc:
        _raise(exc)
    return {"ok": True}


@router.post("/tabs/close", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def browse_close_tab(body: TaskTabRequest, request: Request, scope: ScopeDep) -> dict:
    try:
        await browser_service.close_tab(
            mindshub_credential(request), user_id=scope.user_id, conversation_id=body.conversation_id,
            tab_id=body.tab_id, park=body.park,
        )
    except browser_service.BrowserServiceError as exc:
        _raise(exc)
    return {"ok": True}


@router.post("/close", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def browse_close(body: TaskBrowserRequest, request: Request, scope: ScopeDep) -> dict:
    try:
        await browser_service.close_task_browser(
            mindshub_credential(request), user_id=scope.user_id, conversation_id=body.conversation_id,
        )
    except browser_service.BrowserServiceError as exc:
        _raise(exc)
    return {"ok": True}
