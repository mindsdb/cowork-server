"""Live apps' own routes (cowork.services.live_apps).

The edge sends ``GET /api/v1/live-apps/<n>/access`` for every request to
``app<n>``, with the visitor's Cookie header and nothing else, and serves the
app only on a 200; on a 403, or a 502 or 503 from the slot, it shows
``/api/v1/live-apps/<n>/stopped-page`` instead. A slot program reports the end
of its app to ``/api/v1/live-apps/<n>/stopped``. Cowork reads the caller's own
apps from ``/api/v1/live-apps/mine``. Every route answers 404 with live apps off.
"""

from __future__ import annotations

import logging

from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute

from cowork.api.v1.permissions import AuthenticatedInOrgMode, OpenByDesign, require
from cowork.code_sandbox import live_apps_on
from cowork.principal import Principal, get_principal
from uuid import UUID

from cowork.schemas.live_apps import (
    LiveAppPreviewGrant,
    LiveAppPreviewRequest,
    LiveAppPreviewResponse,
    LiveAppPreviewToken,
    LiveAppStateResponse,
)
from cowork.services import live_app_preview as preview
from cowork.services.live_app_preview_pages import page as preview_page
from cowork.services.live_apps import (
    LiveAppAccessUnavailable,
    LiveAppPerson,
    SlotReportRefused,
    SlotStopReport,
    check_access,
    is_live_app_slot,
    live_app_person,
    person_apps,
    record_slot_stop,
    stopped_page,
)

logger = logging.getLogger(__name__)


class LiveAppRoute(APIRoute):
    """A live-app route: 404 with live apps off, before FastAPI reads the
    request, so a malformed one, a body that isn't JSON included, answers
    404 too, not 422."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def route_handler(request: Request) -> Response:
            if not live_apps_on():
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
            return await handler(request)

        return route_handler


router = APIRouter(route_class=LiveAppRoute)

_UNAVAILABLE = "The live app's access can't be checked right now."


def _require_slot(slot: int) -> None:
    if not is_live_app_slot(slot):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")


# AuthenticatedInOrgMode, declared explicitly. A caller with no verified email
# has launched nothing, so the list is empty.
@router.get("/mine", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def my_live_apps(principal: Principal | None = Depends(get_principal)) -> list[LiveAppStateResponse]:
    """The latest launch of each of the caller's apps, newest first, for the
    state line on its card."""
    person = live_app_person(principal)
    if person is None:
        return []
    return [
        LiveAppStateResponse(
            project_id=app.project_id,
            launch_id=app.record_id,
            slug=app.slug,
            slot=app.slot,
            state="running" if app.running else "stopped",
            address=app.address,
            started_at=app.started_at,
            stopped_at=app.stopped_at,
            # A relaunch keeps the app, so it is no stop to show.
            stop_reason=None if app.stop_reason == "relaunched" else app.stop_reason,
            stop_reason_text=None if app.stop_reason == "relaunched" else app.stop_text,
        )
        for app in await person_apps(person)
    ]


# OpenByDesign, standalone reason: the credential is the visitor's viewer
# sign-in cookie or preview lease, checked against the viewer sign-in and
# the slot's recorded launcher; it answers only whether that visitor launched
# the app.
@router.get("/{slot}/access", dependencies=[Depends(require(OpenByDesign))])
async def live_app_access(slot: int, request: Request) -> Response:
    _require_slot(slot)
    try:
        cookie = request.headers.get("cookie", "")
        if preview.preview_tokens(cookie):
            try:
                await preview.check(slot, cookie)
                outcome = "allowed"
            except preview.PreviewRefused as refused:
                outcome = "signed_out" if refused.status == 401 else "refused"
                # A tab holds the viewer cookie too: a stale lease cookie there,
                # or one an app's own script wrote, mustn't shut the launcher out.
                if viewer := preview.viewer_cookie_only(cookie):
                    outcome = await check_access(slot, cookie=viewer)
        else:
            outcome = await check_access(slot, cookie=cookie)
    except LiveAppAccessUnavailable as error:
        logger.warning(
            "Live app access unavailable",
            extra={"event": "live_app_access_unavailable", "slot": slot, "reason": error.reason},
        )
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_UNAVAILABLE) from None
    if outcome == "signed_out":
        return Response(status_code=status.HTTP_401_UNAUTHORIZED)
    if outcome == "refused":
        # The request boundary logs the refusal (cowork.common.request_logging).
        return Response(status_code=status.HTTP_403_FORBIDDEN)
    return Response(status_code=status.HTTP_200_OK)


# OpenByDesign, standalone reason: the credential is the visitor's viewer
# sign-in cookie, which the route checks itself; it explains a stop only to the
# person who launched the app, and shows anyone else a plain refusal.
@router.get("/{slot}/stopped-page", dependencies=[Depends(require(OpenByDesign))])
async def live_app_stopped_page(slot: int, request: Request) -> HTMLResponse:
    _require_slot(slot)
    try:
        cookie = request.headers.get("cookie", "")
        # A preview frame carries its lease cookie but not the viewer cookie.
        page = await stopped_page(slot, cookie=cookie, owner_id=await preview.lease_owner(slot, cookie))
    except LiveAppAccessUnavailable as error:
        logger.warning(
            "Live app stopped page unavailable",
            extra={"event": "live_app_stopped_page_unavailable", "slot": slot, "reason": error.reason},
        )
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_UNAVAILABLE) from None
    return HTMLResponse(page.html, status_code=page.status_code)


# OpenByDesign, standalone reason: a slot's report that its app ended, believed
# only from that slot's own address and only for the boot id of the slot's
# running record; at most it marks that one launch stopped.
@router.post(
    "/{slot}/stopped",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require(OpenByDesign))],
)
async def live_app_stopped(slot: int, report: SlotStopReport, request: Request) -> Response:
    _require_slot(slot)
    peer = request.client.host if request.client is not None else None
    try:
        await record_slot_stop(slot, report, peer=peer, headers=request.headers.keys())
    except SlotReportRefused as refusal:
        logger.warning(
            "Live app stop report refused",
            extra={"event": "live_app_stop_report_refused", "slot": slot, "reason": refusal.reason},
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden") from None
    except LiveAppAccessUnavailable as error:
        logger.warning(
            "Live app stop report unchecked",
            extra={"event": "live_app_stop_report_unchecked", "slot": slot, "reason": error.reason},
        )
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_UNAVAILABLE) from None
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# Every successful browser response here is private and must never be cached.
_PREVIEW_HEADERS = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"}


T = TypeVar("T")


def _preview_person(principal: Principal | None) -> LiveAppPerson:
    person = live_app_person(principal)
    if person is None:
        raise HTTPException(403, "Sign in with the account that launched this app.")
    return person


async def _preview_call(awaitable: Awaitable[T]) -> T:
    try:
        return await awaitable
    except preview.PreviewRefused as error:
        raise HTTPException(error.status, error.message, headers=_PREVIEW_HEADERS) from None
    except LiveAppAccessUnavailable:
        raise HTTPException(503, _UNAVAILABLE, headers=_PREVIEW_HEADERS) from None


@router.post("/previews", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def create_live_app_preview(body: LiveAppPreviewRequest, request: Request, response: Response,
                                  principal: Principal | None = Depends(get_principal)) -> LiveAppPreviewResponse:
    record_id, intent = await _preview_call(preview.create(_preview_person(principal), body.launch_id, request.headers.get("origin", "")))
    response.headers.update(_PREVIEW_HEADERS)
    return LiveAppPreviewResponse(id=record_id, intent=intent)


@router.post("/previews/{record_id}/keepalive", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def keep_live_app_preview(record_id: UUID, principal: Principal | None = Depends(get_principal)) -> Response:
    await _preview_call(preview.keepalive(_preview_person(principal), record_id))
    return Response(status_code=204, headers=_PREVIEW_HEADERS)


@router.delete("/previews", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def revoke_live_app_previews(principal: Principal | None = Depends(get_principal)) -> Response:
    await _preview_call(preview.revoke(_preview_person(principal)))
    return Response(status_code=204, headers=_PREVIEW_HEADERS)


@router.delete("/previews/{record_id}", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def revoke_live_app_preview(record_id: UUID, principal: Principal | None = Depends(get_principal)) -> Response:
    await _preview_call(preview.revoke(_preview_person(principal), record_id))
    return Response(status_code=204, headers=_PREVIEW_HEADERS)


# OpenByDesign: this page grants no access. Only its parent's secret intent
# and a viewer-confirmed one-use grant can establish a browser lease.
@router.get("/{slot}/preview/bootstrap", dependencies=[Depends(require(OpenByDesign))])
async def live_app_preview_bootstrap(slot: int) -> HTMLResponse:
    _require_slot(slot)
    return HTMLResponse(preview_page(slot, authorize=False), headers=_PREVIEW_HEADERS)


# OpenByDesign: the bridge confirms the viewer only in its POST route below.
@router.get("/{slot}/preview/authorize", dependencies=[Depends(require(OpenByDesign))])
async def live_app_preview_bridge(slot: int) -> HTMLResponse:
    _require_slot(slot)
    return HTMLResponse(preview_page(slot, authorize=True), headers=_PREVIEW_HEADERS)


def _preview_origin(slot: int, request: Request) -> None:
    _require_slot(slot)
    if request.headers.get("origin") != preview.app_origin(slot):
        raise HTTPException(403, "Forbidden", headers=_PREVIEW_HEADERS)


# OpenByDesign: an owner-authenticated secret intent and this browser's
# independently verified viewer cookie must name the same current launcher.
@router.post("/{slot}/preview/authorize", dependencies=[Depends(require(OpenByDesign))])
async def authorize_live_app_preview(slot: int, body: LiveAppPreviewToken, request: Request,
                                     response: Response) -> LiveAppPreviewGrant:
    _preview_origin(slot, request)
    grant = await _preview_call(preview.authorize(slot, body.token, request.headers.get("cookie", "")))
    response.headers.update(_PREVIEW_HEADERS)
    return LiveAppPreviewGrant(grant=grant)


# OpenByDesign: only a still-unused, unexpired viewer-confirmed grant can
# establish a lease. The ORM version check commits consumption before Set-Cookie.
@router.post("/{slot}/preview/exchange", dependencies=[Depends(require(OpenByDesign))])
async def exchange_live_app_preview(slot: int, body: LiveAppPreviewToken, request: Request) -> Response:
    _preview_origin(slot, request)
    name, token = await _preview_call(preview.exchange(slot, body.token))
    return Response(status_code=204, headers={**_PREVIEW_HEADERS,
        "Set-Cookie": f"{name}={token}; Path=/; HttpOnly; Secure; SameSite=None; Partitioned"})


# OpenByDesign: requires both the parent's secret intent and the matching
# HttpOnly cookie, then rechecks viewer identity, lease and current launch.
@router.post("/{slot}/preview/connect", dependencies=[Depends(require(OpenByDesign))])
async def connect_live_app_preview(slot: int, body: LiveAppPreviewToken, request: Request) -> Response:
    _preview_origin(slot, request)
    await _preview_call(preview.check(slot, request.headers.get("cookie", ""), intent=body.token))
    return Response(status_code=204, headers=_PREVIEW_HEADERS)


# OpenByDesign: only revokes leases for the viewer identity proved by the
# presented cookie. The edge calls this before viewer login/session changes.
@router.get("/{slot}/preview/revoke-viewer", dependencies=[Depends(require(OpenByDesign))])
async def revoke_viewer_live_app_previews(slot: int, request: Request) -> Response:
    _require_slot(slot)
    await _preview_call(preview.revoke_viewer(request.headers.get("cookie", "")))
    return Response(status_code=204, headers=_PREVIEW_HEADERS)
