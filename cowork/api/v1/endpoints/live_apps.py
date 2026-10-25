"""Live apps' own routes (cowork.services.live_apps).

The edge sends ``GET /api/v1/live-apps/<n>/access`` for every request to
``app<n>``, with the visitor's Cookie header and nothing else, and serves the
app only on a 200; on a 403, or a 502 or 503 from the slot, it shows
``/api/v1/live-apps/<n>/stopped-page`` instead. A slot program reports the end
of its app to ``/api/v1/live-apps/<n>/stopped``. Cowork reads the caller's own
apps from ``/api/v1/live-apps/mine``. Every route answers 404 with live apps off.
"""

from __future__ import annotations

import hashlib
import logging

from collections.abc import Callable, Coroutine
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute

from cowork.api.v1.permissions import AuthenticatedInOrgMode, OpenByDesign, require
from cowork.code_sandbox import live_apps_on
from cowork.principal import Principal, get_principal
from cowork.schemas.live_apps import (
    LiveAppAvailabilityNotice,
    LiveAppAvailabilityResponse,
    LiveAppStateResponse,
)
from cowork.services.live_apps import (
    LiveAppAccessUnavailable,
    acknowledge_availability,
    SlotReportRefused,
    SlotStopReport,
    check_access,
    is_live_app_slot,
    live_app_person,
    names_a_person,
    person_apps,
    person_availability,
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


def _availability_principal(principal: Principal | None) -> Principal:
    if not names_a_person(principal):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sign in to check live app availability.")
    return principal


@router.get("/availability", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def live_app_availability(
    response: Response, principal: Principal | None = Depends(get_principal)
) -> LiveAppAvailabilityResponse:
    principal = _availability_principal(principal)
    response.headers["Cache-Control"] = "no-store"
    # A stable opaque cache namespace, never another person's id or email.
    identity = f"{len(principal.org_id)}:{principal.org_id}{principal.user_id}"
    scope = hashlib.sha256(identity.encode()).hexdigest()
    person = live_app_person(principal)
    if person is None:
        return LiveAppAvailabilityResponse(waiting=False, scope=scope)
    availability = await person_availability(person)
    notice = None if availability.notice is None else LiveAppAvailabilityNotice(
        id=availability.notice.id, message=availability.notice.message
    )
    return LiveAppAvailabilityResponse(waiting=availability.waiting, notice=notice, scope=scope)


@router.post(
    "/availability/{notice_id}/ack",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require(AuthenticatedInOrgMode))],
)
async def acknowledge_live_app_availability(
    notice_id: UUID, principal: Principal | None = Depends(get_principal)
) -> Response:
    from anton.core.artifacts.live_app_slots import LiveAppLaunchError

    person = live_app_person(_availability_principal(principal))
    if person is not None:
        try:
            await acknowledge_availability(person, notice_id)
        except LiveAppLaunchError as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from None
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# OpenByDesign, standalone reason: the credential is the visitor's viewer
# sign-in cookie, which the route checks itself against the viewer sign-in and
# the slot's recorded launcher; it answers only whether that visitor launched
# the app.
@router.get("/{slot}/access", dependencies=[Depends(require(OpenByDesign))])
async def live_app_access(slot: int, request: Request) -> Response:
    _require_slot(slot)
    try:
        outcome = await check_access(slot, cookie=request.headers.get("cookie", ""))
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
        page = await stopped_page(slot, cookie=request.headers.get("cookie", ""))
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
