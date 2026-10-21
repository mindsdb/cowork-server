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

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse

from cowork.api.v1.permissions import AuthenticatedInOrgMode, OpenByDesign, require
from cowork.code_sandbox import live_apps_on
from cowork.principal import Principal, get_principal
from cowork.schemas.live_apps import LiveAppStateResponse
from cowork.services.live_apps import (
    LiveAppAccessUnavailable,
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

router = APIRouter()

_UNAVAILABLE = "The live app's access can't be checked right now."


def _require_slot(slot: int) -> None:
    if not is_live_app_slot(slot):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")


# AuthenticatedInOrgMode, declared explicitly; the handler itself refuses a
# caller with no verified email, after the 404 a deployment without live apps
# answers.
@router.get("/mine", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def my_live_apps(principal: Principal | None = Depends(get_principal)) -> list[dict]:
    """The latest launch of each of the caller's apps, newest first, for the
    state line on its card."""
    if not live_apps_on():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
    person = live_app_person(principal)
    if person is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return [
        LiveAppStateResponse(
            project_id=app.project_id,
            slug=app.slug,
            slot=app.slot,
            state="running" if app.running else "stopped",
            address=app.address,
            started_at=app.started_at,
            stopped_at=app.stopped_at,
            stop_reason=app.stop_reason,
            stop_reason_text=app.stop_text,
        ).model_dump(by_alias=True, mode="json")
        for app in await person_apps(person)
    ]


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
        logger.warning("live app %d: access can't be told: %s", slot, error)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_UNAVAILABLE) from None
    if outcome == "signed_out":
        return Response(status_code=status.HTTP_401_UNAUTHORIZED)
    if outcome == "refused":
        logger.warning("live app %d: refused a signed-in visitor who didn't launch it", slot)
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
        logger.warning("live app %d: its stopped page can't be told: %s", slot, error)
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
        logger.warning("live app %d: %s", slot, refusal)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden") from None
    except LiveAppAccessUnavailable as error:
        logger.warning("live app %d: its stop report can't be checked: %s", slot, error)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_UNAVAILABLE) from None
    return Response(status_code=status.HTTP_204_NO_CONTENT)
