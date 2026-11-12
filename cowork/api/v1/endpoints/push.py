"""Register the mobile app's phones for push notifications (cowork.services.push).

Org mode only. Elsewhere both routes answer 404: there is no one to push to
and nothing to push with, and a 404 tells the app to stop trying, where a
5xx would read as a fault (and page) at the ingress.
"""
from __future__ import annotations

from functools import partial
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.exc import IntegrityError

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import SYSTEM_SCOPE
from cowork.db.units import run_db
from cowork.principal import Principal, get_principal
from cowork.schemas.push import PushDeviceRegistration, PushDeviceRemoval
from cowork.services.push import devices


def _require_org_mode() -> None:
    if get_app_settings().tenancy_mode != "org":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Push notifications are available only in MindsHub deployments.",
        )


router = APIRouter(
    dependencies=[Depends(_require_org_mode), Depends(require(AuthenticatedInOrgMode))],
)


PrincipalDep = Annotated[Principal | None, Depends(get_principal)]


def _caller(principal: PrincipalDep) -> Principal:
    if principal is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return principal


@router.post("/devices", status_code=status.HTTP_204_NO_CONTENT)
async def register_device(body: PushDeviceRegistration, principal: Principal = Depends(_caller)) -> Response:
    """Record the phone's token as the caller's. A token already on file,
    under anyone, moves to the caller: the token names the phone, and the
    phone is theirs now."""
    register = partial(
        devices.register_device,
        org_id=principal.org_id, user_id=principal.user_id, token=body.token.strip(),
        platform=body.platform, app_version=body.app_version, environment=body.environment,
    )
    # SYSTEM_SCOPE: the token's row may sit in another org (devices module doc);
    # every write is keyed by the caller's own ids.
    try:
        await run_db(register, scope=SYSTEM_SCOPE)
    except IntegrityError:
        # Two registrations of one new token raced; the second now finds the row.
        await run_db(register, scope=SYSTEM_SCOPE)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/devices", status_code=status.HTTP_204_NO_CONTENT)
async def unregister_device(body: PushDeviceRemoval, principal: Principal = Depends(_caller)) -> Response:
    """Stop pushing to the phone. Only the caller's own token goes; answers
    204 either way."""
    await run_db(
        partial(
            devices.unregister_device,
            org_id=principal.org_id, user_id=principal.user_id, token=body.token.strip(),
        ),
        scope=SYSTEM_SCOPE,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
