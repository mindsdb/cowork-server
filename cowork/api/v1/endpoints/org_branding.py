"""An organization's branding for the Cowork app (ENG-4020).

``GET /org-branding``: what every member's app applies. The org's look
(cowork.models.org_appearance) plus its images, whose URLs come from auth's
``GET /v1/organizations/current/branding/`` read with the caller's own bearer
(cowork.principal.caller_bearer), cached per caller. Auth unreachable means
null image URLs, still a 200. ``canEdit`` is org-admin standing
(cowork.principal.can_manage_org). Outside hosted org mode branding is off:
``enabled`` false and everything else null.

``PUT /org-branding/appearance``: an org admin changes the look; everyone
else gets a 403. It answers the GET body.

Both keep the org's Brand guidelines skill in step with its branding
(cowork.services.org_branding.sync_brand_skill).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from cowork.api.v1.permissions import AuthenticatedInOrgMode, AuthenticatedOrgAdmin, require
from cowork.db.scoped import ScopedSession, ScopedSessionDep
from cowork.models.org_appearance import OrgAppearance
from cowork.principal import Principal, caller_bearer, can_manage_org, get_principal
from cowork.schemas.org_branding import OrgAppearanceUpdate, OrgBrandingView
from cowork.services.org_branding import (
    AuthBranding,
    appearance_row,
    appearance_view,
    fetch_auth_branding,
    resolve_sidebar_title,
    save_appearance,
    sync_brand_skill,
)

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])

NOT_IN_ORG_MODE = "Organization branding is only available in a hosted organization."


def _in_org(scoped: ScopedSession) -> bool:
    return bool(scoped.scope.org_mode and scoped.scope.org_id)


async def _images(request: Request, scoped: ScopedSession) -> AuthBranding:
    return await fetch_auth_branding(
        bearer_token=caller_bearer(request),
        org_id=scoped.scope.org_id or "",
        user_id=scoped.scope.user_id or "",
    )


def _sync_skill(scoped: ScopedSession, images: AuthBranding, row: OrgAppearance | None) -> None:
    """Keep the Brand guidelines skill in step, never failing the request
    for it: the next read tries again."""
    if not images.complete:
        return
    try:
        sync_brand_skill(scoped, images, row)
    except Exception:
        logger.exception("Could not update the organization's Brand guidelines skill")


def _view(images: AuthBranding, row: OrgAppearance | None, principal: Principal | None) -> dict:
    return OrgBrandingView(
        enabled=True,
        org_name=images.org_name if images.org_known else None,
        sidebar_title=resolve_sidebar_title(row, images.org_name if images.org_known else None),
        icon_url=images.icon_url,
        logo_url=images.logo_url,
        logo_dark_url=images.logo_dark_url,
        appearance=appearance_view(row),
        can_edit=can_manage_org(principal),
    ).model_dump(by_alias=True)


@router.get("", response_model=OrgBrandingView)
async def get_org_branding(
    request: Request,
    scoped: ScopedSessionDep,
    principal: Principal | None = Depends(get_principal),
):
    if not _in_org(scoped):
        return OrgBrandingView().model_dump(by_alias=True)
    images = await _images(request, scoped)

    def read() -> dict:
        row = appearance_row(scoped)
        _sync_skill(scoped, images, row)
        return _view(images, row, principal)

    return await run_in_threadpool(read)


@router.put("/appearance", response_model=OrgBrandingView)
async def put_org_appearance(
    body: OrgAppearanceUpdate,
    request: Request,
    scoped: ScopedSessionDep,
    principal: Principal | None = Depends(require(AuthenticatedOrgAdmin)),
):
    if not _in_org(scoped):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=NOT_IN_ORG_MODE)
    if not can_manage_org(principal):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only an organization admin can change the organization's branding.",
        )
    images = await _images(request, scoped)

    def write() -> dict:
        row = save_appearance(scoped, body)
        _sync_skill(scoped, images, row)
        return _view(images, row, principal)

    return await run_in_threadpool(write)
