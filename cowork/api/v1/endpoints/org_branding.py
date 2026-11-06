"""An organization's branding for the Cowork app (ENG-4020): a pass-through
to auth, which owns it.

``GET /org-branding``: auth's ``GET /v1/organizations/current/branding/``,
read with the caller's own credential and cached per caller for at most 60 s
(cowork.services.org_branding.fetch_auth_branding), answered in camelCase.
No credential, or auth answering that the caller has no current org, is
``enabled: false``. auth unreachable is a 200 with the stock look.

``PUT /org-branding/appearance``: the snake_case body (camelCase too)
passed to auth's ``PATCH /v1/organizations/current/branding/appearance/``,
which decides who may (``organization.branding.manage``) and what is valid.
auth's 403 and 422 keep their status and auth's message; no answer from auth
is a 502. It answers the GET body, and the caller's next GET sees the change.

Both work in hosted org mode and on a local server (the desktop). Which
credential is the caller's:

- Hosted org mode: the ``Authorization`` the gateway validated
  (cowork.principal.caller_bearer), never the client-set
  ``X-MindsHub-Authorization``. That header may name another org of the
  caller's, whose brand a GET would then write into this org's always-on
  skill.
- Local server: the ``X-MindsHub-Authorization`` the app sends with its hub
  calls (cowork.principal.mindshub_credential). Without it, no org.

Both keep the org's Brand guidelines skill in step
(cowork.services.org_branding.sync_brand_skill): always on for the org in
hosted org mode, an ordinary on-demand skill of the local store on a local
server.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import ScopedSession, ScopedSessionDep
from cowork.principal import caller_bearer, mindshub_credential
from cowork.schemas.org_branding import OrgAppearanceUpdate, OrgAppearanceView, OrgBrandingView
from cowork.services.org_branding import (
    AuthAppearanceError,
    AuthBranding,
    fetch_auth_branding,
    sync_brand_skill,
    update_auth_appearance,
)

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])

NO_ORG = "Organization branding needs a MindsHub organization. Sign in to MindsHub and try again."


def _credential(request: Request, scoped: ScopedSession) -> str:
    """The caller's MindsHub credential for this deployment (module docstring)."""
    if scoped.scope.org_mode:
        return caller_bearer(request) if scoped.scope.org_id else ""
    return mindshub_credential(request)


def _who(scoped: ScopedSession) -> dict[str, str]:
    return {"org_id": scoped.scope.org_id or "", "user_id": scoped.scope.user_id or ""}


def _sync_skill(scoped: ScopedSession, branding: AuthBranding) -> None:
    """Keep the Brand guidelines skill in step, never failing the request
    for it: the next read tries again."""
    try:
        sync_brand_skill(scoped.scope, branding, scoped)
    except Exception:
        logger.exception(
            "Could not update the organization's Brand guidelines skill",
            extra={"event": "org_branding_skill_sync_failed"},
        )


def _view(branding: AuthBranding) -> dict:
    if not branding.enabled:
        return OrgBrandingView().model_dump(by_alias=True)
    look = branding.look
    return OrgBrandingView(
        enabled=True,
        org_name=branding.org_name,
        sidebar_title=branding.sidebar_title,
        icon_url=branding.icon_url,
        logo_url=branding.logo_url,
        logo_dark_url=branding.logo_dark_url,
        appearance=OrgAppearanceView(
            accent=look.accent,
            background=look.background_light or look.background_dark,
            background_light=look.background_light,
            background_dark=look.background_dark,
            font=look.font,
            radius=look.radius,
            skin=look.skin,
            default_theme=look.default_theme,
            greeting=look.greeting,
            sidebar_title=look.sidebar_title,
            sidebar_title_mode=look.sidebar_title_mode,
            members_can_choose=look.members_can_choose,
            guidelines=look.guidelines,
        ),
        can_edit=branding.can_edit,
    ).model_dump(by_alias=True)


@router.get("", response_model=OrgBrandingView)
async def get_org_branding(request: Request, scoped: ScopedSessionDep):
    branding = await fetch_auth_branding(bearer_token=_credential(request, scoped), **_who(scoped))

    def settle() -> dict:
        _sync_skill(scoped, branding)
        return _view(branding)

    return await run_in_threadpool(settle)


@router.put("/appearance", response_model=OrgBrandingView)
async def put_org_appearance(body: OrgAppearanceUpdate, request: Request, scoped: ScopedSessionDep):
    credential = _credential(request, scoped)
    if not credential:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=NO_ORG)
    try:
        branding = await update_auth_appearance(
            bearer_token=credential, changes=body.auth_changes(), **_who(scoped)
        )
    except AuthAppearanceError as refused:
        if refused.status_code >= 500:
            logger.warning(
                "auth couldn't take an appearance change",
                extra={"event": "org_branding_appearance_save_failed", "status": refused.status_code},
            )
        raise HTTPException(status_code=refused.status_code, detail=refused.detail)

    def settle() -> dict:
        _sync_skill(scoped, branding)
        return _view(branding)

    return await run_in_threadpool(settle)
