"""An organization's branding for the Cowork app: a pass-through
to auth, which owns it.

``GET /org-branding``: auth's ``GET /v1/organizations/current/branding/``,
read with the caller's own credential and cached per caller for at most 60 s
(cowork.services.org_branding.fetch_auth_branding), answered in camelCase.
No credential, or auth answering that the caller has no current org, is
``enabled: false``; only auth's 403 says the caller has no current org.
When auth answers nothing usable (no answer, a 5xx, a 429, a 401 or a 404)
the caller's last answer is served for up to 10 minutes. Without one, any
4xx from auth but its 403 answers 200 with ``reachable: false``, and no
answer, a 5xx or a body that isn't the branding answers 503. The app takes
both as "keep the look you have" (mindsdb/cowork's useOrgBranding hook), so
an outage or an expired session never flips the look to stock. Only the
outage is a 5xx: auth answering 404 while its release trails this
server's, or a 401 for an expired desktop session, isn't one.

``PUT /org-branding/appearance``: the snake_case body (camelCase too)
passed to auth's ``PATCH /v1/organizations/current/branding/appearance/``,
which decides who may (``organization.branding.manage``) and what is valid.
auth's 403 and 422 keep their status and auth's message; no answer from auth
is a 502. auth takes a write only from a console session, so the PUT
forwards only a credential shaped like the user's session JWT
(cowork.principal.is_session_jwt): an API key, a turn key or no credential
at all is a 403 asking the caller to sign in, and so is auth's 401. It
answers the GET body, and the caller's next GET sees the change.

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
from functools import partial
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import ScopedSession, TenantScope, get_tenant_scope
from cowork.db.units import run_efs_unit
from cowork.principal import hub_credential_for, is_session_jwt
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

ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]

SIGN_IN_TO_EDIT = "Sign in to MindsHub to change branding."
READ_UNREACHABLE = "Couldn't reach MindsHub for the organization's branding. Try again."


def _credential(request: Request, scope: TenantScope) -> str:
    """The caller's MindsHub credential for this deployment (module docstring)."""
    return hub_credential_for(request, org_mode=scope.org_mode, org_id=scope.org_id)


def _who(scope: TenantScope) -> dict[str, str]:
    return {"org_id": scope.org_id or "", "user_id": scope.user_id or ""}


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
        reachable=branding.reachable,
    ).model_dump(by_alias=True)


def _settle(scoped: ScopedSession, *, branding: AuthBranding) -> dict:
    """Sync the skill and answer the view. The skill is a folder on the
    shared file store and a skill_access row, so this runs as one unit on the
    file store's threads (run_efs_unit), never on anyio's default threads."""
    _sync_skill(scoped, branding)
    return _view(branding)


@router.get("", response_model=OrgBrandingView)
async def get_org_branding(request: Request, scope: ScopeDep):
    branding = await fetch_auth_branding(bearer_token=_credential(request, scope), **_who(scope))
    if not branding.reachable:
        # Nothing is known about the org and there is no recent answer to
        # serve. Never the stock look or enabled: false, which would flip it.
        if not branding.declined:
            # auth couldn't be asked: an outage, which the ingress alerts see.
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=READ_UNREACHABLE)
        # auth is up and answered a 4xx. reachable: false keeps the look, and
        # nothing about the org reaches the Brand guidelines skill.
        return _view(branding)
    return await run_efs_unit(partial(_settle, branding=branding), scope=scope)


@router.put("/appearance", response_model=OrgBrandingView)
async def put_org_appearance(body: OrgAppearanceUpdate, request: Request, scope: ScopeDep):
    credential = _credential(request, scope)
    if not is_session_jwt(credential):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=SIGN_IN_TO_EDIT)
    try:
        branding = await update_auth_appearance(
            bearer_token=credential, changes=body.auth_changes(), **_who(scope)
        )
    except AuthAppearanceError as refused:
        if refused.status_code == status.HTTP_401_UNAUTHORIZED:
            # auth didn't take the credential as a session: never proxy a 401,
            # which the app would read as its own sign-out.
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=SIGN_IN_TO_EDIT)
        if refused.status_code >= 500:
            logger.warning(
                "auth couldn't take an appearance change",
                extra={"event": "org_branding_appearance_save_failed", "status_code": refused.status_code},
            )
        raise HTTPException(status_code=refused.status_code, detail=refused.detail)
    return await run_efs_unit(partial(_settle, branding=branding), scope=scope)
