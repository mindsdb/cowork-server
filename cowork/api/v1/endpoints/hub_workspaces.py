"""The MindsHub workspace selector's two routes.

- GET  /            the gate, the caller's workspaces, and which one is active
- PUT  /active      switch the caller into another workspace

A MindsHub Workspace is an org-internal container that owns hub resources and
lives in the auth service. It is unrelated to the filesystem directories this
repo calls workspaces; the stored key is ``hub_workspace_id`` for that reason.

**This selector changes what the client shows, not what a turn is billed to.**
Which workspace a usage row carries is decided by the credential the turn
presents, and neither credential carries one today: a desktop turn runs against
a long-lived key bound to a user and an organization, and a cloud turn runs
against a short-TTL key whose mint body has no workspace field. So nothing here
touches the turn path, and a test asserts it.

**The stored key has a second writer, and it grants nothing.**
``hub_workspace_id`` is a declared ``UserSettings`` field, so
``PUT /api/v1/settings/hub_workspace_id`` writes it with no grant check at all.
That is fine rather than a hole, and only because of where the filtering lives:
``resolve_active`` matches the stored id against the listing auth returned, so an
id the caller holds no grant on resolves to the default workspace and the menu
renders the same rows either way. The stored value decides which row carries a
check, never which rows exist. Move the filtering and that stops being true.

So nothing that REFUSES a request may read it. The archived check on
``PUT /active`` reads the target row's own ``archived_at`` for exactly that
reason: phrasing it as "is this in the set the menu offered" would have gone
through ``selectable``, which keeps the active row even when archived, and one
call to the settings route would then have made an archived workspace current
and talked the refusal into accepting it.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlmodel import Session

from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.db.session import get_session
from cowork.principal import hub_credential
from cowork.schemas.hub_workspaces import (
    HubWorkspaceActivateRequest,
    HubWorkspaceRow,
    HubWorkspaceView,
)
from cowork.services.hub_workspaces import (
    authorization_ui_enabled,
    fetch_hub_workspaces,
    resolve_active,
    selectable,
)
from cowork.services.settings import SettingService

logger = logging.getLogger(__name__)

router = APIRouter()

SessionDep = Annotated[Session, Depends(get_session)]
ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]

SETTING_KEY = "hub_workspace_id"

# Switched off, so there is nothing to say about workspaces. Deliberately the
# same body a caller gets when the gate is off for them specifically: the client
# renders nothing either way, and a response that distinguished the two would
# tell an unflagged caller that a surface exists.
_DISABLED = HubWorkspaceView()


async def _build_view(
    request: Request, session: Session, scope: TenantScope
) -> HubWorkspaceView:
    """The selector's whole state, gate first.

    The gate is read before the listing, not alongside it, so a switched-off app
    makes no workspace request at all. That is worth one sequential round trip:
    the alternative asks auth for a listing nobody is going to render.
    """
    bearer = hub_credential(request)
    org_id = scope.org_id or ""
    user_id = scope.user_id or ""
    if not await authorization_ui_enabled(bearer_token=bearer, org_id=org_id, user_id=user_id):
        return _DISABLED

    listing = await fetch_hub_workspaces(bearer_token=bearer, org_id=org_id, user_id=user_id)
    stored = SettingService(session, scope).load().hub_workspace_id
    active = resolve_active(listing.workspaces, stored)
    active_id = active.id if active else None
    return HubWorkspaceView(
        enabled=True,
        reachable=listing.reachable,
        workspaces=[
            HubWorkspaceRow.model_validate(workspace, from_attributes=True)
            for workspace in selectable(listing.workspaces, active_id)
        ],
        active_workspace_id=active_id,
    )


@router.get("/", response_model=HubWorkspaceView)
async def get_hub_workspaces(
    request: Request, session: SessionDep, scope: ScopeDep
) -> HubWorkspaceView:
    return await _build_view(request, session, scope)


@router.put("/active", response_model=HubWorkspaceView)
async def set_active_hub_workspace(
    body: HubWorkspaceActivateRequest,
    request: Request,
    session: SessionDep,
    scope: ScopeDep,
) -> HubWorkspaceView:
    """Switch the caller into another workspace.

    The id has to appear in the caller's own listing before it is stored. Auth
    already decides who may see which workspace and the listing is the answer, so
    checking against it costs nothing and means a caller cannot store a workspace
    from another organization by asking for it directly. An unreachable listing
    refuses rather than storing an id nobody could verify: a stored id that names
    nothing resolves to the default workspace on the next read, which would look
    like the switch silently doing nothing.

    Two refusals rather than one, because "you may not" and "not any more" are
    different answers and the client can only act on one of them. A workspace
    missing from the listing is a grant the caller does not hold, and that is a
    403. A workspace in the listing but stamped archived is a 409, so the UI can
    say retrying will not help instead of offering a loop with no exit.
    """
    bearer = hub_credential(request)
    org_id = scope.org_id or ""
    user_id = scope.user_id or ""
    if not await authorization_ui_enabled(bearer_token=bearer, org_id=org_id, user_id=user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="workspace selection is not enabled",
        )

    listing = await fetch_hub_workspaces(bearer_token=bearer, org_id=org_id, user_id=user_id)
    if not listing.reachable:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="could not reach MindsHub to confirm the workspace",
        )
    if not any(workspace.id == body.workspace_id for workspace in listing.workspaces):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="no access to that workspace",
        )

    # Archived is refused on the target's own flag, deliberately NOT by asking
    # whether it is in `selectable(...)`. `selectable` keeps the active row even
    # when archived, so the set it returns depends on the stored pick, and the
    # stored pick has a second writer that grants nothing (see this module's
    # header): `PUT /api/v1/settings/hub_workspace_id` writes the same key with
    # no gate and no listing check. Reading the offered set here would have let
    # one call to that route make an archived workspace "current", and the
    # refusal below would then wave it through. Nobody loses anything by reading
    # the flag instead: re-selecting the archived workspace you are already in is
    # a no-op, and every other row in the menu is live.
    target = next(w for w in listing.workspaces if w.id == body.workspace_id)
    if target.archived_at:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="that workspace is archived",
        )

    SettingService(session, scope).upsert_setting(SETTING_KEY, body.workspace_id)
    return await _build_view(request, session, scope)
