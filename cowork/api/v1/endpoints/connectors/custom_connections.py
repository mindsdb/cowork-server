"""A member's custom API connections, relayed to auth.

Listing and deleting are relays on the member's own bearer: auth derives the
owner and organization from it. They stay open while the connector setting is
off, so a member can always see and remove what is stored. Nothing here
creates a connection; the connect route does, only after the check passed.
Outside org mode the whole surface answers 404.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.common.settings.app_settings import OAuthSettings
from cowork.db.scoped import ScopedSession, TenantScope, get_tenant_scope
from cowork.db.units import run_db
from cowork.models.custom_connector import CustomConnector
from cowork.schemas.connectors import CustomConnectionResponse
from cowork.services.connectors.custom_connectors import scope_definitions
from cowork.services.connectors.oauth import auth_proxy

ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


async def _require_org(scope: ScopeDep) -> None:
    """Refuse the whole surface outside org mode, before any body is read."""
    if not scope.org_mode:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not available outside org deployments")


router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode)), Depends(_require_org)])


def _own_template_ids(session: ScopedSession) -> set[str]:
    """The connector ids of the caller's own saved templates."""
    return set(session.exec(scope_definitions(session).with_only_columns(CustomConnector.connector_id)).all())


@router.get("/", response_model=list[CustomConnectionResponse])
async def list_custom_connections(request: Request, scope: ScopeDep) -> list[CustomConnectionResponse]:
    """List the caller's custom API connections, marking those whose template is gone."""
    items = await auth_proxy.proxy_custom_connection_list(request, OAuthSettings())
    own = await run_db(_own_template_ids, scope=scope)
    return [
        CustomConnectionResponse.model_validate({**item, "template_missing": item.get("connector_id") not in own})
        for item in items
    ]


@router.delete("/{connection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_custom_connection(connection_id: int, request: Request) -> Response:
    """Delete one of the caller's custom API connections; auth answers 404 for anyone else's."""
    await auth_proxy.proxy_custom_connection_delete(connection_id, request, OAuthSettings())
    return Response(status_code=status.HTTP_204_NO_CONTENT)
