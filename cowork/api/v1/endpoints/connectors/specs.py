from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.common.settings.app_settings import OAuthSettings
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.db.units import run_db
from cowork.schemas.connectors import (
    ConnectorMetadataResponse,
    ConnectorSpecResponse,
    MatchRequest,
    MatchResponse,
)
from cowork.services.connectors.catalog import ConnectorCatalog
from cowork.services.connectors.oauth import auth_proxy
from cowork.services.connectors.specs._registry import registry

router = APIRouter()

# Same alias as connections.py/oauth.py: the vault/relay choice is per-request
# tenancy context, not a bare settings flag.
ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


# AuthenticatedInOrgMode, not OpenByDesign: in org mode this forwards to
# auth_proxy.proxy_catalogue, the same "relies on the request already having
# passed identity enforcement" shape as oauth.py/connections.py.
@router.get(
    "/",
    response_model=list[ConnectorMetadataResponse],
    dependencies=[Depends(require(AuthenticatedInOrgMode))],
)
async def list_connector_specs(
    scope: ScopeDep,
    request: Request,
    include_unavailable: bool = False,
):
    """List connector metadata.

    Desktop returns the whole registry. Org (cloud) mode returns only what
    auth's catalogue authorizes — unless `include_unavailable` is set, in
    which case the rest comes back too, flagged `cloud_available=False` so
    the caller can show them as desktop-only rather than pretend they don't
    exist. The default stays filtered so existing callers are unaffected.

    Desktop also lists the install's custom connectors. Org mode does not
    yet: no org definitions are written, and auth's catalogue would file
    them as desktop-only.
    """
    if not scope.org_mode:
        return await run_db(lambda session: ConnectorCatalog(session).list_connectors(), scope=scope)
    connectors = registry.list_connectors()
    # The full registry (~230 connectors, most with no OAuth relay or
    # org-mode save path) is a desktop concept. auth's catalogue is the
    # same allow-list list_connections already trusts.
    catalogue = await auth_proxy.proxy_catalogue(request, OAuthSettings())
    allowed_ids = {item["id"] for item in catalogue.get("items", [])}
    if not include_unavailable:
        return [c for c in connectors if c.id in allowed_ids]
    return [
        c.model_copy(update={"cloud_available": c.id in allowed_ids})
        for c in connectors
    ]


# AuthenticatedInOrgMode: the answer includes the scope's custom connectors,
# which are tenant data, so it is no longer the same for every caller.
@router.get(
    "/{connector_id}",
    response_model=ConnectorSpecResponse,
    dependencies=[Depends(require(AuthenticatedInOrgMode))],
)
async def get_connector_spec(connector_id: str, scope: ScopeDep):
    """Return one connector's full spec, built-in or custom.

    Raises:
        HTTPException: 404 when the scope has no connector with that id.
    """
    spec = await run_db(lambda session: ConnectorCatalog(session).get_connector(connector_id), scope=scope)
    if not spec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connector not found.")
    return spec


# AuthenticatedInOrgMode for the same reason as the lookup above.
@router.post("/match", response_model=MatchResponse, dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def match_connector_spec(req: MatchRequest, scope: ScopeDep) -> MatchResponse:
    """Match a free-text query against the scope's built-in and custom connectors."""
    return await run_db(
        lambda session: ConnectorCatalog(session).match_connector(req.query, req.max_candidates),
        scope=scope,
    )
