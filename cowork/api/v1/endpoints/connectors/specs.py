from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from cowork.common.settings.app_settings import OAuthSettings
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.schemas.connectors import (
    ConnectorMetadataResponse,
    ConnectorSpecResponse,
    MatchRequest,
    MatchResponse,
)
from cowork.services.connectors.oauth import auth_proxy
from cowork.services.connectors.specs._registry import registry

router = APIRouter()

# Same alias as connections.py/oauth.py: the vault/relay choice is per-request
# tenancy context, not a bare settings flag.
ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


@router.get("/", response_model=list[ConnectorMetadataResponse])
async def list_connector_specs(scope: ScopeDep, request: Request):
    connectors = registry.list_connectors()
    if scope.org_mode:
        # The full registry (~230 connectors, most with no OAuth relay or
        # org-mode save path) is a desktop concept. auth's catalogue is the
        # same allow-list list_connections already trusts — restrict the
        # directory cowork's ConnectorPicker renders to it too.
        catalogue = await auth_proxy.proxy_catalogue(request, OAuthSettings())
        allowed_ids = {item["id"] for item in catalogue.get("items", [])}
        connectors = [c for c in connectors if c.id in allowed_ids]
    return connectors


@router.get("/{connector_id}", response_model=ConnectorSpecResponse)
def get_connector_spec(connector_id: str):
    spec = registry.get_connector(connector_id)
    if not spec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connector not found.")
    return spec


@router.post("/match", response_model=MatchResponse)
def match_connector_spec(req: MatchRequest) -> MatchResponse:
    return registry.match_connector(req.query, req.max_candidates)
