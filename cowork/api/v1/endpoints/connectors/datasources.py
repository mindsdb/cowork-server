"""Cloud datasource connection management, relayed to auth.

Every route here is a relay. Nothing on this path opens a session, touches a
vault or stages a submission: auth holds the encrypted credential and derives
owner and org from the caller's own bearer, so no identity is ever read from
the body or the query. Outside org mode the whole surface answers 404.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.common.settings.app_settings import OAuthSettings
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.schemas.connectors import (
    DatasourceConnectionResponse,
    DatasourceCreateRequest,
    DatasourceEditRequest,
)
from cowork.services.connectors.datasources import normalize_datasource_input
from cowork.services.connectors.oauth import auth_proxy

router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])

ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


def _require_org(scope: TenantScope) -> None:
    """Make the org branch explicit in the handler, not only in the router.

    Copies the shape of oauth.py's _require_picker_engine: a desktop caller
    gets 404 rather than a relay attempt against an auth it has no credential
    for.
    """
    if not scope.org_mode:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="not available outside org deployments"
        )


@router.post("/", response_model=DatasourceConnectionResponse, status_code=status.HTTP_201_CREATED)
async def create_datasource_connection(
    body: DatasourceCreateRequest, scope: ScopeDep, request: Request
) -> DatasourceConnectionResponse:
    """Store a new datasource credential in auth's encrypted vault."""
    _require_org(scope)
    payload = normalize_datasource_input(body)
    result = await auth_proxy.proxy_datasource_create(request, OAuthSettings(), payload)
    return DatasourceConnectionResponse.model_validate(result)


@router.get("/", response_model=list[DatasourceConnectionResponse])
async def list_datasource_connections(
    scope: ScopeDep, request: Request
) -> list[DatasourceConnectionResponse]:
    """List the caller's own datasource connections as masked metadata."""
    _require_org(scope)
    items = await auth_proxy.proxy_datasource_list(request, OAuthSettings())
    return [DatasourceConnectionResponse.model_validate(item) for item in items]


@router.get("/{connection_id}", response_model=DatasourceConnectionResponse)
async def get_datasource_connection(
    connection_id: int, scope: ScopeDep, request: Request
) -> DatasourceConnectionResponse:
    """Read masked metadata for one owned connection."""
    _require_org(scope)
    result = await auth_proxy.proxy_datasource_detail(connection_id, request, OAuthSettings())
    return DatasourceConnectionResponse.model_validate(result)


@router.patch("/{connection_id}", response_model=DatasourceConnectionResponse)
async def edit_datasource_connection(
    connection_id: int, body: DatasourceEditRequest, scope: ScopeDep, request: Request
) -> DatasourceConnectionResponse:
    """Replace a connection's credential, guarded by the version the caller saw."""
    _require_org(scope)
    payload = normalize_datasource_input(body)
    payload["expected_version"] = body.expected_version
    result = await auth_proxy.proxy_datasource_edit(connection_id, request, OAuthSettings(), payload)
    return DatasourceConnectionResponse.model_validate(result)


@router.delete("/{connection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_datasource_connection(
    connection_id: int, scope: ScopeDep, request: Request
) -> Response:
    """Delete an owned connection and its stored credential."""
    _require_org(scope)
    await auth_proxy.proxy_datasource_delete(connection_id, request, OAuthSettings())
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{connection_id}/validation-retry", response_model=DatasourceConnectionResponse)
async def retry_datasource_validation(
    connection_id: int, scope: ScopeDep, request: Request
) -> DatasourceConnectionResponse:
    """Start a new validation attempt for an owned connection."""
    _require_org(scope)
    result = await auth_proxy.proxy_datasource_retry(connection_id, request, OAuthSettings())
    return DatasourceConnectionResponse.model_validate(result)
