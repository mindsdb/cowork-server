"""Cloud datasource connection management, relayed to auth.

Every route here is a relay. Nothing on this path opens a session, touches a
vault or stages a submission: auth holds the encrypted credential and derives
owner and org from the caller's own bearer, so no identity is ever read from
the body or the query. Outside org mode the whole surface answers 404.

Capture obeys the deployment's capability policy, the same one the submission
relay and the capability response read, so a method this deployment does not
run cannot be stored through the side door. Reading and deleting stay open, so
switching a method off never traps a connection captured while it was on.

Create, edit and retry then drive one validation attempt and answer with what
auth records, because a connection nobody probed is pending forever and the
interface excludes it from every conversation.
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
from cowork.services.connectors.datasource_validation import validate_connection
from cowork.services.connectors.datasources import (
    normalize_datasource_input,
    require_cloud_method_enabled,
)
from cowork.services.connectors.oauth import auth_proxy

ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


def _require_org(scope: ScopeDep) -> None:
    """Refuse the whole surface outside org mode: a desktop caller gets 404.

    A router dependency rather than a handler call, because FastAPI validates
    the body first: an in-handler check lets a malformed desktop request
    answer 422 and advertise the routes and their field names.
    """
    if not scope.org_mode:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="not available outside org deployments"
        )


router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode)), Depends(_require_org)])


async def _validated(connection: dict, request: Request) -> dict:
    """Validate a connection auth has just stored, and answer with the outcome.

    A connection auth creates is pending until a probe says otherwise, so a
    capture that returned here would report a state the user cannot act on and
    the interface hides. The attempt is best effort: when it could not run, the
    connection stands as auth left it and the caller can retry.
    """
    connection_id = connection.get("id")
    if not isinstance(connection_id, int):
        return connection
    outcome = await validate_connection(connection_id, request.headers.get("authorization", ""))
    if not outcome.ran:
        return connection
    checked = await auth_proxy.proxy_datasource_detail(connection_id, request, OAuthSettings())
    # Auth's row says only that validation failed. The gateway's code is what
    # lets a caller offer the user something to do about it.
    return {**checked, "validation_code": outcome.code} if outcome.code else checked


@router.post("/", response_model=DatasourceConnectionResponse, status_code=status.HTTP_201_CREATED)
async def create_datasource_connection(
    body: DatasourceCreateRequest, request: Request
) -> DatasourceConnectionResponse:
    """Store a new datasource credential in auth's encrypted vault."""
    require_cloud_method_enabled(body.connector_id, body.method)
    payload = normalize_datasource_input(body)
    result = await auth_proxy.proxy_datasource_create(request, OAuthSettings(), payload)
    return DatasourceConnectionResponse.model_validate(await _validated(result, request))


@router.get("/", response_model=list[DatasourceConnectionResponse])
async def list_datasource_connections(request: Request) -> list[DatasourceConnectionResponse]:
    """List the caller's own datasource connections as masked metadata."""
    items = await auth_proxy.proxy_datasource_list(request, OAuthSettings())
    return [DatasourceConnectionResponse.model_validate(item) for item in items]


@router.get("/{connection_id}", response_model=DatasourceConnectionResponse)
async def get_datasource_connection(connection_id: int, request: Request) -> DatasourceConnectionResponse:
    """Read masked metadata for one owned connection."""
    result = await auth_proxy.proxy_datasource_detail(connection_id, request, OAuthSettings())
    return DatasourceConnectionResponse.model_validate(result)


@router.patch("/{connection_id}", response_model=DatasourceConnectionResponse)
async def edit_datasource_connection(
    connection_id: int, body: DatasourceEditRequest, request: Request
) -> DatasourceConnectionResponse:
    """Edit a connection, guarded by the revision the caller read."""
    require_cloud_method_enabled(body.connector_id, body.method)
    payload = normalize_datasource_input(body)
    payload["expected_revision"] = body.expected_revision
    result = await auth_proxy.proxy_datasource_edit(connection_id, request, OAuthSettings(), payload)
    # Auth keeps a connection verified only when the edit left its credential
    # untouched, as a rename does; probing it again would dial the same database.
    if result.get("status") == "verified":
        return DatasourceConnectionResponse.model_validate(result)
    return DatasourceConnectionResponse.model_validate(await _validated(result, request))


@router.delete("/{connection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_datasource_connection(connection_id: int, request: Request) -> Response:
    """Delete an owned connection and its stored credential."""
    await auth_proxy.proxy_datasource_delete(connection_id, request, OAuthSettings())
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{connection_id}/validation-retry", response_model=DatasourceConnectionResponse)
async def retry_datasource_validation(connection_id: int, request: Request) -> DatasourceConnectionResponse:
    """Start a new validation attempt for an owned connection."""
    result = await auth_proxy.proxy_datasource_retry(connection_id, request, OAuthSettings())
    return DatasourceConnectionResponse.model_validate(await _validated(result, request))
