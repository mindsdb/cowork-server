"""GET and POST /hub/deletion-request/: the caller's request to delete their
MindsHub account.

The app's way to start an account deletion. MindsHub records the request and
its support deletes the account (cowork.services.mindshub_account_deletion).
Asked with the Authorization the gateway validated
(cowork.principal.caller_bearer), never the client-set
X-MindsHub-Authorization, which could name another account.

Outside MindsHub mode there is no MindsHub account here to delete, so both
routes answer 404, as GET /hub/teams/ answers no teams there.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.principal import caller_bearer
from cowork.schemas.mindshub_account_deletion import MindsHubDeletionRequestView
from cowork.services.mindshub_account_deletion import (
    DeletionRequestError,
    read_deletion_request,
    request_account_deletion,
)

# Gated as the teams router is: MindsHub answers per caller, and refuses an
# absent or invalid credential on its own.
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])

ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]

NO_MINDSHUB_ACCOUNT = "There is no MindsHub account here to delete."
NO_OPEN_REQUEST = "No account deletion request is open."


def _bearer(request: Request, scope: TenantScope | None) -> str:
    """The gateway's credential, or a 404 outside MindsHub mode."""
    bearer = caller_bearer(request)
    if scope is None or not scope.org_mode or not bearer:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=NO_MINDSHUB_ACCOUNT)
    return bearer


@router.get("/", response_model=MindsHubDeletionRequestView)
async def get_deletion_request(request: Request, scope: ScopeDep) -> MindsHubDeletionRequestView:
    bearer = _bearer(request, scope)
    try:
        view = await read_deletion_request(bearer_token=bearer)
    except DeletionRequestError as failed:
        raise HTTPException(status_code=failed.status_code, detail=failed.detail) from None
    if view is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=NO_OPEN_REQUEST)
    return view


@router.post("/", response_model=MindsHubDeletionRequestView, status_code=status.HTTP_202_ACCEPTED)
async def post_deletion_request(request: Request, scope: ScopeDep) -> MindsHubDeletionRequestView:
    bearer = _bearer(request, scope)
    try:
        return await request_account_deletion(bearer_token=bearer)
    except DeletionRequestError as failed:
        raise HTTPException(status_code=failed.status_code, detail=failed.detail) from None
