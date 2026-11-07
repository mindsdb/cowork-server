from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import ScopedSessionDep
from cowork.schemas.connectors import ConnectorMetadataResponse, CustomConnectorUpdate
from cowork.services.connectors.catalog import custom_connector_spec
from cowork.services.connectors.custom_connectors import CustomConnectorService, UnstorableFormError
from cowork.services.connectors.specs._registry import connector_metadata

# In org mode a custom connector is visible to its builder only (see
# scope_definitions), so the member who built it manages it. A no-op on a
# local install, where the one user owns every definition.
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])

#: What a member may change on their own connector in org mode. Its form came
#: from a checked draft and stays as saved; featured is a directory setting.
_ORG_MODE_CHANGES = frozenset({"label", "description", "category"})


@router.patch("/{connector_id}", response_model=ConnectorMetadataResponse)
def update_custom_connector(
    connector_id: str, body: CustomConnectorUpdate, session: ScopedSessionDep,
) -> ConnectorMetadataResponse:
    """Rename, re-describe, recategorize, unfeature or re-form a custom connector.

    Raises:
        HTTPException: 404 when the scope has no such custom connector, 422
            when ``spec`` is invalid or unsafe to keep, or in org mode when
            anything but label, description or category is sent.
    """
    if session.scope.org_mode and not body.model_fields_set <= _ORG_MODE_CHANGES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only label, description and category can be changed here.",
        )
    try:
        row = CustomConnectorService(session).update(connector_id, body)
    except UnstorableFormError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Custom connector not found.")
    return connector_metadata(custom_connector_spec(row))


@router.delete("/{connector_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_custom_connector(connector_id: str, session: ScopedSessionDep) -> Response:
    """Delete a custom connector's definition; saved connections stay and can be disconnected.

    Raises:
        HTTPException: 404 when the scope has no such custom connector.
    """
    if not CustomConnectorService(session).delete(connector_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Custom connector not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
