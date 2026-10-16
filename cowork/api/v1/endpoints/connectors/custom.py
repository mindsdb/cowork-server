from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status

from cowork.api.v1.permissions import AuthenticatedOrgAdmin, require
from cowork.db.scoped import ScopedSessionDep
from cowork.models.custom_connector import CustomConnector
from cowork.schemas.connectors import ConnectorMetadataResponse, CustomConnectorUpdate
from cowork.services.connectors.custom_connectors import CustomConnectorService, UnstorableFormError

# AuthenticatedOrgAdmin: a custom connector is listed for everyone in the
# organization, so changing or removing one is an admin decision. A no-op on a
# local install, where the one user owns every definition.
router = APIRouter(dependencies=[Depends(require(AuthenticatedOrgAdmin))])


def _metadata(row: CustomConnector) -> ConnectorMetadataResponse:
    return ConnectorMetadataResponse(
        id=row.connector_id,
        label=row.label,
        description=row.description,
        category=row.category or "other",
        logo_color=row.logo_color,
        featured=row.featured,
        custom=True,
    )


@router.patch("/{connector_id}", response_model=ConnectorMetadataResponse)
def update_custom_connector(
    connector_id: str, body: CustomConnectorUpdate, session: ScopedSessionDep,
) -> ConnectorMetadataResponse:
    """Rename, re-describe, recategorize, unfeature or re-form a custom connector.

    Raises:
        HTTPException: 404 when the scope has no such custom connector, 422
            when ``spec`` is invalid or unsafe to keep.
    """
    try:
        row = CustomConnectorService(session).update(connector_id, body.model_dump(exclude_none=True))
    except UnstorableFormError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Custom connector not found.")
    return _metadata(row)


@router.delete("/{connector_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_custom_connector(connector_id: str, session: ScopedSessionDep) -> Response:
    """Delete a custom connector's definition; saved connections stay and can be disconnected.

    Raises:
        HTTPException: 404 when the scope has no such custom connector.
    """
    if not CustomConnectorService(session).delete(connector_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Custom connector not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
