from __future__ import annotations

from typing import Annotated, Any

from anton.core.tools.connector_draft import ConnectorDraftError, parse_connector_draft
from fastapi import APIRouter, Body, Depends, HTTPException, Response, status

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import ScopedSessionDep
from cowork.models.custom_connector import CustomConnector
from cowork.schemas.connectors import ConnectorMetadataResponse, CustomConnectorUpdate
from cowork.services.connectors.catalog import custom_connector_spec
from cowork.services.connectors.custom_connectors import (
    CustomConnectorService,
    MemberConnectorLimitError,
    UnstorableFormError,
)
from cowork.services.connectors.datasource_capabilities import load_datasource_capabilities
from cowork.services.connectors.specs._registry import connector_metadata, registry

# In org mode a custom connector is visible to its builder only (see
# scope_definitions), so the member who built it manages it. A no-op on a
# local install, where the one user owns every definition.
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])

#: What a member may change on their own connector in org mode. Its form came
#: from a checked draft and stays as saved; featured is a directory setting.
_ORG_MODE_CHANGES = frozenset({"label", "description", "category"})


def _unprocessable(detail: str) -> HTTPException:
    """A 422 with ``detail``, which never quotes a submitted value."""
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)


def _check_member_edit(row: CustomConnector, body: CustomConnectorUpdate) -> None:
    """Check an org-mode edit's texts with the draft rules they were saved under.

    The parser checks a whole draft, so the row's id and one stand-in field
    complete it; only the label, description and category are under test.

    Raises:
        HTTPException: 422 naming the field, without its value.
    """
    sent = body.model_fields_set
    draft = {
        "connector_id": row.connector_id,
        "label": body.label if "label" in sent and body.label is not None else row.label,
        "fields": [{"name": "field", "label": "Field"}],
    }
    for key in ("description", "category"):
        if key in sent and getattr(body, key) is not None:
            draft[key] = getattr(body, key)
    try:
        parse_connector_draft(draft)
    except ConnectorDraftError as e:
        raise _unprocessable(str(e)) from e


@router.post("", response_model=ConnectorMetadataResponse)
def save_connector_draft(
    draft: Annotated[dict[str, Any], Body()], session: ScopedSessionDep,
) -> ConnectorMetadataResponse:
    """Save a connector draft from a chat card as the member's own connector.

    Org mode only, for an authenticated member; the draft is untrusted client
    data and is checked again here with the parser the pod used.

    Raises:
        HTTPException: 404 outside org mode, 422 for a draft outside the
            allowlist or a built-in id, 409 at the member's connector limit.
    """
    if not session.scope.org_mode:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")
    try:
        parsed = parse_connector_draft(draft)
    except ConnectorDraftError as e:
        raise _unprocessable(str(e)) from e
    connector_id = parsed.connector_id
    if registry.get_connector(connector_id) or load_datasource_capabilities().is_cloud_datasource(connector_id):
        raise _unprocessable(f"{connector_id!r} is already a built-in connector.")
    try:
        row = CustomConnectorService(session).save_member_draft(parsed)
    except MemberConnectorLimitError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    except UnstorableFormError as e:
        raise _unprocessable(str(e)) from e
    return connector_metadata(custom_connector_spec(row))


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
    if session.scope.org_mode:
        if not body.model_fields_set <= _ORG_MODE_CHANGES:
            raise _unprocessable("Only label, description and category can be changed here.")
        row = CustomConnectorService(session).get(connector_id)
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Custom connector not found.")
        _check_member_edit(row, body)
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
