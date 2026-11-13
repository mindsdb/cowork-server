from __future__ import annotations

from typing import Annotated, Any

from anton.core.tools.connector_draft import ConnectorDraftError, check_draft_text, parse_connector_draft
from fastapi import APIRouter, Body, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.common.settings.app_settings import TurnQueueSettings
from cowork.db.scoped import ScopedSession, ScopedSessionDep, TenantScope, get_tenant_scope
from cowork.db.units import run_db
from cowork.models.custom_connector import CustomConnector
from cowork.schemas.connectors import (
    ConnectorMetadataResponse,
    CustomConnectionResponse,
    CustomConnectorUpdate,
    CustomConnectRequest,
)
from cowork.services.connectors import custom_connect
from cowork.services.connectors.catalog import custom_connector_spec
from cowork.services.connectors.custom_connections import (
    CustomConnectionRefused,
    CustomConnectionUnavailable,
    create_custom_connection,
)
from cowork.services.connectors.custom_connectors import (
    CustomConnectorService,
    UnstorableFormError,
    is_builtin_connector_id,
)
from cowork.services.connectors.specs._registry import connector_metadata
from cowork.turnqueue.redis_client import get_redis

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


def _check_member_edit(body: CustomConnectorUpdate) -> None:
    """Check an org-mode edit's texts with the draft rules they were saved under.

    Raises:
        HTTPException: 422 naming the field, without its value.
    """
    for key in sorted(body.model_fields_set & _ORG_MODE_CHANGES):
        value = getattr(body, key)
        if value is None:
            continue
        try:
            checked = check_draft_text(key, value)
        except ConnectorDraftError as e:
            raise _unprocessable(str(e)) from e
        if key == "label" and checked is None:
            raise _unprocessable("label is required.")


@router.post("", response_model=ConnectorMetadataResponse)
def save_connector_draft(
    draft: Annotated[dict[str, Any], Body()], session: ScopedSessionDep,
) -> ConnectorMetadataResponse:
    """Save a connector draft from a chat card as the member's own connector.

    Org mode only, for an authenticated member, and only while the setting
    that lets cloud turns build drafts is on. The draft is untrusted client
    data and is checked again here with the parser the pod used.

    Raises:
        HTTPException: 404 outside org mode or with the setting off, 422 for
            a draft outside the allowlist or a built-in id.
    """
    if not session.scope.org_mode or not TurnQueueSettings().connector_drafts_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")
    try:
        parsed = parse_connector_draft(draft)
    except ConnectorDraftError as e:
        raise _unprocessable(str(e)) from e
    connector_id = parsed.connector_id
    if is_builtin_connector_id(connector_id):
        raise _unprocessable(f"{connector_id!r} is already a built-in connector.")
    try:
        row = CustomConnectorService(session).save_member_draft(parsed)
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
        if CustomConnectorService(session).get(connector_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Custom connector not found.")
        _check_member_edit(body)
        for key in _ORG_MODE_CHANGES:
            if isinstance(getattr(body, key), str):
                setattr(body, key, getattr(body, key).strip())
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


def _connect_template(
    connector_id: str, session: ScopedSession
) -> tuple[list[dict[str, Any]], custom_connect.StoredTemplate]:
    """The caller's own template's fields, and its check path and auth section parsed again.

    Raises:
        custom_connect.ConnectRefused: 404 when the caller has no such
            template, 409 ``no_check`` when it has no usable check or auth section.
    """
    row = CustomConnectorService(session).get(connector_id)
    if row is None:
        raise custom_connect.ConnectRefused("not_found", 404, "Custom connector not found.")
    return list(row.spec.get("fields", [])), custom_connect.stored_template(row)


#: What auth's refusals mean to the member; any other code reads as the fallback.
_AUTH_REFUSALS = {
    "conflict": "You already have a connection with this name. Choose another name, or remove the old one in Connect Apps and Data.",
    "quota_exceeded": "You have reached the limit of custom connections. Remove one in Connect Apps and Data to connect another.",
    "organization_quota_exceeded": "Your organization has reached its limit of custom connections. An admin or another member must remove one before you can connect another.",
    "organization_mismatch": "Your active organization changed while connecting. Reload and try again.",
    "disabled": "Custom connections are turned off on this deployment.",
}


def _refusal(code: str, status_code: int, message: str) -> JSONResponse:
    """A refusal as ``{"detail": {"code", "message"}}``; never a submitted value."""
    return JSONResponse({"detail": {"code": code, "message": message}}, status_code=status_code)


@router.post("/{connector_id}/connect", response_model=CustomConnectionResponse, status_code=status.HTTP_201_CREATED)
async def connect_custom_connector(
    connector_id: str,
    body: CustomConnectRequest,
    request: Request,
    scope: Annotated[TenantScope, Depends(get_tenant_scope)],
) -> CustomConnectionResponse | JSONResponse:
    """Check a member's credentials for their own template at the address they typed, then store them.

    Org mode only, with the setting on and the turn queue on Redis, which
    counts each member's attempts. auth is asked first whether it would take
    the connection (dry run), then the check runs against the typed address
    only, then auth stores it verified.

    Returns:
        201 with the stored connection. A refusal answers its own status with
        ``detail`` ``{code, message}``; a check that ran and failed answers
        422 with ``detail`` ``{code, provider_status, excerpt}``.
    """
    settings = TurnQueueSettings()
    if not scope.org_mode or not settings.connector_drafts_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")
    try:
        if not settings.is_remote:
            raise custom_connect.ConnectRefused("unavailable", 503, "Connecting is unavailable on this deployment.")
        await custom_connect.take_attempt(get_redis(), f"cowork:custom-connect:{scope.org_id}:{scope.user_id}")
        fields, template = await run_db(lambda session: _connect_template(connector_id, session), scope=scope)
        origin = custom_connect.normalize_address(body.address, settings.custom_connect_refused_domains)
        values = custom_connect.check_values(fields, body.values)
        custom_connect.place_values(template.auth, values)
    except custom_connect.ConnectRefused as refused:
        return _refusal(refused.code, refused.status_code, refused.message)
    payload = {
        "connector_id": connector_id,
        "name": body.name,
        "fields": {**values, custom_connect.BASE_URL_FIELD: origin},
        "secure_keys": sorted(values),
        "auth": template.auth,
    }
    credential = request.headers.get("authorization", "")
    try:
        await create_custom_connection(
            credential=credential, organization_id=str(scope.org_id), payload=payload, dry_run=True, settings=settings,
        )
        failure = await custom_connect.run_check(origin, template, values)
        if failure is not None:
            return JSONResponse(
                {"detail": {"code": failure.code, "provider_status": failure.provider_status, "excerpt": failure.excerpt}},
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            )
        stored = await create_custom_connection(
            credential=credential, organization_id=str(scope.org_id), payload=payload, dry_run=False, settings=settings,
        )
    except CustomConnectionRefused as refused:
        message = _AUTH_REFUSALS.get(refused.code, "The connection could not be stored.")
        return _refusal(refused.code, refused.status_code, message)
    except CustomConnectionUnavailable:
        return _refusal("unavailable", status.HTTP_503_SERVICE_UNAVAILABLE, "Connecting is unavailable right now; try again later.")
    return CustomConnectionResponse.model_validate(stored)
