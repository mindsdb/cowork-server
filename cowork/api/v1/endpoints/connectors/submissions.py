"""
DISCLAIMER: The probe for connectors will always run through Anton regardless of the harness used.
"""


from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from cowork.api.v1.permissions import AuthenticatedInOrgMode, SingleUserOnly, require
from cowork.db.scoped import TenantScope, get_tenant_scope, tenant_scope_required
from cowork.db.units import run_db
from cowork.handlers.datasource_relay import relay_cloud_submission
from cowork.handlers.probe import ProbeHandler
from cowork.schemas.connectors import ConnectorField, ConnectorTestAnswer, InvalidConnectorIdError, SubmitFormRequest
from cowork.services.connectors.catalog import ConnectorCatalog
from cowork.services.connectors.datasource_capabilities import load_datasource_capabilities
from cowork.services.connectors.custom_connectors import (
    NEEDS_REBUILD_FOR_MEMBER,
    UnstorableFormError,
    check_handcrafted_form,
    handcrafted_form_allowed,
    has_test_requests,
)
from cowork.services.connectors.connections import ConnectionsService
from cowork.services.connectors.connector_test_flow import (
    CANCELLED,
    NOT_ALLOWED,
    advance,
    failed_patch,
    record_outcome,
    save_confirmed,
    saved_patch,
)
from cowork.services.connectors.connector_test_run import ConnectorTestError
from cowork.services.connectors.layered_vault import SHARED_CONNECTION_REFUSAL
from cowork.services.connectors.persist import connection_a_new_save_replaces, vault_for_scope
from cowork.services.connectors.sharepoint import ENGINE as SHAREPOINT, generic_setup_refusal
from cowork.services.connectors.specs._registry import registry
from cowork.services.connectors.pending_tests import pending_tests
from cowork.services.connectors.submissions import missing_required_fields, store

logger = logging.getLogger(__name__)

# AuthenticatedInOrgMode: in org mode a request with no verified principal is
# refused with 401 before the form is read. The stream holds no request
# session: the probe handler's and the cloud relay's database units open
# their own tenant-scoped sessions, so no connection stays checked out while
# the probe waits on the model or the relay waits on auth.
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])
TenantScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]
_DATABASE_CATEGORY = "database"
_GONE = "This test result is no longer waiting. Submit the form again."
_NO_CUSTOM_IN_ORG = (
    "Custom connectors cannot be connected in an organization yet. "
    "Use a connector from Connect Apps and Data."
)


def _resolve_fields(spec, method_id: str | None) -> list:
    form = spec.form
    methods = form.methods or []
    if methods:
        if not method_id:
            return []
        method_def = next((m for m in methods if m.id == method_id), None)
        return list(method_def.fields or []) if method_def else []
    return list(form.fields or [])


def _fields_from_spec_dict(form_spec: dict, method_id: str | None) -> list[ConnectorField]:
    # Reads fields leniently; submit_form has already checked the form
    # itself with check_handcrafted_form.
    methods = form_spec.get("methods") or []
    if methods:
        method_def = next((m for m in methods if isinstance(m, dict) and m.get("id") == method_id), None)
        raw = (method_def or {}).get("fields") or []
    else:
        raw = form_spec.get("fields") or []
    fields = []
    for f in raw:
        if isinstance(f, dict) and f.get("name"):
            fields.append(ConnectorField(
                name=f["name"],
                label=f.get("label") or f["name"],
                type=f.get("type") or "text",
                required=bool(f.get("required", False)),
                secret=bool(f.get("secret", False)),
            ))
    return fields


def _connector_id(req: SubmitFormRequest) -> str:
    """Return the submission's connector id.

    Raises the HTTP error a malformed id earns: 422 for an invalid id, 400
    when the form names none.
    """
    try:
        return req.resolve_connector_id()
    except InvalidConnectorIdError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


def _relayed_in_org_mode(connector_id: str) -> bool:
    """True when an org's credential for this connector may go only to auth's store.

    That is a connector with a cloud datasource form, or any built-in
    database connector; the relay refuses those it has no enabled method for.
    """
    if load_datasource_capabilities().is_cloud_datasource(connector_id):
        return True
    spec = registry.get_connector(connector_id)
    return spec is not None and spec.category == _DATABASE_CATEGORY


@router.post("/", name="submit_form")
async def submit_form_route(req: SubmitFormRequest, request: Request, scope: TenantScopeDep) -> StreamingResponse:
    """Answer one form submission.

    In org mode a cloud datasource or built-in database submission goes to
    the relay, which stores an enabled method in auth's encrypted store and
    refuses the rest, so no database password is saved locally. Every other
    submission takes submit_form. The relay needs ``request`` to forward the
    caller's bearer to auth.
    """
    if scope.org_mode:
        connector_id = _connector_id(req)
        if _relayed_in_org_mode(connector_id):
            return await relay_cloud_submission(req, connector_id, req.resolve_method(), scope, request)
    return await submit_form(req, scope)


@router.post(
    "/allow-private", name="submission_allow_private", dependencies=[Depends(require(SingleUserOnly))],
)
async def allow_private_route(answer: ConnectorTestAnswer, scope: TenantScopeDep) -> dict:
    """Continue a custom connector's test after the member allows a private host.

    The approval covers the host, port and addresses the member was shown, for
    this test only. Answers with the next host to allow, the preview to
    confirm, or the failure patch; 404 when the id is unknown, used, expired
    or another scope's.
    """
    entry = pending_tests.pop(answer.approval_id or "", "approval", scope)
    if entry is None or entry.asking is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_GONE)
    key, addresses = entry.asking
    entry.approved[key] = addresses
    try:
        return await advance(entry)
    except ConnectorTestError as exc:
        await record_outcome(entry, f"Not saved: {exc}")
        return {"patch": failed_patch(entry.form_id, str(exc))}


@router.post("/confirm", name="submission_confirm", dependencies=[Depends(require(SingleUserOnly))])
async def confirm_route(answer: ConnectorTestAnswer, scope: TenantScopeDep) -> dict:
    """Save the connection whose test result the member confirmed.

    Single use: the entry is gone before anything is saved, so a second
    confirm answers 404. A save that then fails is not retried; the member
    submits the form again.
    """
    entry = pending_tests.pop(answer.confirm_id or "", "confirm", scope)
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_GONE)
    try:
        slug, user_label, note = await save_confirmed(entry)
    except Exception:
        logger.exception("Could not save a confirmed %s connection", entry.connector_id)
        reason = "Cowork could not save the connection. Submit the form again."
        await record_outcome(entry, f"Not saved: {reason}")
        return {"patch": failed_patch(entry.form_id, reason)}
    await record_outcome(entry, f"Saved as `{slug}`." + (note or ""))
    return {"patch": saved_patch(entry.form_id, slug), "saved_slug": slug, "user_label": user_label}


@router.post(
    "/cancel", name="submission_cancel", status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require(SingleUserOnly))],
)
async def cancel_route(answer: ConnectorTestAnswer, scope: TenantScopeDep) -> None:
    """Drop a waiting test and the values it holds. Idempotent: an unknown id
    is not an error, and another scope's entry is left in place."""
    if (answer.approval_id is None) == (answer.confirm_id is None):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Send one of approval_id or confirm_id.")
    if answer.approval_id is not None:
        entry, text = pending_tests.pop(answer.approval_id, "approval", scope), NOT_ALLOWED
    else:
        entry, text = pending_tests.pop(answer.confirm_id, "confirm", scope), CANCELLED
    if entry is not None:
        await record_outcome(entry, text)


@router.post("/replaces", name="submission_replaces")
async def submission_replaces_route(req: SubmitFormRequest, scope: TenantScopeDep) -> dict:
    """The saved connection this submission would overwrite, so the form can ask first.

    ``{"replaces": {"name", "label"}}``, or ``{"replaces": null}`` when submitting
    would create a new connection. A named submission already targets its
    connection, so it never counts as a replacement.
    """
    connector_id = _connector_id(req)
    form_spec = req.form_spec or {}
    if (
        req.name.strip()
        or form_spec.get("_extends_connection") is not None
        or connector_id == SHAREPOINT
        or (scope.org_mode and _relayed_in_org_mode(connector_id))
    ):
        return {"replaces": None}
    credentials = {
        k: v for k, v in req.values.items()
        if k not in req.skipped and v is not None and v != ""
    }
    builtin = registry.get_connector(connector_id) is not None
    replaces = connection_a_new_save_replaces(
        connector_id, req.resolve_method(), credentials,
        vault=vault_for_scope(scope), spec_form=None if builtin else form_spec,
    )
    return {"replaces": replaces}


async def submit_form(req: SubmitFormRequest, scope: TenantScope) -> StreamingResponse:
    """Stage a submission and stream its probe; never used for a cloud datasource in org mode."""
    connector_id = _connector_id(req)

    if connector_id == SHAREPOINT:
        # Only the SharePoint routes may save it: they check the key first.
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=generic_setup_refusal(scope.org_mode))

    method = req.resolve_method()

    # A stamped form came from a stored spec and is checked against it. A
    # handcrafted form is checked against itself, so it may not reuse a known id.
    form_spec = req.form_spec or {}
    stamped = bool(req.connector_id or form_spec.get("_connector_id"))
    extends_name = form_spec.get("_extends_connection")
    # Built-in ids never touch the database; only an unknown id is looked up
    # among the scope's custom connectors.
    known = registry.get_connector(connector_id)
    is_custom = False
    if known is None:
        known = await run_db(lambda session: ConnectorCatalog(session).get_connector(connector_id), scope=scope)
        is_custom = known is not None
    if scope.org_mode and (is_custom or (known is None and req.form_spec)):
        # Nothing tests a custom connector in an organization yet; refused
        # before anything is checked or staged, so no value is kept.
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=_NO_CUSTOM_IN_ORG)
    if known is not None and stamped and extends_name is not None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="`extends_connection` applies only to handcrafted forms.",
        )
    stored_test_request = has_test_requests(known.form.model_dump()) if is_custom else None
    handcrafted = handcrafted_form_allowed(
        known=known is not None, custom=is_custom, extends=extends_name is not None,
        stored_test_request=stored_test_request,
    )
    if stamped and stored_test_request is False and not tenant_scope_required():
        # Saved before connection tests existed: it can never pass one.
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=NEEDS_REBUILD_FOR_MEMBER)
    if not stamped and not handcrafted:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{connector_id!r} is already a connector; submit the form lookup_connector returns for it.",
        )
    spec = known if stamped else None
    if spec:
        form = spec.form
        form_id = form.form_id
        methods = form.methods or []

        if methods and not method:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="`method` is required for connectors with multiple auth methods.",
            )

        if methods and method:
            method_def = next((m for m in methods if m.id == method), None)
            if not method_def:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unknown method: {method!r}",
                )

        fields = _resolve_fields(spec, method)
    else:
        # Handcrafted form: validated against itself, then probed like any
        # other connector before anything is saved.
        if not req.form_spec:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connector not found.")
        # Checked before the probe, so a form that could never be stored as a
        # connector fails here instead of after its credentials are saved.
        try:
            check_handcrafted_form(req.form_spec)
        except UnstorableFormError as e:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e
        form_id = req.form_spec.get("form_id") or req.form_id or f"{connector_id}-connector"
        fields = _fields_from_spec_dict(req.form_spec, method)
        if extends_name is not None and (
            not isinstance(extends_name, str)
            or not extends_name.strip()
            or vault_for_scope(scope).read_record(connector_id, extends_name) is None
        ):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"No {connector_id} connection named {extends_name!r} to add these fields to.",
            )
        if extends_name is not None and not ConnectionsService(scope).can_change(connector_id, extends_name):
            # A later step adds fields only to a connection the caller can
            # change (ConnectionsService.can_change), never to one someone
            # shares or the operator manages; refused before its values are
            # staged. The vault refuses the save itself too.
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=SHARED_CONNECTION_REFUSAL)

    missing = missing_required_fields(fields, req.values, req.skipped)
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required fields: {', '.join(missing)}",
        )

    submission_id = store.stage(
        form_id=form_id,
        connector_id=connector_id,
        conversation_id=req.conversation_id,
        values=req.values,
        skipped=req.skipped,
        form_spec=req.form_spec,
        custom_spec=spec.model_dump() if spec is not None and is_custom else None,
        checked_against_stored_spec=stamped,
    )

    handler = ProbeHandler(scope=scope)
    return StreamingResponse(
        handler.run(submission_id, connector_id, method, req.name, req.conversation_id),
        media_type="text/event-stream",
        # The submission stream can carry connection credentials (DSNs, keys);
        # keep it out of the client's on-disk HTTP cache. See ENG-462.
        headers={"Cache-Control": "no-store"},
    )
