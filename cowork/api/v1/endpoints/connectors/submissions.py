"""
DISCLAIMER: The probe for connectors will always run through Anton regardless of the harness used.
"""


from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.db.units import run_db
from cowork.handlers.probe import ProbeHandler
from cowork.schemas.connectors import ConnectorField, InvalidConnectorIdError, SubmitFormRequest
from cowork.services.connectors.catalog import ConnectorCatalog
from cowork.services.connectors.custom_connectors import UnstorableFormError, check_handcrafted_form
from cowork.services.connectors.persist import vault_for_scope
from cowork.services.connectors.specs._registry import registry
from cowork.services.connectors.submissions import store

# AuthenticatedInOrgMode: in org mode a request with no verified principal is
# refused with 401 before the form is read. The stream holds no request
# session: the probe handler's database units open their own tenant-scoped
# sessions, so no connection stays checked out while the probe waits on the
# model.
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])
TenantScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


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


def _missing_required(fields: list, values: dict, skipped: list[str]) -> list[str]:
    skipped_set = set(skipped)
    return [
        f.name for f in fields
        if f.required
        and f.name not in skipped_set
        and (values.get(f.name) is None or str(values.get(f.name, "")).strip() == "")
    ]


@router.post("/")
async def submit_form(req: SubmitFormRequest, scope: TenantScopeDep) -> StreamingResponse:
    try:
        connector_id = req.resolve_connector_id()
    except InvalidConnectorIdError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

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
    if known is not None and stamped and extends_name is not None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="`extends_connection` applies only to handcrafted forms.",
        )
    if known is not None and not stamped and not (is_custom and extends_name is not None):
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

    missing = _missing_required(fields, req.values, req.skipped)
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
