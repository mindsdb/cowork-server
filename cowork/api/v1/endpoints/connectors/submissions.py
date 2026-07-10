"""
DISCLAIMER: The probe for connectors will always run through Anton regardless of the harness used.
"""


from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.handlers.probe import ProbeHandler
from cowork.schemas.connectors import ConnectorField, SubmitFormRequest
from cowork.services.connectors.specs._registry import registry
from cowork.services.connectors.submissions import store

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]


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
    # Agent-handcrafted specs are loose JSON — tolerate missing label/type
    # rather than rejecting a save the form UI already accepted.
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
async def submit_form(req: SubmitFormRequest, session: SessionDep) -> StreamingResponse:
    try:
        connector_id = req.resolve_connector_id()
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    method = req.resolve_method()

    spec = registry.get_connector(connector_id)
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
        # Non-registry connector: the agent handcrafted this form. Validate
        # against the submitted form_spec; the probe handler saves it to the
        # vault without a live probe.
        if not req.form_spec:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connector not found.")
        form_id = req.form_spec.get("form_id") or req.form_id or f"{connector_id}-connector"
        fields = _fields_from_spec_dict(req.form_spec, method)

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
    )

    handler = ProbeHandler(session)
    return StreamingResponse(
        handler.run(submission_id, connector_id, method, req.name, req.conversation_id),
        media_type="text/event-stream",
        # The submission stream can carry connection credentials (DSNs, keys);
        # keep it out of the client's on-disk HTTP cache. See ENG-462.
        headers={"Cache-Control": "no-store"},
    )
