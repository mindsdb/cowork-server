from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.handlers.probe import ProbeHandler
from cowork.schemas.connectors import SubmitFormRequest
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
    spec = registry.get_connector(req.connector_id)
    if not spec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connector not found.")

    form = spec.form
    methods = form.methods or []

    if methods and not req.method:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="`method` is required for connectors with multiple auth methods.",
        )

    if methods and req.method:
        method_def = next((m for m in methods if m.id == req.method), None)
        if not method_def:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown method: {req.method!r}",
            )

    fields = _resolve_fields(spec, req.method)
    missing = _missing_required(fields, req.values, req.skipped)
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required fields: {', '.join(missing)}",
        )

    submission_id = store.stage(
        form_id=form.form_id,
        connector_id=req.connector_id,
        conversation_id=req.conversation_id,
        values=req.values,
        skipped=req.skipped,
    )

    handler = ProbeHandler(session)
    return StreamingResponse(
        handler.run(submission_id, req.connector_id, req.method, req.name, req.conversation_id),
        media_type="text/event-stream",
    )
