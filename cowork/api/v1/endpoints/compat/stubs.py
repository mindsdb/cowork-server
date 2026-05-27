"""Compat stub endpoints.  # SHIM:client-compat

These exist solely so the Cowork renderer doesn't 404 on endpoints that
haven't been migrated to cowork-server yet. Each returns a safe empty
response. Replace with real implementations as they're ported over.
"""
from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlmodel import Session

from cowork.db.session import get_session

# ── Integrations ─────────────────────────────────────────────────────

integrations_router = APIRouter()


@integrations_router.get("")
def list_integrations():
    return []


@integrations_router.post("/{service}/oauth/start")
def oauth_start(service: str, body: dict[str, Any] | None = None):
    return {"url": None, "error": "OAuth not yet available in cowork-server"}


# ── Attachments ──────────────────────────────────────────────────────
# TODO(migration): Per MIGRATION.md, attachments should be removed.
# The client should upload via POST /v1/files/ and reference files as
# input_file content blocks in the Responses request input field.
# These endpoints exist as a compat bridge for the current client.

attachments_router = APIRouter()
_SessionDep = Annotated[Session, Depends(get_session)]


@attachments_router.get("/{project_name}/{session_id}")
def list_attachments(project_name: str, session_id: str, session: _SessionDep):
    from cowork.services.files import FileService
    return FileService(session).list_files(purpose="attachment")


@attachments_router.post("/{project_name}/{session_id}/upload")
async def upload_attachment(
    project_name: str,
    session_id: str,
    session: _SessionDep,
    files: list[UploadFile] = File(...),
):
    from cowork.services.files import FileService
    svc = FileService(session)
    results = []
    for f in files:
        result = await svc.create_file(upload=f, purpose="attachment")
        results.append(result)
    return results


@attachments_router.delete("/{attachment_id}")
def delete_attachment(attachment_id: UUID, session: _SessionDep):
    from cowork.services.files import FileService
    FileService(session).delete_file(attachment_id)
    return {"ok": True}


@attachments_router.delete("/{project_name}/{session_id}/{attachment_id}")
def delete_attachment_scoped(project_name: str, session_id: str, attachment_id: UUID, session: _SessionDep):
    from cowork.services.files import FileService
    FileService(session).delete_file(attachment_id)
    return {"ok": True}


# ── Scratchpad ───────────────────────────────────────────────────────

scratchpad_router = APIRouter()


@scratchpad_router.post("/cancel")
def cancel_scratchpad():
    return {"ok": True}


# ── Browse ───────────────────────────────────────────────────────────

browse_router = APIRouter()


@browse_router.get("/status")
def browse_status():
    return {"available": False}


