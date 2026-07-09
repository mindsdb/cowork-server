"""Compat stub endpoints.  # SHIM:client-compat

These exist solely so the Cowork renderer doesn't 404 on endpoints that
haven't been migrated to cowork-server yet. Each returns a safe empty
response. Replace with real implementations as they're ported over.
"""
from __future__ import annotations

import logging
import mimetypes
import os
import shutil
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlmodel import Session

from cowork.db.session import get_session

logger = logging.getLogger(__name__)

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


def _attachment_purpose(project_name: str, session_id: str) -> str:
    # `project_name` is still part of the client-facing route but is
    # deliberately ignored: purposes are keyed by conversation id only, so a
    # project rename can't strand attachments (ENG-338).
    from cowork.services.files import attachment_purpose
    return attachment_purpose(session_id)


def _to_attachment(file) -> dict:
    """Legacy FileAttachment shape the rail's Task Uploads rows render
    (name / mime / size + ISO timestamps). Returning the OpenAI-style
    FileResponse here (filename / bytes / epoch-seconds) left the rows
    showing raw file ids and a 1970s age (ENG-264)."""
    created = file.created_at.isoformat() if file.created_at else None
    return {
        "id": str(file.id),
        "name": file.filename,
        "mime": file.content_type,
        "size": file.size,
        "created_at": created,
        # File rows are immutable once uploaded — created is updated.
        "updated_at": created,
        "purpose": file.purpose,
    }


@attachments_router.get("/{project_name}/{session_id}")
def list_attachments(
    project_name: str,
    session_id: str,
    session: _SessionDep,
    ids: list[str] | None = Query(default=None),
):
    from cowork.services.files import FileService
    rows = FileService(session).list_file_rows(purpose=_attachment_purpose(project_name, session_id))
    if ids:
        wanted = {str(i) for i in ids}
        rows = [r for r in rows if str(r.id) in wanted]
    return [_to_attachment(r) for r in rows]


@attachments_router.post("/{project_name}/{session_id}/upload")
async def upload_attachment(
    project_name: str,
    session_id: str,
    session: _SessionDep,
    files: list[UploadFile] = File(...),
):
    from cowork.services.files import FileService
    svc = FileService(session)
    purpose = _attachment_purpose(project_name, session_id)
    results = []
    for f in files:
        try:
            created = await svc.create_file(upload=f, purpose=purpose)
            results.append(_to_attachment(svc.get_file_row(UUID(created.id))))
        except Exception as exc:
            # Previously any failure here surfaced as an opaque 500 with no
            # server log (e.g. an over-long `purpose` failing model
            # validation — see the files.purpose width fix). Log the real
            # cause and return an actionable error instead of a bare crash.
            logger.exception(
                "Attachment upload failed (project=%s session=%s file=%s)",
                project_name, session_id, getattr(f, "filename", "?"),
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to store attachment '{getattr(f, 'filename', '')}'.",
            ) from exc
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


@attachments_router.get("/{project_name}/{session_id}/{attachment_id}/raw")
def attachment_raw(project_name: str, session_id: str, attachment_id: UUID, session: _SessionDep):
    from cowork.services.files import FileService
    try:
        content_type, filename, path = FileService(session).get_file_content(attachment_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    # Inline, not attachment: the rail's row click opens this URL in a
    # browser tab expecting the file to RENDER (image/pdf/text). The
    # default attachment disposition silently downloads instead, which
    # reads as "open does nothing" (ENG-264).
    return FileResponse(
        path,
        media_type=content_type,
        filename=filename,
        content_disposition_type="inline",
    )


def _unique_project_target(project_dir: Path, filename: str) -> Path:
    safe_name = os.path.basename(filename or "upload").strip() or "upload"
    target = project_dir / safe_name
    if not target.exists():
        return target
    stem = target.stem or "upload"
    suffix = target.suffix
    for i in range(2, 10_000):
        candidate = project_dir / f"{stem}-{i}{suffix}"
        if not candidate.exists():
            return candidate
    raise HTTPException(status_code=409, detail="Could not choose a unique project filename")


@attachments_router.post("/{project_name}/{session_id}/{attachment_id}/move-to-project")
def move_attachment_to_project(project_name: str, session_id: str, attachment_id: UUID, session: _SessionDep):
    from cowork.services.files import FileService
    from cowork.services.projects import ProjectService

    try:
        project = ProjectService(session).get_project_by_name(project_name)
        content_type, filename, source = FileService(session).get_file_content(attachment_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    project_dir = Path(project.path)
    project_dir.mkdir(parents=True, exist_ok=True)
    target = _unique_project_target(project_dir, filename)
    shutil.copy2(source, target)
    FileService(session).delete_file(attachment_id)
    return {
        "ok": True,
        "project_path": target.name,
        "absolute_path": str(target),
        "content_type": content_type or mimetypes.guess_type(str(target))[0] or "application/octet-stream",
    }


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

