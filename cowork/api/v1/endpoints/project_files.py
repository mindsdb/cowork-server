"""Project file-browsing endpoints.

Ported from cowork/server/routes/projects.py in the old server.
This is not necessarily final state — it was migrated to eliminate
compat stubs and may be refactored later.
"""

import logging
import mimetypes
import os
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.models.project import Project
from cowork.services.artifacts import (
    get_preview_mount,
    register_preview_mount,
    sign_serve_url_path,
    verify_serve_url_token,
)
from cowork.services.artifact_versions import snapshot_artifact as _snapshot_artifact
from cowork.services.project_permissions import ProjectPermissionError, require_project_permission_if_owned
from cowork.services.projects import ProjectService
from cowork.services.request_identity import (
    AuthenticationError,
    RequestPrincipal,
    principal_from_authorization_header,
)


logger = logging.getLogger(__name__)
router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]

ANTON_INSTRUCTIONS_FILENAME = "anton.md"
TEXT_MAX_BYTES = 2 * 1024 * 1024  # 2 MiB

PROJECT_FILE_DOWNLOAD_TOKEN_TTL_SECONDS = 15 * 60


def get_request_principal(request: Request) -> RequestPrincipal | None:
    try:
        return principal_from_authorization_header(request.headers.get("authorization"))
    except AuthenticationError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


PrincipalDep = Annotated[RequestPrincipal | None, Depends(get_request_principal)]


class _FileWriteRequest(BaseModel):
    content: str


class _PreviewMountRequest(BaseModel):
    name: str
    path: str


def _project_dir(name: str, session: Session) -> Path:
    """Resolve a project name to its on-disk directory or 404."""
    project = _project_record(name, session)
    return _project_base(project)


def _project_record(name: str, session: Session) -> Project:
    try:
        return ProjectService(session).get_project_by_name(name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _project_base(project: Project) -> Path:
    base = Path(project.path)
    if not base.is_dir():
        raise HTTPException(status_code=404, detail="Project directory not found on disk")
    return base


def _require_project_capability_if_owned(
    session: Session,
    project: Project,
    principal: RequestPrincipal | None,
    capability: str,
) -> None:
    try:
        require_project_permission_if_owned(
            session,
            project.id,
            principal.email if principal is not None else None,
            capability,
        )
    except ProjectPermissionError as exc:
        if principal is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication is required for this shared project",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this project file",
        ) from exc


def _browser_file_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    return {
        "Cache-Control": "no-store",
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
        **(extra or {}),
    }


def _raw_file_url(project_name: str, path: str, target: Path) -> str:
    safe_project = quote(project_name)
    safe_path = "/".join(quote(part) for part in path.replace("\\", "/").split("/") if part)
    token = quote(sign_serve_url_path(target, ttl_seconds=PROJECT_FILE_DOWNLOAD_TOKEN_TTL_SECONDS))
    return f"/api/v1/projects/{safe_project}/files-raw/{safe_path}?token={token}"


def _anton_md_path(base: Path) -> Path:
    return base / ".anton" / ANTON_INSTRUCTIONS_FILENAME


def _safe_relpath(rel: str, base: Path) -> Path:
    if not rel:
        raise HTTPException(status_code=400, detail="path required")
    cleaned = rel.replace("\\", "/").lstrip("/")
    candidate = (base / cleaned).resolve()
    base_resolved = base.resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid path") from exc
    return candidate


def _artifact_folder_for_project_file(target: Path, base: Path) -> Path | None:
    artifacts_root = (base / ".anton" / "artifacts").resolve(strict=False)
    try:
        rel = target.resolve(strict=False).relative_to(artifacts_root)
    except ValueError:
        return None
    if not rel.parts:
        return None
    folder = artifacts_root / rel.parts[0]
    return folder if (folder / "metadata.json").is_file() else None


def _checkpoint_artifact_file_edit(
    session: Session,
    target: Path,
    base: Path,
    *,
    operation_type: str,
    label: str,
    actor_name: str | None = None,
    actor_email: str | None = None,
    actor_subject: str | None = None,
) -> None:
    folder = _artifact_folder_for_project_file(target, base)
    if folder is None:
        return
    try:
        _snapshot_artifact(
            session,
            str(folder),
            operation_type=operation_type,
            label=label,
            actor_name=actor_name,
            actor_email=actor_email,
            actor_subject=actor_subject,
        )
    except Exception as exc:
        logger.warning("Could not checkpoint artifact file edit for %s", target, exc_info=True)
        raise HTTPException(status_code=500, detail="Could not checkpoint artifact before changing this artifact file") from exc


def _file_meta(p: Path, base: Path) -> dict[str, Any] | None:
    try:
        st = p.stat()
    except FileNotFoundError:
        return None
    try:
        resolved = p.resolve()
        rel = resolved.relative_to(base.resolve())
    except ValueError:
        return None
    return {
        "path": str(rel).replace("\\", "/"),
        "name": p.name,
        "size": st.st_size,
        "modified": st.st_mtime,
        "is_dir": p.is_dir(),
    }


@router.get("/{project_name}/instructions")
def get_project_instructions(project_name: str, session: SessionDep, principal: PrincipalDep):
    project = _project_record(project_name, session)
    _require_project_capability_if_owned(session, project, principal, "view")
    base = _project_base(project)
    p = _anton_md_path(base)
    rel = p.relative_to(base).as_posix()
    if p.is_file():
        try:
            st = p.stat()
        except OSError:
            return {"file": {"path": rel, "name": ANTON_INSTRUCTIONS_FILENAME, "size": 0, "modified": None, "is_dir": False, "synthetic": True}}
        return {"file": {"path": rel, "name": ANTON_INSTRUCTIONS_FILENAME, "size": st.st_size, "modified": st.st_mtime, "is_dir": False}}
    return {"file": {"path": rel, "name": ANTON_INSTRUCTIONS_FILENAME, "size": 0, "modified": None, "is_dir": False, "synthetic": True}}


@router.get("/{project_name}/files")
def list_project_files(project_name: str, session: SessionDep, principal: PrincipalDep):
    project = _project_record(project_name, session)
    _require_project_capability_if_owned(session, project, principal, "view")
    base = _project_base(project)
    files: list[dict[str, Any]] = []
    for p in sorted(base.rglob("*")):
        if p.is_dir():
            continue
        meta = _file_meta(p, base)
        if meta:
            files.append(meta)

    anton_rel = _anton_md_path(base).relative_to(base).as_posix()
    if not any(f["path"] == anton_rel for f in files):
        files.insert(0, {
            "path": anton_rel,
            "name": ANTON_INSTRUCTIONS_FILENAME,
            "size": 0,
            "modified": None,
            "is_dir": False,
            "synthetic": True,
        })
    else:
        files.sort(key=lambda f: (f["path"] != anton_rel, f["path"]))

    return {"files": files}


@router.get("/{project_name}/files/{path:path}")
def read_project_file(project_name: str, path: str, session: SessionDep, principal: PrincipalDep):
    project = _project_record(project_name, session)
    _require_project_capability_if_owned(session, project, principal, "view")
    base = _project_base(project)
    target = _safe_relpath(path, base)
    if not target.exists():
        anton_rel = _anton_md_path(base).relative_to(base).as_posix()
        if path == anton_rel:
            return {"path": path, "content": "", "size": 0, "modified": None}
        raise HTTPException(status_code=404, detail="File not found")
    if target.is_dir():
        raise HTTPException(status_code=400, detail="Path is a directory")
    if target.stat().st_size > TEXT_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large to read inline")
    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=415, detail="File is not valid UTF-8 text") from exc
    st = target.stat()
    return {"path": path, "content": content, "size": st.st_size, "modified": st.st_mtime}


@router.put("/{project_name}/files/{path:path}")
def write_project_file(project_name: str, path: str, req: _FileWriteRequest, session: SessionDep, principal: PrincipalDep):
    project = _project_record(project_name, session)
    _require_project_capability_if_owned(session, project, principal, "edit")
    base = _project_base(project)
    target = _safe_relpath(path, base)
    if target.exists() and target.is_dir():
        raise HTTPException(status_code=400, detail="Path is a directory")
    body = req.content or ""
    if len(body.encode("utf-8")) > TEXT_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Content exceeds 2 MiB cap")
    target.parent.mkdir(parents=True, exist_ok=True)
    existed_before = target.exists()
    previous_bytes = target.read_bytes() if existed_before else None
    _checkpoint_artifact_file_edit(
        session,
        target,
        base,
        operation_type="pre_edit",
        label="Before file edit",
        actor_name=principal.name if principal is not None else None,
        actor_email=principal.email if principal is not None else None,
        actor_subject=principal.subject if principal is not None else None,
    )
    target.write_text(body, encoding="utf-8")
    try:
        _checkpoint_artifact_file_edit(
            session,
            target,
            base,
            operation_type="edit",
            label="File edit",
            actor_name=principal.name if principal is not None else None,
            actor_email=principal.email if principal is not None else None,
            actor_subject=principal.subject if principal is not None else None,
        )
    except HTTPException:
        if existed_before and previous_bytes is not None:
            target.write_bytes(previous_bytes)
        else:
            target.unlink(missing_ok=True)
        raise
    st = target.stat()
    return {"path": path, "size": st.st_size, "modified": st.st_mtime}


@router.post("/{project_name}/files/upload")
async def upload_project_files(
    request: Request,
    project_name: str,
    session: SessionDep,
    principal: PrincipalDep,
    files: list[UploadFile] = File(...),
):
    from cowork.services.files import (
        FileValidationError,
        UploadTooLarge,
        reject_if_content_length_over_cap,
        stream_upload_to_path,
    )

    project = _project_record(project_name, session)
    _require_project_capability_if_owned(session, project, principal, "edit")
    base = _project_base(project)
    # Up-front reject for an obviously oversized body before streaming; the
    # per-file cap below is the authoritative check.
    try:
        reject_if_content_length_over_cap(request.headers.get("content-length"))
    except UploadTooLarge as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(exc)
        ) from exc
    results: list[dict[str, Any]] = []
    for f in files:
        if not f.filename:
            results.append({"name": "", "ok": False, "error": "filename missing"})
            continue
        safe_name = os.path.basename(f.filename).strip()
        if not safe_name or safe_name.startswith("."):
            results.append({"name": f.filename, "ok": False, "error": "invalid filename"})
            continue
        target = base / safe_name
        try:
            # Stream chunked to disk so a giant file can't OOM the server,
            # enforcing the size cap mid-stream. No type allow-list here: this
            # is the project working directory, where arbitrary source/config/
            # archive files are expected (unlike the chat-attachment path).
            size = await stream_upload_to_path(f, target)
            results.append({"name": safe_name, "ok": True, "size": size})
        except FileValidationError as exc:
            # Over the size cap: report per-file (the batch contract is
            # per-file ok/error) with the user-facing reason. The partial file
            # is cleaned up by stream_upload_to_path on breach.
            results.append({"name": safe_name, "ok": False, "error": str(exc)})
        except Exception as exc:
            logger.error("Failed to write file %s: %s", safe_name, exc)
            results.append({"name": safe_name, "ok": False, "error": "File write failed"})
    return {"results": results}


@router.delete("/{project_name}/files/{path:path}")
def delete_project_file(project_name: str, path: str, session: SessionDep, principal: PrincipalDep):
    project = _project_record(project_name, session)
    _require_project_capability_if_owned(session, project, principal, "edit")
    base = _project_base(project)
    target = _safe_relpath(path, base)
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if target.is_dir():
        raise HTTPException(status_code=400, detail="Path is a directory")
    previous_bytes = target.read_bytes()
    _checkpoint_artifact_file_edit(
        session,
        target,
        base,
        operation_type="pre_delete_file",
        label="Before file delete",
        actor_name=principal.name if principal is not None else None,
        actor_email=principal.email if principal is not None else None,
        actor_subject=principal.subject if principal is not None else None,
    )
    target.unlink()
    try:
        _checkpoint_artifact_file_edit(
            session,
            target,
            base,
            operation_type="delete_file",
            label="File delete",
            actor_name=principal.name if principal is not None else None,
            actor_email=principal.email if principal is not None else None,
            actor_subject=principal.subject if principal is not None else None,
        )
    except HTTPException:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(previous_bytes)
        raise
    return {"status": "deleted", "path": path}


@router.post("/preview-mount-file")
def preview_mount_file(req: _PreviewMountRequest, session: SessionDep, principal: PrincipalDep):
    project = _project_record(req.name, session)
    _require_project_capability_if_owned(session, project, principal, "view")
    base = _project_base(project)
    target = _safe_relpath(req.path, base)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if target.suffix.lower() != ".html":
        raise HTTPException(status_code=415, detail="Preview mount is only available for HTML files")
    parent = target.parent.resolve()
    token = register_preview_mount(parent)
    return {
        "token": token,
        "entry": target.name,
        "relUrl": f"/projects/preview-asset/{token}/{target.name}",
    }


@router.get("/preview-asset/{token}/{rel_path:path}")
def preview_asset(token: str, rel_path: str):
    parent = get_preview_mount(token)
    if parent is None:
        raise HTTPException(status_code=404, detail="Preview mount has expired or is unknown")
    try:
        target = (parent / rel_path).resolve()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid asset path") from exc
    try:
        target.relative_to(parent)
    except ValueError:
        raise HTTPException(status_code=403, detail="Asset is outside the mounted directory")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    return FileResponse(target, media_type=media_type, headers=_browser_file_headers())


class _RawFileTokenRequest(BaseModel):
    path: str


@router.post("/{project_name}/files-raw-token")
def create_project_file_download_token(
    project_name: str,
    req: _RawFileTokenRequest,
    session: SessionDep,
    principal: PrincipalDep,
):
    project = _project_record(project_name, session)
    _require_project_capability_if_owned(session, project, principal, "view")
    base = _project_base(project)
    target = _safe_relpath(req.path, base)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return {
        "url": _raw_file_url(project_name, req.path, target),
        "expiresIn": PROJECT_FILE_DOWNLOAD_TOKEN_TTL_SECONDS,
    }


@router.get("/{project_name}/files-raw/{path:path}")
def download_project_file(
    project_name: str,
    path: str,
    request: Request,
    session: SessionDep,
    token: str | None = Query(default=None),
):
    project = _project_record(project_name, session)
    base = _project_base(project)
    target = _safe_relpath(path, base)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if not verify_serve_url_token(target, token):
        principal = get_request_principal(request)
        _require_project_capability_if_owned(session, project, principal, "view")
    media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    return FileResponse(
        target,
        media_type=media_type,
        filename=target.name,
        headers=_browser_file_headers({"Content-Disposition": f'attachment; filename="{target.name}"'}),
    )
