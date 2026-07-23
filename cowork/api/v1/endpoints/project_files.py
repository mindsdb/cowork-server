"""Project file-browsing endpoints.

Ported from cowork/server/routes/projects.py in the old server.
This is not necessarily final state — it was migrated to eliminate
compat stubs and may be refactored later.
"""

import hashlib
import logging
import mimetypes
import os
import shutil
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from cowork.db.scoped import ScopedSession, ScopedSessionDep
from cowork.services.projects import ProjectService


logger = logging.getLogger(__name__)
router = APIRouter()

ANTON_INSTRUCTIONS_FILENAME = "anton.md"
TEXT_MAX_BYTES = 2 * 1024 * 1024  # 2 MiB

_PROJECT_PREVIEW_MOUNTS: dict[str, Path] = {}


class _FileWriteRequest(BaseModel):
    content: str


class _PreviewMountRequest(BaseModel):
    name: str
    path: str


def _project_dir(name: str, scoped: ScopedSession) -> Path:
    """Resolve a project name to its on-disk directory or 404."""
    try:
        project = ProjectService(scoped).get_project_by_name(name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    base = Path(project.path)
    if not base.is_dir():
        raise HTTPException(status_code=404, detail="Project directory not found on disk")
    return base


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
def get_project_instructions(project_name: str, scoped: ScopedSessionDep):
    base = _project_dir(project_name, scoped)
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
def list_project_files(project_name: str, scoped: ScopedSessionDep):
    base = _project_dir(project_name, scoped)
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
def read_project_file(project_name: str, path: str, scoped: ScopedSessionDep):
    base = _project_dir(project_name, scoped)
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
def write_project_file(project_name: str, path: str, req: _FileWriteRequest, scoped: ScopedSessionDep):
    base = _project_dir(project_name, scoped)
    target = _safe_relpath(path, base)
    if target.exists() and target.is_dir():
        raise HTTPException(status_code=400, detail="Path is a directory")
    body = req.content or ""
    if len(body.encode("utf-8")) > TEXT_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Content exceeds 2 MiB cap")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    st = target.stat()
    return {"path": path, "size": st.st_size, "modified": st.st_mtime}


@router.post("/{project_name}/files/upload")
async def upload_project_files(
    project_name: str,
    scoped: ScopedSessionDep,
    files: list[UploadFile] = File(...),
):
    base = _project_dir(project_name, scoped)
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
            data = await f.read()
            target.write_bytes(data)
            results.append({"name": safe_name, "ok": True, "size": len(data)})
        except Exception as exc:
            logger.error("Failed to write file %s: %s", safe_name, exc)
            results.append({"name": safe_name, "ok": False, "error": "File write failed"})
    return {"results": results}


@router.delete("/{project_name}/files/{path:path}")
def delete_project_file(project_name: str, path: str, scoped: ScopedSessionDep):
    base = _project_dir(project_name, scoped)
    target = _safe_relpath(path, base)
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if target.is_dir():
        raise HTTPException(status_code=400, detail="Path is a directory")
    target.unlink()
    return {"status": "deleted", "path": path}


@router.delete("/{project_name}/skill_drafts/{slug}", status_code=status.HTTP_204_NO_CONTENT)
def delete_skill_draft(project_name: str, slug: str, scoped: ScopedSessionDep):
    """Remove a staged skill draft once it is Saved (or dismissed).

    Idempotent: a missing draft is a no-op — Save may race the sweep, and a
    lingering draft is the safe default we're clearing, not a hard error. The
    slug is confined to a direct child of the drafts dir (no traversal).
    """
    base = _project_dir(project_name, scoped)
    drafts_root = base / ".anton" / "skill_drafts"
    target = _safe_relpath(slug, drafts_root)
    if target.parent != drafts_root.resolve():
        raise HTTPException(status_code=400, detail="invalid slug")
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)


@router.post("/preview-mount-file")
def preview_mount_file(req: _PreviewMountRequest, scoped: ScopedSessionDep):
    base = _project_dir(req.name, scoped)
    target = _safe_relpath(req.path, base)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if target.suffix.lower() != ".html":
        raise HTTPException(status_code=415, detail="Preview mount is only available for HTML files")
    parent = target.parent.resolve()
    token = hashlib.sha256(str(parent).encode("utf-8")).hexdigest()[:16]
    _PROJECT_PREVIEW_MOUNTS[token] = parent
    return {
        "token": token,
        "entry": target.name,
        "relUrl": f"/projects/preview-asset/{token}/{target.name}",
    }


@router.get("/preview-asset/{token}/{rel_path:path}")
def preview_asset(token: str, rel_path: str):
    parent = _PROJECT_PREVIEW_MOUNTS.get(token)
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
    return FileResponse(target, media_type=media_type, headers={"Cache-Control": "private, max-age=300"})


@router.get("/{project_name}/files-raw/{path:path}")
def download_project_file(project_name: str, path: str, scoped: ScopedSessionDep):
    base = _project_dir(project_name, scoped)
    target = _safe_relpath(path, base)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    return FileResponse(
        target,
        media_type=media_type,
        filename=target.name,
        headers={"Content-Disposition": f'attachment; filename="{target.name}"'},
    )
