"""Project file-browsing endpoints.

Ported from cowork/server/routes/projects.py in the old server.
This is not necessarily final state — it was migrated to eliminate
compat stubs and may be refactored later.
"""

import logging
import mimetypes
import os
import secrets
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from cowork.db.scoped import ScopedSession, ScopedSessionDep, TenantScope, get_tenant_scope
from cowork.services.projects import ProjectService


logger = logging.getLogger(__name__)
router = APIRouter()

ANTON_INSTRUCTIONS_FILENAME = "anton.md"
TEXT_MAX_BYTES = 2 * 1024 * 1024  # 2 MiB

#: How long a preview token stays usable. Long enough that a preview left open
#: keeps loading its sub-assets, short enough that a token which escapes the
#: browser it was minted in dies the same session.
PREVIEW_TOKEN_TTL_SECONDS = 30 * 60


@dataclass(frozen=True)
class _PreviewMount:
    """A directory one preview token may read, and who may read it.

    The token is a bearer string in a URL, so an iframe can load it and a
    screenshot, a proxy log or a browser history can leak it. The record is
    therefore what carries the authority, not the string: whoever presents the
    token still has to be the member it was minted for, in the organization it
    was minted in, before it expires.
    """

    parent: Path
    org_id: str | None
    user_id: str | None
    expires_at: float

    def readable_by(self, scope: TenantScope) -> bool:
        """Desktop has one user and no organization, so nothing to compare."""
        if not scope.org_mode:
            return True
        return self.org_id == scope.org_id and self.user_id == scope.user_id


_PROJECT_PREVIEW_MOUNTS: dict[str, _PreviewMount] = {}


def _register_preview_mount(parent: Path, scope: TenantScope) -> str:
    """Mint a token for `parent` and bind it to the caller.

    Expired records are dropped here rather than on read: minting is rare and
    reading is per sub-asset, and nothing else ever removes an entry.
    """
    now = time.time()
    for stale in [t for t, m in _PROJECT_PREVIEW_MOUNTS.items() if m.expires_at <= now]:
        _PROJECT_PREVIEW_MOUNTS.pop(stale, None)
    token = secrets.token_urlsafe(32)
    _PROJECT_PREVIEW_MOUNTS[token] = _PreviewMount(
        parent=parent,
        org_id=scope.org_id,
        user_id=scope.user_id,
        expires_at=now + PREVIEW_TOKEN_TTL_SECONDS,
    )
    return token


class _FileWriteRequest(BaseModel):
    content: str


class _PreviewMountRequest(BaseModel):
    name: str
    path: str


def _project_dir(name: str, scoped: ScopedSession) -> Path:
    """Resolve a project name to its on-disk directory or 404."""
    service = ProjectService(scoped)
    try:
        project = service.get_project_by_name(name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    # Any project can lose its directory (fresh pod, wiped volume), not just the
    # default one. The row is authoritative, so provision rather than 404 the
    # owner out of their own project.
    service.ensure_dir_exists(project)
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


def _conversation_workspace_ok(
    rel_posix: str, scoped: ScopedSession, cache: dict | None = None
) -> bool:
    """`conversations/<id>/…` is a per-conversation PRIVATE workspace even
    though the project directory is org-shared, so only that conversation's
    owner may list/read/write/delete inside it. Anything else is a shared
    project file. Local mode has one user, so nothing is gated."""
    if not scoped.scope.org_mode:
        return True
    parts = rel_posix.split("/", 2)
    if len(parts) < 2 or parts[0] != "conversations":
        return True  # shared project file
    seg = parts[1]
    if cache is not None and seg in cache:
        return cache[seg]
    try:
        conv_id = UUID(seg)
    except ValueError:
        ok = True  # not a real conversation dir → treat as shared
    else:
        from cowork.services.conversations import ConversationService

        ok = ConversationService(scoped)._owned(conv_id) is not None
    if cache is not None:
        cache[seg] = ok
    return ok


def _require_workspace_access(target: Path, base: Path, scoped: ScopedSession) -> None:
    """404 (no existence oracle) if `target` is inside another member's
    conversation workspace.

    `target` came from `_safe_relpath`, which already confirmed it sits under
    `base`, but re-derive the relative path defensively: a symlink that makes
    the resolved path escape `base` must 404, never raise (CodeQL: user data in
    a path expression)."""
    try:
        rel = target.relative_to(base.resolve()).as_posix()
    except (ValueError, OSError):
        raise HTTPException(status_code=404, detail="File not found")
    if not _conversation_workspace_ok(rel, scoped):
        raise HTTPException(status_code=404, detail="File not found")


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
    _conv_cache: dict = {}
    for p in sorted(base.rglob("*")):
        if p.is_dir():
            continue
        meta = _file_meta(p, base)
        if meta and _conversation_workspace_ok(meta["path"], scoped, _conv_cache):
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
    _require_workspace_access(target, base, scoped)
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
    _require_workspace_access(target, base, scoped)
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
    _require_workspace_access(target, base, scoped)
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
    # Resolve, then require the target to stay inside the drafts dir — rejects any
    # traversal in `slug` regardless of what it contains.
    drafts_root = os.path.realpath(_project_dir(project_name, scoped) / ".anton" / "skill_drafts")
    folder = os.path.realpath(os.path.join(drafts_root, slug))
    if not folder.startswith(drafts_root + os.sep):
        raise HTTPException(status_code=400, detail="invalid slug")
    if os.path.isdir(folder):
        shutil.rmtree(folder, ignore_errors=True)


@router.post("/preview-mount-file")
def preview_mount_file(req: _PreviewMountRequest, scoped: ScopedSessionDep):
    base = _project_dir(req.name, scoped)
    target = _safe_relpath(req.path, base)
    _require_workspace_access(target, base, scoped)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if target.suffix.lower() != ".html":
        raise HTTPException(status_code=415, detail="Preview mount is only available for HTML files")
    parent = target.parent.resolve()
    token = _register_preview_mount(parent, scoped.scope)
    return {
        "token": token,
        "entry": target.name,
        "relUrl": f"/projects/preview-asset/{token}/{target.name}",
    }


@router.get("/preview-asset/{token}/{rel_path:path}")
def preview_asset(
    token: str, rel_path: str, scope: TenantScope = Depends(get_tenant_scope)
):
    # An unknown token, an expired one and another member's one all answer the
    # same 404: which of the three it was is itself information about somebody
    # else's files.
    mount = _PROJECT_PREVIEW_MOUNTS.get(token)
    if mount is None or mount.expires_at <= time.time() or not mount.readable_by(scope):
        raise HTTPException(status_code=404, detail="Preview mount has expired or is unknown")
    parent = mount.parent
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
    _require_workspace_access(target, base, scoped)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    return FileResponse(
        target,
        media_type=media_type,
        filename=target.name,
        headers={"Content-Disposition": f'attachment; filename="{target.name}"'},
    )
