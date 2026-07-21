"""Artifacts API endpoints.

Ported from cowork/server/routes/artifacts.py. Provides listing,
preview, iframe mount, open-in-OS, and reveal-in-finder for
agent-produced artifacts.
"""
from __future__ import annotations

import mimetypes
import os
import subprocess
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from sqlmodel import Session

from cowork.db.scoped import ScopedSession, ScopedSessionDep
from cowork.db.session import get_session
from cowork.services.comments_layer import ACTIVATION_PARAM, inject_layer
from cowork.services.artifacts import (
    _project_artifacts_base,
    artifact_status as _artifact_status,
    delete_artifact as _delete_artifact,
    get_preview_mount,
    list_artifacts as _list_artifacts,
    mount_preview,
    preview_artifact as _preview_artifact,
    resolve_artifact_path,
    reveal_in_file_manager,
)
from cowork.services.projects import ProjectService

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]


class _PathBody(BaseModel):
    path: str


# ``no-cache`` mirrors the FileResponse headers used elsewhere so a rebuilt
# artifact is always re-fetched.
_NO_CACHE = {"Cache-Control": "no-cache, must-revalidate"}


def _wants_comment_layer(media_type: str, request: Request) -> bool:
    """The comment marker layer is injected only into the top-level HTML
    document, and only when the renderer opts in via the activation query flag.
    Asset requests and flag-less loads stream untouched."""
    return media_type == "text/html" and ACTIVATION_PARAM in request.query_params


def _html_with_layer(target: Path):
    """Read an HTML file and return an HTMLResponse with the marker layer
    injected, or None when it can't be read as text (caller falls back to a
    plain FileResponse)."""
    try:
        html = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return HTMLResponse(inject_layer(html), headers=_NO_CACHE)


@router.get("/")
async def list_artifacts(project_path: str | None = Query(default=None)):
    return _list_artifacts(project_path)


@router.get("/status")
async def artifact_status(path: str = Query(...)):
    # Cheap published/modified/access read for the preview viewer's in-place
    # refresh. Never raises for an unknown path — returns the blank default.
    return _artifact_status(path)


@router.get("/preview")
async def preview_artifact(path: str = Query(...)):
    try:
        artifact = resolve_artifact_path(path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    try:
        return _preview_artifact(artifact)
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not read artifact") from e


class _ExportBody(BaseModel):
    path: str
    format: str  # 'pdf' | 'docx' | 'html'


@router.post("/export")
async def export_artifact_endpoint(req: _ExportBody):
    """Convert a document artifact (markdown/HTML) to PDF/Word/HTML, writing
    the result into the same artifact folder. Returns the new file's path so
    the client can open or download it."""
    from fastapi.concurrency import run_in_threadpool

    from cowork.services.artifact_export import ExportError, export_artifact

    try:
        source = resolve_artifact_path(req.path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    try:
        out = await run_in_threadpool(export_artifact, source, req.format)
    except ExportError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Export failed") from e
    return {"path": str(out), "filename": out.name}


@router.post("/preview-mount")
async def preview_mount_endpoint(req: _PathBody, request: Request):
    try:
        artifact = resolve_artifact_path(req.path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    try:
        payload = await mount_preview(artifact)
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e))

    if payload.get("kind") == "proxy":
        # Build the absolute proxy URL from the incoming request. Using
        # scheme+netloc means the iframe loads through the same host
        # the client used to reach us — works equally for desktop
        # (127.0.0.1:port) and cloud (reverse-proxy origin).
        token = payload["token"]
        payload["proxyUrl"] = (
            f"{request.url.scheme}://{request.url.netloc}"
            f"/api/v1/artifacts/proxy/{token}/"
        )
    return payload


@router.get("/preview-asset/{token}/{rel_path:path}")
async def preview_asset(token: str, rel_path: str, request: Request):
    parent = get_preview_mount(token)
    if parent is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preview mount has expired or is unknown")
    try:
        target = (parent / rel_path).resolve()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid asset path") from exc
    try:
        target.relative_to(parent)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Asset is outside the artifact directory")
    if not target.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")
    media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    if _wants_comment_layer(media_type, request):
        # Offload the (potentially large) synchronous read so it doesn't stall
        # the event loop / other in-flight SSE streams — this endpoint is async.
        resp = await run_in_threadpool(_html_with_layer, target)
        if resp is not None:
            return resp
    return FileResponse(target, media_type=media_type, headers=_NO_CACHE)


@router.get("/serve/{project_name}/{file_path:path}")
def serve_artifact_file(project_name: str, file_path: str, request: Request):
    """Serve a file from `<project>/.anton/artifacts/<file_path>` over
    HTTP. Stateless, origin-relative, frame-able so the in-app iframe
    and new-tab open both work in web deployments."""
    base = _project_artifacts_base(project_name)
    if base is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown project")
    try:
        target = (base / file_path).resolve()
        target.relative_to(base.resolve())
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid artifact path") from exc
    if not target.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact file not found")
    media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    # This endpoint is a sync `def`, so FastAPI already runs it in a threadpool
    # — the blocking read here doesn't touch the event loop.
    if _wants_comment_layer(media_type, request):
        resp = _html_with_layer(target)
        if resp is not None:
            return resp
    return FileResponse(target, media_type=media_type, headers=_NO_CACHE)


@router.post("/open")
async def open_artifact(req: _PathBody):
    try:
        artifact = resolve_artifact_path(req.path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    try:
        subprocess.run(["open", str(artifact)], check=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not open artifact") from exc
    return {"status": "ok", "path": str(artifact)}


def _resolve_reveal_path(path: str, session: ScopedSession) -> Path:
    try:
        return resolve_artifact_path(path)
    except FileNotFoundError:
        pass
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    try:
        requested = os.path.realpath(Path(path).expanduser())
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid path") from exc
    for project in ProjectService(session).list_projects():
        project_dir = os.path.realpath(project.path)
        # Trailing separator so `<dir>` doesn't match sibling `<dir>-other`.
        if requested == project_dir or requested.startswith(project_dir + os.sep):
            resolved = Path(requested)
            if resolved.exists():
                return resolved
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Path is not in a known project or artifact directory")


@router.post("/reveal")
async def reveal_artifact(req: _PathBody, session: ScopedSessionDep):
    target = _resolve_reveal_path(req.path, session)
    try:
        reveal_in_file_manager(target)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not reveal artifact") from exc
    return {"status": "ok", "path": str(target)}


@router.api_route(
    "/proxy/{token}/{rel_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy(token: str, rel_path: str, request: Request):
    """HTTP forwarder for fullstack-artifact previews.

    Streams the request to the artifact's backend running on
    `127.0.0.1:<metadata.json port>`, injects CORS, strips hop-by-hop
    headers. See `cowork.services.preview_proxy` for the body.
    """
    from cowork.services.preview_proxy import proxy_artifact_request
    return await proxy_artifact_request(token, rel_path, request)


@router.delete("/", status_code=status.HTTP_204_NO_CONTENT)
def delete_artifact_endpoint(path: str = Query(...)):
    try:
        _delete_artifact(path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not delete artifact") from e
