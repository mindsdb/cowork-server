"""Artifacts API endpoints.

Ported from cowork/server/routes/artifacts.py. Provides listing,
preview, iframe mount, open-in-OS, and reveal-in-finder for
agent-produced artifacts.
"""
from __future__ import annotations

import mimetypes
import subprocess

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from cowork.services.artifacts import (
    get_preview_mount,
    list_artifacts as _list_artifacts,
    mount_preview,
    preview_artifact as _preview_artifact,
    resolve_artifact_path,
    reveal_in_file_manager,
)

router = APIRouter()


class _PathBody(BaseModel):
    path: str


@router.get("/")
async def list_artifacts(project_path: str | None = Query(default=None)):
    return _list_artifacts(project_path)


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


@router.post("/preview-mount")
async def preview_mount_endpoint(req: _PathBody):
    try:
        artifact = resolve_artifact_path(req.path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    try:
        return mount_preview(artifact)
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e))


@router.get("/preview-asset/{token}/{rel_path:path}")
async def preview_asset(token: str, rel_path: str):
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
    return FileResponse(target, media_type=media_type, headers={
        "Cache-Control": "private, max-age=300",
    })


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


@router.post("/reveal")
async def reveal_artifact(req: _PathBody):
    try:
        artifact = resolve_artifact_path(req.path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    try:
        reveal_in_file_manager(artifact)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not reveal artifact") from exc
    return {"status": "ok", "path": str(artifact)}
