"""Artifacts API endpoints.

Ported from cowork/server/routes/artifacts.py. Provides listing,
preview, iframe mount, open-in-OS, and reveal-in-finder for
agent-produced artifacts.
"""
from __future__ import annotations

import mimetypes
import subprocess
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.services.artifacts import (
    _project_artifacts_base,
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
        "Cache-Control": "no-cache, must-revalidate",
    })


@router.get("/serve/{project_name}/{file_path:path}")
def serve_artifact_file(project_name: str, file_path: str):
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
    return FileResponse(target, media_type=media_type, headers={
        "Cache-Control": "no-cache, must-revalidate",
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


def _resolve_reveal_path(path: str, session: Session) -> Path:
    try:
        return resolve_artifact_path(path)
    except FileNotFoundError:
        pass
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    try:
        requested = Path(path).expanduser().resolve()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid path") from exc
    for project in ProjectService(session).list_projects():
        project_dir = Path(project.path).resolve()
        try:
            requested.relative_to(project_dir)
        except ValueError:
            continue
        if requested.exists():
            return requested
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Path is not in a known project or artifact directory")


@router.post("/reveal")
async def reveal_artifact(req: _PathBody, session: SessionDep):
    target = _resolve_reveal_path(req.path, session)
    try:
        reveal_in_file_manager(target)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not reveal artifact") from exc
    return {"status": "ok", "path": str(target)}


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
