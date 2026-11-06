"""Read-only recipient routes. Every request rechecks the operator's grant.

The grant registry and the project's files sit under the shared storage root,
on the shared file store where hosted, so each route reads them as one unit on
the file store's threads (run_efs_unit).
"""
from contextlib import contextmanager
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response

from cowork.api.v1.content_disposition import attachment_disposition
from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.db.units import run_efs_unit
from cowork.schemas.project_files import ProjectFileListResponse, ProjectFileReadResponse
from cowork.services.shared_projects import DOWNLOAD_MAX_BYTES, TEXT_MAX_BYTES, UNREADABLE_ERRNOS, SharedProjectFiles


async def _no_cache(response: Response):
    # A coroutine: FastAPI runs a sync dependency on anyio's default threads.
    response.headers["Cache-Control"] = "no-store"


router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode)), Depends(_no_cache)])
ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


@contextmanager
def _file_errors():
    try:
        yield
    except OverflowError as error:
        raise HTTPException(413, "File too large", headers={"Cache-Control": "no-store"}) from error
    except UnicodeDecodeError as error:
        raise HTTPException(415, "File is not valid UTF-8 text", headers={"Cache-Control": "no-store"}) from error
    except OSError as error:
        if isinstance(error, FileNotFoundError) or error.errno in UNREADABLE_ERRNOS:
            raise HTTPException(404, "File not found", headers={"Cache-Control": "no-store"}) from error
        raise


@router.get("/{project_id}/files", response_model=ProjectFileListResponse, response_model_exclude_none=True)
async def list_files(project_id: UUID, scope: ScopeDep):
    with _file_errors():
        return await run_efs_unit(lambda session: SharedProjectFiles(session).list_files(project_id), scope=scope)


@router.get("/{project_id}/files/{path:path}", response_model=ProjectFileReadResponse, response_model_exclude_none=True)
async def read_file(project_id: UUID, path: str, scope: ScopeDep):
    with _file_errors():
        content, info = await run_efs_unit(
            lambda session: SharedProjectFiles(session).read_file(project_id, path, limit=TEXT_MAX_BYTES), scope=scope
        )
        return ProjectFileReadResponse(path=path, content=content.decode("utf-8"), size=len(content), modified=info.st_mtime)


@router.get("/{project_id}/files-raw/{path:path}")
async def download_file(project_id: UUID, path: str, scope: ScopeDep):
    with _file_errors():
        content, _ = await run_efs_unit(
            lambda session: SharedProjectFiles(session).read_file(project_id, path, limit=DOWNLOAD_MAX_BYTES), scope=scope
        )
        return Response(content, media_type="application/octet-stream", headers={
            "Content-Disposition": attachment_disposition(path.rsplit("/", 1)[-1]),
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
            "Content-Security-Policy": "sandbox; default-src 'none'",
        })
