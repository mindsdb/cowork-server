"""Read-only recipient routes. Every request rechecks the operator's grant."""
from contextlib import contextmanager
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import ScopedSessionDep
from cowork.schemas.project_files import ProjectFileListResponse, ProjectFileReadResponse
from cowork.services.shared_projects import DOWNLOAD_MAX_BYTES, TEXT_MAX_BYTES, UNREADABLE_ERRNOS, SharedProjectFiles


def _no_cache(response: Response):
    response.headers["Cache-Control"] = "no-store"


router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode)), Depends(_no_cache)])


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
def list_files(project_id: UUID, session: ScopedSessionDep):
    with _file_errors():
        return SharedProjectFiles(session).list_files(project_id)


@router.get("/{project_id}/files/{path:path}", response_model=ProjectFileReadResponse, response_model_exclude_none=True)
def read_file(project_id: UUID, path: str, session: ScopedSessionDep):
    with _file_errors():
        content, info = SharedProjectFiles(session).read_file(project_id, path, limit=TEXT_MAX_BYTES)
        return ProjectFileReadResponse(path=path, content=content.decode("utf-8"), size=len(content), modified=info.st_mtime)


@router.get("/{project_id}/files-raw/{path:path}")
def download_file(project_id: UUID, path: str, session: ScopedSessionDep):
    with _file_errors():
        content, _ = SharedProjectFiles(session).read_file(project_id, path, limit=DOWNLOAD_MAX_BYTES)
        return Response(content, media_type="application/octet-stream", headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(path.rsplit('/', 1)[-1], safe='')}",
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
            "Content-Security-Policy": "sandbox; default-src 'none'",
        })
