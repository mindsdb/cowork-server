from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse as FileContentResponse

from cowork.db.scoped import ScopedSessionDep
from cowork.schemas.files import FileListResponse, FileResponse
from cowork.services.files import FileService


router = APIRouter()
# Every endpoint here uses the session solely for FileService — scoped module-wide.
SessionDep = ScopedSessionDep


@router.post("/", status_code=status.HTTP_201_CREATED, response_model=FileResponse)
async def upload_file(
    file: UploadFile,
    purpose: Annotated[str, Form()],
    session: SessionDep,
):
    # The "attachment:" namespace is server-owned: conversation attachments
    # are tagged "attachment:{conversation_id}" by the compat routes, and the
    # legacy-format rekey (cowork.db.migrations) rewrites multi-colon rows in
    # that namespace on boot. A client-minted purpose like "attachment:a:b"
    # would be silently mangled by that pass (ENG-338 review) — reject the
    # prefix here so every attachment tag is provably server-shaped.
    if purpose.startswith("attachment:"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The 'attachment:' purpose namespace is reserved; upload "
            "conversation attachments via /v1/attachments/… instead.",
        )
    return await FileService(session).create_file(file, purpose)


@router.get("/", response_model=FileListResponse)
def list_files(session: SessionDep, purpose: str | None = None):
    return FileListResponse(data=FileService(session).list_files(purpose=purpose))


@router.get("/{file_id}", response_model=FileResponse)
def retrieve_file(file_id: UUID, session: SessionDep):
    try:
        return FileService(session).get_file(file_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete("/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_file(file_id: UUID, session: SessionDep):
    found = FileService(session).delete_file(file_id)
    if not found:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")


@router.get("/{file_id}/content")
def retrieve_file_content(file_id: UUID, session: SessionDep):
    try:
        content_type, filename, path = FileService(session).get_file_content(file_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return FileContentResponse(path=str(path), media_type=content_type, filename=filename)
