from pathlib import Path

from pydantic import BaseModel


class FileResponse(BaseModel):
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str


class FileListResponse(BaseModel):
    object: str = "list"
    data: list[FileResponse]


class StoredFileContent(BaseModel):
    """Where an uploaded file's bytes are, and how to serve them."""

    content_type: str
    filename: str
    path: Path


class AttachmentMoveResponse(BaseModel):
    """An attachment moved into its project's root."""

    ok: bool
    project_path: str
    absolute_path: str
    content_type: str
