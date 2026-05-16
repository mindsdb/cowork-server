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
