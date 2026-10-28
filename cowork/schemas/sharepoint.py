"""What the SharePoint browse routes return.

Fixed fields only: Graph's own item objects carry pre-authenticated download
links, which must never reach a response.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class SharePointLibrary(BaseModel):
    id: str
    name: str


class SharePointLibrariesResponse(BaseModel):
    libraries: list[SharePointLibrary]


class SharePointItem(BaseModel):
    id: str
    name: str
    size: int
    kind: Literal["file", "folder", "other"]
    modified: str | None


class SharePointChildrenResponse(BaseModel):
    items: list[SharePointItem]
    cursor: str | None
