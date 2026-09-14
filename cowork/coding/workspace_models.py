from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class WorkspaceResource(BaseModel):
    id: str
    name: str
    kind: Literal["repository", "folder"]


class WorkspaceResourcePage(BaseModel):
    items: list[WorkspaceResource]


class WorkspaceEntry(BaseModel):
    resource_id: str
    resource_name: str
    path: str
    name: str
    kind: Literal["file", "directory"]
    size: int | None = None


class WorkspaceEntryPage(BaseModel):
    resource_id: str
    path: str
    items: list[WorkspaceEntry]
    truncated: bool = False


class WorkspaceFileContent(BaseModel):
    resource_id: str
    resource_name: str
    path: str
    name: str
    content: str
    content_hash: str
    line_count: int
    line_start: int
    line_end: int
    truncated: bool = False


class WorkspaceSearchMatch(BaseModel):
    resource_id: str
    resource_name: str
    path: str
    name: str
    line: int | None = None
    preview: str = ""
    match_kind: Literal["path", "content"]


class WorkspaceSearchPage(BaseModel):
    items: list[WorkspaceSearchMatch]
    truncated: bool = False

