from typing import Literal

from pydantic import BaseModel, Field

from cowork.schemas.base import CamelResponse


class ArtifactCapabilitiesResponse(CamelResponse):
    role: Literal["owner", "reviewer"]
    can_preview: bool
    can_comment: bool
    can_edit: bool
    can_address_with_agent: bool
    can_resolve_comments: bool


class ArtifactCardResponse(CamelResponse):
    id: str
    slug: str
    title: str
    description: str
    type: str
    kind: str
    ext: str
    updated: str
    mtime: int
    live: bool
    bg: str
    file_count: int
    folder: str
    path: str
    primary: str | None
    project_id: str | None
    project_name: str
    origin_conversation_id: str
    published_url: str
    modified: bool
    access_mode: Literal["public", "password", "restricted"]
    access_protected: bool
    access_password: str = ""
    access_emails: list[str] = Field(default_factory=list)
    org_allowed: bool
    owner_only: bool
    artifact_key: str
    serve_url: str
    draft_url: str = ""
    capabilities: ArtifactCapabilitiesResponse


class ArtifactPreviewResponse(BaseModel):
    path: str
    title: str
    kind: str
    mime: str
    content: str
    truncated: bool


class ArtifactOpenResponse(BaseModel):
    status: Literal["ok"]
    path: str
