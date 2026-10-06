from pydantic import BaseModel

from cowork.schemas.shared_resources import (
    MutableResourceCapabilities,
    ResourceAttribution,
)


class ProjectFileMetadata(BaseModel):
    path: str
    name: str
    size: int
    modified: float | None
    is_dir: bool
    synthetic: bool | None = None
    attribution: ResourceAttribution | None = None
    capabilities: MutableResourceCapabilities | None = None


class AttributedProjectFileMetadata(ProjectFileMetadata):
    attribution: ResourceAttribution
    capabilities: MutableResourceCapabilities


class ProjectInstructionsResponse(BaseModel):
    file: AttributedProjectFileMetadata


class ProjectFileListResponse(BaseModel):
    files: list[ProjectFileMetadata]
    # Set only when the walk hit its cap, so an untruncated listing keeps the
    # wire shape it had before a project could point at a folder of any size.
    truncated: bool | None = None


class ProjectFileReadResponse(BaseModel):
    path: str
    content: str
    size: int
    modified: float | None
    attribution: ResourceAttribution | None = None
    capabilities: MutableResourceCapabilities | None = None


class ProjectFileWriteResponse(BaseModel):
    path: str
    size: int
    modified: float
    attribution: ResourceAttribution | None = None
    capabilities: MutableResourceCapabilities | None = None


class ProjectFileDeleteResponse(BaseModel):
    status: str
    path: str
