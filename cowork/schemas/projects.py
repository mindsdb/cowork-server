from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from cowork.schemas.base import CamelRequest
from cowork.schemas.shared_resources import ProjectCapabilities, ResourceAttribution


class ProjectCreateRequest(CamelRequest):
    name: str


class ProjectUpdateRequest(CamelRequest):
    name: str | None = None
    is_active: bool | None = None


class ProjectResponse(BaseModel):
    """Project wire shape; legacy project keys intentionally remain snake_case."""

    id: UUID
    created_at: datetime | None = None
    modified_at: datetime | None = None
    name: str
    # The label the user typed; `name` stays the slug. NULL means the row
    # predates the column, and every reader resolves it as `display_name or
    # name` (ENG-1676). It has to be declared here or FastAPI filters it out
    # of the response and the whole feature disappears from the wire.
    display_name: str | None = None
    path: str
    is_active: bool
    org_id: str | None = None
    created_by: str | None = None
    attribution: ResourceAttribution
    capabilities: ProjectCapabilities
