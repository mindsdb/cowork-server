from datetime import datetime
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, field_validator

from cowork.schemas.base import CamelRequest
from cowork.schemas.shared_resources import ProjectCapabilities, ResourceAttribution


class ProjectCreateRequest(CamelRequest):
    name: str
    path: Path | None = None

    @field_validator("path")
    @classmethod
    def _absolute_path_only(cls, path: Path | None) -> Path | None:
        # Syntax only: a field validator runs before the handler, so a
        # filesystem check here would precede the service's tenancy refusal.
        if path is None or not str(path).strip():
            return None
        try:
            expanded = path.expanduser()
        except RuntimeError as exc:
            raise ValueError("path could not be resolved") from exc
        if not expanded.is_absolute():
            raise ValueError("path must be absolute")
        return expanded


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
