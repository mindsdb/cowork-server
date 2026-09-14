from datetime import datetime
from pathlib import PurePath
from uuid import UUID

from pydantic import BaseModel, field_validator

from cowork.schemas.base import CamelRequest
from cowork.schemas.shared_resources import ProjectCapabilities, ResourceAttribution


class ProjectCreateRequest(CamelRequest):
    name: str
    # A string, not a Path: pydantic coerces "" to Path("."), which is then
    # indistinguishable from a caller who really sent ".". One of those means
    # "no folder chosen" and the other is a relative path to refuse.
    path: str | None = None

    @field_validator("path")
    @classmethod
    def _absolute_path_only(cls, path: str | None) -> str | None:
        # Syntax only: a field validator runs before the handler, so a
        # filesystem check here would precede the service's tenancy refusal.
        if path is None or not path.strip():
            return None
        # Not stripped. A directory name may legally end in a space on POSIX
        # and the picker returns it verbatim, so trimming would refuse a
        # folder that exists. Leading whitespace fails the absolute check
        # below rather than being silently repaired.
        raw = path
        # `~` is deliberately not expanded. expanduser consults the passwd
        # database, so over HTTP it separates a real account from a missing one
        # before the request has been refused at all. The picker sends an
        # absolute path, so nothing legitimate needs it.
        if raw.startswith("~"):
            raise ValueError("path must be absolute, not ~-relative")
        if not PurePath(raw).is_absolute():
            raise ValueError("path must be absolute")
        return raw


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
