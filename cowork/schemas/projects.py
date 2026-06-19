from pathlib import Path
from pydantic import field_validator

from cowork.schemas.base import CamelRequest


class ProjectCreateRequest(CamelRequest):
    name: str
    path: Path | None = None

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: str | None) -> Path | None:
        return Path(path) if path else None


class ProjectUpdateRequest(CamelRequest):
    name: str | None = None
    is_active: bool | None = None
