from pathlib import Path
from pydantic import field_validator

from cowork.schemas.base import CamelRequest


class ProjectCreateRequest(CamelRequest):
    name: str
    path: Path | None = None
    instructions: str | None = None

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: Path | None) -> Path | None:
        # A relative path would be created against the server process's
        # working directory, and pathlib never expands "~" on its own —
        # both silently land somewhere the user didn't pick.
        if path is None or not str(path).strip():
            return None
        expanded = Path(path).expanduser()
        if not expanded.is_absolute():
            raise ValueError("path must be an absolute path")
        return expanded


class ProjectUpdateRequest(CamelRequest):
    name: str | None = None
    is_active: bool | None = None
    instructions: str | None = None