from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskRepositorySetup(BaseModel):
    """Per-task choices, never persisted back into the project's defaults."""

    model_config = ConfigDict(extra="forbid")
    branch: str | None = Field(default=None, min_length=1, max_length=255)
    base_branches: dict[str, str] = Field(default_factory=dict, max_length=64)
    include_local_changes: bool = False

    @field_validator("base_branches")
    @classmethod
    def bounded_branches(cls, value: dict[str, str]) -> dict[str, str]:
        if any(
            not key or len(key) > 120 or not branch or len(branch) > 255
            for key, branch in value.items()
        ):
            raise ValueError("Choose a valid resource and base branch")
        return value


class RepositoryStatus(BaseModel):
    resource_id: str
    available: bool = True
    local: bool = False
    branch: str | None = None
    branches: list[str] = Field(default_factory=list)
    changes: list[str] = Field(default_factory=list)
    change_count: int = 0
    detail: str = ""
