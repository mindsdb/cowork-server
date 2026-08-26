from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from cowork.coding.contracts import ResolvedSkill, utc_now


class ProjectSkillSource(BaseModel):
    source_id: str = Field(min_length=1, max_length=120)
    enabled_paths: list[str] = Field(default_factory=list, max_length=250)


class TeamSkillSource(BaseModel):
    schema_version: int = 1
    id: str
    name: str = Field(min_length=1, max_length=120)
    repository: str = Field(min_length=1, max_length=32_768)
    branch: str = Field(default="main", min_length=1, max_length=255)
    applied_revision: str
    available_revision: str
    cache_path: str
    last_checked_at: datetime = Field(default_factory=utc_now)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SkillLibraryItem(BaseModel):
    id: str
    kind: Literal["skill", "instructions", "workflow"]
    name: str
    description: str = ""
    origin: Literal["team", "personal", "built_in"]
    source_id: str | None = None
    source_name: str
    path: str
    version: str | None = None
    enabled: bool = True
    enabled_project_ids: list[str] = Field(default_factory=list)


class SkillLibrarySource(BaseModel):
    id: str
    name: str
    repository: str
    branch: str
    current_revision: str
    available_revision: str
    update_available: bool = False
    last_checked_at: datetime
    item_count: int = 0
    enabled_project_count: int = 0
    diff: str = ""
    error: str | None = None


class SkillLibraryPage(BaseModel):
    sources: list[SkillLibrarySource] = Field(default_factory=list)
    items: list[SkillLibraryItem] = Field(default_factory=list)


class SkillLibraryDocument(BaseModel):
    item: SkillLibraryItem
    files: list[str] = Field(min_length=1, max_length=100)
    selected_path: str
    content: str


class SkillSourceCreateRequest(BaseModel):
    name: str | None = Field(default=None, max_length=120)
    repository: str = Field(min_length=1, max_length=32_768)
    branch: str = Field(default="main", min_length=1, max_length=255)


class SkillSourceItemsRequest(BaseModel):
    enabled_paths: list[str] = Field(default_factory=list, max_length=250)


class SkillProjectAssignment(BaseModel):
    project_id: str = Field(min_length=1, max_length=120)
    enabled_paths: list[str] = Field(default_factory=list, max_length=250)


class SkillSourceAssignmentsRequest(BaseModel):
    assignments: list[SkillProjectAssignment] = Field(min_length=1, max_length=250)

    @model_validator(mode="after")
    def unique_projects(self) -> SkillSourceAssignmentsRequest:
        project_ids = [item.project_id for item in self.assignments]
        if len(project_ids) != len(set(project_ids)):
            raise ValueError("a project can only appear once in a skill assignment update")
        return self


class SkillResolution(BaseModel):
    items: list[ResolvedSkill] = Field(default_factory=list)
    roots: list[str] = Field(default_factory=list)
    developer_instructions: str = ""
    summary: str | None = None

    @model_validator(mode="after")
    def unique_items(self) -> SkillResolution:
        ids = [item.id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("resolved skill ids must be unique")
        return self
