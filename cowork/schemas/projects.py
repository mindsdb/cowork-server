from pydantic import BaseModel


class ProjectCreateRequest(BaseModel):
    name: str


class ProjectUpdateRequest(BaseModel):
    name: str | None = None
    is_active: bool | None = None
