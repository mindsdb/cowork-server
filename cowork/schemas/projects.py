
from cowork.schemas.base import CamelRequest


class ProjectCreateRequest(CamelRequest):
    name: str


class ProjectUpdateRequest(CamelRequest):
    name: str | None = None
    is_active: bool | None = None
