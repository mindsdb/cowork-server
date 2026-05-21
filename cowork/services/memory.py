from __future__ import annotations

from uuid import UUID

from sqlmodel import Session

from cowork.common.settings.user_settings import get_user_settings
from cowork.harnesses.base import get_harness
from cowork.models.project import Project
from cowork.schemas.memory import MemoryResponse, MemoryScope

settings = get_user_settings()


class MemoryService:
    def __init__(self, session: Session) -> None:
        self.session = session

    async def get_memory(self, scope: MemoryScope, category: str, project_id: UUID | None = None) -> MemoryResponse:
        harness = get_harness(settings.harness)
        project = self._resolve_project(scope, project_id)
        content = await harness.retrieve_memory(scope, category, project)
        return MemoryResponse(scope=scope, category=category, content=content or "", project_id=project_id)

    async def update_memory(self, scope: MemoryScope, category: str, content: str, project_id: UUID | None = None) -> MemoryResponse:
        harness = get_harness(settings.harness)
        project = self._resolve_project(scope, project_id)
        await harness.overwrite_memory(scope, category, content, project)
        return MemoryResponse(scope=scope, category=category, content=content, project_id=project_id)

    async def delete_memory(self, scope: MemoryScope, category: str, project_id: UUID | None = None) -> None:
        harness = get_harness(settings.harness)
        project = self._resolve_project(scope, project_id)
        await harness.overwrite_memory(scope, category, "", project)

    def _resolve_project(self, scope: MemoryScope, project_id: UUID | None) -> Project | None:
        if scope == MemoryScope.project:
            if project_id is None:
                raise ValueError("project_id is required for project-scoped memory.")
            project = self.session.get(Project, project_id)
            if project is None:
                raise ValueError(f"Project {project_id} not found.")
            return project
        return None