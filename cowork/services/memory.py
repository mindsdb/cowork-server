from __future__ import annotations

from pathlib import Path
from uuid import UUID

from sqlmodel import Session, select

from cowork.harnesses.memory.registry import MemorySlot
from cowork.harnesses.memory.store import PROJECT_SLOTS, ProjectMemoryStore, SharedMemoryStore
from cowork.models.project import Project
from cowork.schemas.memory import MemoryResponse, MemoryScope


class MemoryService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self._global_store = SharedMemoryStore()

    async def get_memory(
        self,
        scope: MemoryScope,
        category: MemorySlot,
        project_id: UUID | None = None,
    ) -> MemoryResponse:
        content = self._read(scope, category, project_id)
        return MemoryResponse(
            scope=scope,
            category=category,
            content=content,
            project_id=project_id if scope == MemoryScope.project else None,
        )

    async def update_memory(
        self,
        scope: MemoryScope,
        category: MemorySlot,
        content: str,
        project_id: UUID | None = None,
    ) -> MemoryResponse:
        self._write(scope, category, content, project_id)
        return MemoryResponse(
            scope=scope,
            category=category,
            content=content,
            project_id=project_id if scope == MemoryScope.project else None,
        )

    async def delete_memory(
        self,
        scope: MemoryScope,
        category: MemorySlot,
        project_id: UUID | None = None,
    ) -> None:
        self._delete(scope, category, project_id)

    async def list_memory(self, project_id: UUID | None = None) -> list[MemoryResponse]:
        items: list[MemoryResponse] = []

        for category in MemorySlot:
            items.append(
                MemoryResponse(
                    scope=MemoryScope.global_,
                    category=category,
                    content=self._global_store.read(category),
                    project_id=None,
                )
            )

        projects = list(self.session.exec(select(Project)).all())
        if project_id is not None:
            projects = [p for p in projects if p.id == project_id]

        for project in projects:
            store = ProjectMemoryStore(Path(project.path))
            for category in PROJECT_SLOTS:
                items.append(
                    MemoryResponse(
                        scope=MemoryScope.project,
                        category=category,
                        content=store.read(category),
                        project_id=project.id,
                    )
                )

        if project_id is not None:
            items = [
                item
                for item in items
                if item.scope == MemoryScope.global_ or item.project_id == project_id
            ]

        return items

    def _read(self, scope: MemoryScope, category: MemorySlot, project_id: UUID | None) -> str:
        if scope == MemoryScope.global_:
            return self._global_store.read(category)
        project = self._resolve_project(project_id)
        return ProjectMemoryStore(Path(project.path)).read(category)

    def _write(
        self,
        scope: MemoryScope,
        category: MemorySlot,
        content: str,
        project_id: UUID | None,
    ) -> None:
        if scope == MemoryScope.global_:
            self._global_store.write(category, content)
            return
        project = self._resolve_project(project_id)
        ProjectMemoryStore(Path(project.path)).write(category, content)

    def _delete(
        self,
        scope: MemoryScope,
        category: MemorySlot,
        project_id: UUID | None,
    ) -> None:
        if scope == MemoryScope.global_:
            self._global_store.delete(category)
            return
        project = self._resolve_project(project_id)
        ProjectMemoryStore(Path(project.path)).delete(category)

    def _resolve_project(self, project_id: UUID | None) -> Project:
        if project_id is None:
            raise ValueError("project_id is required for project-scoped memory.")
        project = self.session.get(Project, project_id)
        if project is None:
            raise ValueError(f"Project {project_id} not found.")
        return project
