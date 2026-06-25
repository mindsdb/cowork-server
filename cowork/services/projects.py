from __future__ import annotations

import re
import shutil
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.models.project import Project


GENERAL_PROJECT = "general"
GENERAL_PROJECT_ID = UUID("00000000-0000-0000-0000-000000000001")

_NAME_DISALLOWED = re.compile(r"[^A-Za-z0-9._-]+")
_NAME_HYPHEN_RUNS = re.compile(r"-{2,}")
_WIN_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}
_NAME_MAX_LEN = 48
_NAME_FALLBACK = "untitled-project"


class ProjectService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _root_dir(self) -> Path:
        return Path(get_app_settings().project.root_dir)

    def _project_path(self, name: str) -> Path:
        return self._root_dir() / name

    # TODO: Move this. This should only be done when using Anton.
    def _scaffold(self, target: Path) -> None:
        anton_dir = target / ".anton"
        anton_dir.mkdir(parents=True, exist_ok=True)
        (anton_dir / "anton.md").touch()

    def _unique_name(self, base: str, *, exclude: str | None = None) -> str:
        existing = {
            p.name for p in self.session.exec(select(Project)).all()
            if p.name != exclude
        }
        if base not in existing:
            return base
        i = 2
        while True:
            candidate = f"{base}-{i}"
            if candidate not in existing:
                return candidate
            i += 1

    def _sanitize_name(self, name: str) -> str:
        raw = (name or "").strip()
        cleaned = _NAME_DISALLOWED.sub("-", raw)
        cleaned = _NAME_HYPHEN_RUNS.sub("-", cleaned)
        cleaned = cleaned.strip("-._")
        if len(cleaned) > _NAME_MAX_LEN:
            cleaned = cleaned[:_NAME_MAX_LEN].rstrip("-._")
        if not cleaned:
            cleaned = _NAME_FALLBACK
        if cleaned.lower() in _WIN_RESERVED:
            cleaned = f"{cleaned}-x"
        return cleaned

    def list_projects(self, *, include_archived: bool = True) -> list[Project]:
        statement = select(Project)
        if not include_archived:
            statement = statement.where(Project.archived == False)  # noqa: E712
        # Deterministic order so the persisted organization metadata is
        # meaningful for any consumer (not just clients that re-sort): pinned
        # first, then manual sort_order, then name as a stable tiebreaker.
        statement = statement.order_by(
            Project.pinned.desc(),
            Project.sort_order.asc(),
            Project.name.asc(),
        )
        return list(self.session.exec(statement).all())

    def get_project(self, project_id: UUID) -> Project:
        project = self.session.get(Project, project_id)
        if project is None:
            raise ValueError("Project not found")
        return project

    def get_project_by_name(self, name: str) -> Project:
        project = self.session.exec(select(Project).where(Project.name == name)).first()
        if project is None:
            raise ValueError("Project not found")
        return project

    def get_project_by_name_or_none(self, name: str) -> Project | None:
        return self.session.exec(select(Project).where(Project.name == name)).first()

    def create_project(self, name: str) -> Project:
        sanitized = self._sanitize_name(name)
        final_name = self._unique_name(sanitized)
        path = self._project_path(final_name)
        path.mkdir(parents=True)
        # self._scaffold(path)
        project = Project(name=final_name, path=str(path), is_active=False)
        self.session.add(project)
        self.session.commit()
        self.session.refresh(project)
        return project

    def update_project(
        self,
        project_id: UUID,
        name: str | None = None,
        is_active: bool | None = None,
    ) -> Project:
        project = self.session.get(Project, project_id)
        if project is None:
            raise ValueError("Project not found")

        if name is not None:
            if project.name == GENERAL_PROJECT:
                raise ValueError("Cannot rename the General project")
            sanitized = self._sanitize_name(name)
            final_name = self._unique_name(sanitized, exclude=project.name)
            if final_name != project.name:
                old_path = Path(project.path)
                new_path = self._project_path(final_name)
                if old_path.exists():
                    old_path.rename(new_path)
                project.name = final_name
                project.path = str(new_path)

        if is_active is not None:
            if is_active:
                for other in self.session.exec(select(Project)).all():
                    if other.id != project_id and other.is_active:
                        other.is_active = False
                        self.session.add(other)
            project.is_active = is_active

        self.session.add(project)
        self.session.commit()
        self.session.refresh(project)
        return project

    def update_project_metadata(
        self,
        project_id: UUID,
        *,
        pinned: bool | None = None,
        sort_order: int | None = None,
        archived: bool | None = None,
        touch_last_selected: bool = False,
    ) -> Project:
        """Update organization metadata (pin/order/archived/last-selected).

        Independent of name/active-state mutation so the list UI can persist
        organization without side effects. Only provided fields change.
        """
        project = self.session.get(Project, project_id)
        if project is None:
            raise ValueError("Project not found")

        if pinned is not None:
            project.pinned = pinned
        if sort_order is not None:
            project.sort_order = sort_order
        if archived is not None:
            project.archived = archived
        if touch_last_selected:
            project.last_selected_at = datetime.now(timezone.utc)

        self.session.add(project)
        self.session.commit()
        self.session.refresh(project)
        return project

    def reorder_projects(self, ordered_ids: Iterable[UUID]) -> list[Project]:
        """Assign sort_order from the given id sequence (0, 1, 2, ...).

        Unknown ids are ignored; projects not named keep their existing order.
        Returns the full project list after reordering.
        """
        by_id = {p.id: p for p in self.session.exec(select(Project)).all()}
        for index, project_id in enumerate(ordered_ids):
            project = by_id.get(project_id)
            if project is not None and project.sort_order != index:
                project.sort_order = index
                self.session.add(project)
        self.session.commit()
        return list(self.session.exec(select(Project)).all())

    def delete_project(self, project_id: UUID) -> bool:
        project = self.session.get(Project, project_id)
        if project is None:
            return False
        if project.name == GENERAL_PROJECT:
            raise ValueError("Cannot delete the General project")
        path = Path(project.path)
        if path.exists():
            shutil.rmtree(path)
        was_active = project.is_active
        self.session.delete(project)
        self.session.commit()
        if was_active:
            general = self.session.get(Project, GENERAL_PROJECT_ID)
            if general is not None and not general.is_active:
                general.is_active = True
                self.session.add(general)
                self.session.commit()
        return True

    def get_active_project(self) -> Project:
        project = self.session.exec(select(Project).where(Project.is_active)).first()
        if project is None:
            raise ValueError("No active project")
        return project
