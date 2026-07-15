from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from uuid import UUID

from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.models.project import Project

logger = logging.getLogger(__name__)


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

    def list_projects(self) -> list[Project]:
        return list(self.session.exec(select(Project)).all())

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

        from cowork.services.skill_links import reconcile_project
        from cowork.services.skills import SkillService
        reconcile_project(path, SkillService().list_skills())

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
                old_name = project.name
                old_path = Path(project.path)
                new_path = self._project_path(final_name)
                if old_path.exists():
                    old_path.rename(new_path)
                project.name = final_name
                project.path = str(new_path)

                # Update skill metadata that referenced the old project name,
                # then reconcile links for the renamed dir.
                from cowork.services.skill_links import reconcile_project
                from cowork.services.skills import SkillService
                svc = SkillService()
                for skill in svc.list_skills():
                    if old_name in skill.projects:
                        updated = [final_name if p == old_name else p for p in skill.projects]
                        svc.update_skill(skill.name, projects=updated)
                reconcile_project(new_path, svc.list_skills())

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

    def delete_project(self, project_id: UUID) -> bool:
        project = self.session.get(Project, project_id)
        if project is None:
            return False
        if project.name == GENERAL_PROJECT:
            raise ValueError("Cannot delete the General project")
        # Cascade to the project's conversations FIRST (ENG-701). Deleting a
        # project used to only rmtree its dir + drop the row, orphaning every
        # conversation in it — and their messages, events, task objects, and
        # uploaded attachments (whose bytes live OUTSIDE the project dir, so the
        # rmtree never reached them). There's no DB-level FK cascade. Deleting
        # each conversation cleans all of that up (incl. attachments), and does
        # it while the conversation still exists so the cleanup is safe.
        from cowork.models.conversation import Conversation
        from cowork.services.conversations import ConversationService
        conv_svc = ConversationService(self.session)
        conv_ids = list(
            self.session.exec(
                select(Conversation.id).where(Conversation.project_id == project_id)
            ).all()
        )
        for cid in conv_ids:
            # Fault-isolated: one conversation failing to delete must not abort
            # the whole project delete and leave it half-cascaded. Log and move
            # on — a skipped conversation just retains today's orphan behavior.
            try:
                conv_svc.delete_conversation(cid)
            except Exception:
                # Roll back the failed conversation's partial work FIRST.
                # delete_conversation stages its row deletes (messages, events,
                # task objects, attachment rows) before its own commit; without
                # this rollback those pending deletes would be silently flushed
                # by the next commit in the cascade (or the project commit below)
                # — wiping the conversation's data while its row survives.
                self.session.rollback()
                logger.warning(
                    "delete_project: failed to delete conversation %s; skipping", cid,
                    exc_info=True,
                )
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
