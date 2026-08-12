from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlmodel import select

from cowork.common.paths import safe_join
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession, scoped_storage_root, unsafe_unscoped_session
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
    def __init__(self, session: ScopedSession) -> None:
        self.session = session

    def ensure_general_for_scope(self) -> Project | None:
        """The caller's default project, provisioned on demand.

        Desktop keeps the seeded GENERAL row. Org mode gives each org its own row
        and directory: one seeded id can't serve N tenants (the first org claimed
        it, the rest got None → 404). Idempotent — called on every request.
        """
        scope = self.session.scope
        if not scope.org_mode or scope.org_id is None:
            project = self.session.get(Project, GENERAL_PROJECT_ID)
            if project is not None:
                self._ensure_dir_exists(project)
            return project

        existing = self.get_project_by_name_or_none(GENERAL_PROJECT)
        if existing is not None:
            self._repoint_if_stale(existing)
            self._ensure_dir_exists(existing)
            return existing

        path = self._project_path(GENERAL_PROJECT)
        path.mkdir(parents=True, exist_ok=True)
        self._insert_general_if_absent(path)
        return self.get_project_by_name_or_none(GENERAL_PROJECT)

    def _insert_general_if_absent(self, path: Path) -> None:
        """Insert this org's default project unless it already has one.

        Separate method so the no-duplicate property is testable — the caller's
        pre-check would short-circuit before the insert.
        """
        scope = self.session.scope
        # One atomic statement: `projects` has no unique constraint, so the two
        # replicas would otherwise both insert on first load. Core insert, not
        # session.add, so the flush hook can't stamp created_by — this project is
        # the org's, not the first member's. Timestamps: column server_default.
        raw = unsafe_unscoped_session(self.session)  # bootstrap op, not query path
        raw.execute(
            sa.insert(Project)
            .from_select(
                ["id", "name", "path", "is_active", "org_id"],
                sa.select(
                    sa.literal(str(uuid4())),
                    sa.literal(GENERAL_PROJECT),
                    sa.literal(str(path)),
                    sa.literal(False),
                    sa.literal(scope.org_id),
                ).where(
                    ~sa.exists().where(
                        Project.name == GENERAL_PROJECT,  # type: ignore[arg-type]
                        Project.org_id == scope.org_id,  # type: ignore[arg-type]
                    )
                ),
            )
        )
        raw.commit()

    def _repoint_if_stale(self, project: Project) -> None:
        """Move a row off a pre-org-keyed path, but only when nothing is there.

        Such rows point at `<root>/<name>`, which _ensure_dir_exists won't
        recreate, so they resolve to a missing directory forever. A path that
        still has content stays put — swapping in an empty dir strands work.
        """
        current = Path(project.path)
        try:
            root = self._root_dir().resolve()
            if current.resolve().parent == root or current.is_dir():
                return
        except OSError:
            return
        project.path = str(self._project_path(project.name))
        self.session.add(project)
        self.session.commit()
        logger.info("re-pointed %r off a pre-org-keyed path: %s", project.name, current)

    def _ensure_dir_exists(self, project: Project) -> None:
        """Recreate a missing directory for a project the caller owns.

        The row is authoritative; a missing dir is unprovisioned state, not a
        reason to 404 an org out of its own project. Scoped root only, so a
        stale path from another deployment is left alone.
        """
        path = Path(project.path)
        if path.is_dir():
            return
        try:
            root = self._root_dir().resolve()
            if path.resolve().parent != root:
                return
        except OSError:
            return
        path.mkdir(parents=True, exist_ok=True)
        logger.info("provisioned missing project directory: %s", path)

    def _root_dir(self) -> Path:
        """Projects root, org-keyed in org mode (same helper as skills/memory).

        Without the org segment all tenants shared one directory: two orgs using
        the same project name collided, and the second create hit an existing
        dir — a cross-org existence oracle.
        """
        return scoped_storage_root(Path(get_app_settings().project.root_dir), self.session.scope)

    def _project_path(self, name: str) -> Path:
        # Containment guard: a project dir is always a direct child of the
        # projects root, regardless of what sanitization produced. Validate
        # the name before building the path, then re-check the result.
        root = self._root_dir().resolve()
        if not name or _NAME_DISALLOWED.search(name):
            raise ValueError("Invalid project name")
        candidate = Path(name)
        if candidate.is_absolute() or len(candidate.parts) != 1 or candidate.name in {"", ".", ".."}:
            raise ValueError("Invalid project name")
        # safe_join rejects anything that escapes root; the parent re-check keeps
        # a project dir a *direct* child of the projects root.
        path = safe_join(root, candidate.name)
        if path.parent != root or path == root:
            raise ValueError("Invalid project name")
        return path

    # TODO: Move this. This should only be done when using Anton.
    def _scaffold(self, target: Path) -> None:
        anton_dir = target / ".anton"
        anton_dir.mkdir(parents=True, exist_ok=True)
        (anton_dir / "anton.md").touch()

    def _unique_name(self, base: str, *, exclude: str | None = None) -> str:
        existing = {
            p.name for p in self.session.exec(self.session.select(Project)).all()
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
        return list(self.session.exec(self.session.select(Project)).all())

    def get_project(self, project_id: UUID) -> Project:
        project = self.session.get(Project, project_id)
        if project is None:
            raise ValueError("Project not found")
        return project

    def get_project_by_name(self, name: str) -> Project:
        project = self.session.exec(
            self.session.select(Project).where(Project.name == name)
        ).first()
        if project is None:
            raise ValueError("Project not found")
        return project

    def get_project_by_name_or_none(self, name: str) -> Project | None:
        return self.session.exec(
            self.session.select(Project).where(Project.name == name)
        ).first()

    def create_project(self, name: str) -> Project:
        sanitized = self._sanitize_name(name)
        final_name = self._unique_name(sanitized)
        path = self._project_path(final_name)
        # exist_ok: the root is org-keyed, so a leftover dir is this org's own.
        path.mkdir(parents=True, exist_ok=True)
        # self._scaffold(path)
        project = Project(name=final_name, path=str(path), is_active=False)
        self.session.add(project)
        self.session.commit()
        self.session.refresh(project)

        # Skill symlink distribution is desktop-only (see SkillService).
        if not self.session.scope.org_mode:
            from cowork.services.skill_links import reconcile_project
            from cowork.services.skills import SkillService
            reconcile_project(path, SkillService(self.session.scope).list_skills())

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
                svc = SkillService(self.session.scope)
                for skill in svc.list_skills():
                    if old_name in skill.projects:
                        updated = [final_name if p == old_name else p for p in skill.projects]
                        svc.update_skill(skill.name, projects=updated)
                if not self.session.scope.org_mode:  # symlinks are desktop-only
                    reconcile_project(new_path, svc.list_skills())

        if is_active is not None:
            if is_active:
                for other in self.session.exec(self.session.select(Project)).all():
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
        conv_ids = [
            c.id
            for c in self.session.exec(
                self.session.select(Conversation).where(Conversation.project_id == project_id)
            ).all()
        ]
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
        project = self.session.exec(
            self.session.select(Project).where(Project.is_active)
        ).first()
        if project is None:
            raise ValueError("No active project")
        return project
