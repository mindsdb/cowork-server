from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import logging
import re
from pathlib import Path
from uuid import UUID, uuid4

import sqlalchemy as sa

from cowork.common.paths import (
    PinnedDir,
    dir_mkdir,
    dir_rename_into,
    dir_rmtree,
    pinned_dir,
    safe_join,
)
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
                self.ensure_dir_exists(project)
            return project

        existing = self.get_project_by_name_or_none(GENERAL_PROJECT)
        if existing is not None:
            self._repoint_if_stale(existing)
            self.ensure_dir_exists(existing)
            return existing

        path = self._project_path(GENERAL_PROJECT)
        self._mkdir_in_root(path, exist_ok=True)
        self._insert_general_if_absent(path)
        return self.get_project_by_name_or_none(GENERAL_PROJECT)

    def default_project_id(self) -> UUID | None:
        """Id of the caller's default project — never the fixed constant.

        Org rows carry their own uuid, so `GENERAL_PROJECT_ID` resolves to None in
        org mode (the seeded row keeps org_id NULL) and every "no project given"
        caller 404'd. Provisions on first use, so it doubles as the bootstrap.
        """
        project = self.ensure_general_for_scope()
        return project.id if project is not None else None

    def _insert_general_if_absent(self, path: Path) -> None:
        """Insert this org's default project unless it already has one.

        Separate method so the no-duplicate property is testable — the caller's
        pre-check would short-circuit before the insert.
        """
        scope = self.session.scope
        # NOT EXISTS narrows the race; `uq_projects_default_per_org` settles it —
        # on Postgres both replicas can pass the check at READ COMMITTED (only
        # SQLite serialises writes). The caller re-reads, so the loser adopts the
        # winner's row. Core insert, not session.add, so the flush hook can't stamp
        # created_by: this project is the org's, not the first member's.
        raw = unsafe_unscoped_session(self.session)  # bootstrap op, not query path
        try:
            self._execute_general_insert(raw, path, scope.org_id)
        except sa.exc.IntegrityError:
            raw.rollback()

    @staticmethod
    def _insert_stmt(raw):
        """Dialect insert with ON CONFLICT DO NOTHING where supported.

        Preferred over catching IntegrityError: on Postgres a failed insert aborts
        the transaction, so the loser would have to roll back work the request may
        still need. Unknown dialects fall back to a plain insert — the caller's
        IntegrityError handler still covers them.
        """
        name = raw.get_bind().dialect.name
        if name == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as dialect_insert
        elif name == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as dialect_insert
        else:
            return sa.insert(Project)
        return dialect_insert(Project)

    def _execute_general_insert(self, raw, path: Path, org_id: str | None) -> None:
        stmt = (
            self._insert_stmt(raw)
            .from_select(
                ["id", "name", "path", "is_active", "org_id"],
                sa.select(
                    # type_ is required: a bare literal binds as String and skips
                    # the Uuid column's bind processor, so the id is stored in a
                    # shape no other lookup matches ("Project not found").
                    sa.literal(uuid4(), type_=sa.Uuid()),
                    sa.literal(GENERAL_PROJECT),
                    sa.literal(str(path)),
                    # Active: it is the org's only project at this point, and
                    # get_active_project raises when nothing is active.
                    sa.literal(True),
                    sa.literal(org_id),
                ).where(
                    ~sa.exists().where(
                        Project.name == GENERAL_PROJECT,  # type: ignore[arg-type]
                        Project.org_id == org_id,  # type: ignore[arg-type]
                    )
                ),
            )
        )
        if hasattr(stmt, "on_conflict_do_nothing"):
            stmt = stmt.on_conflict_do_nothing()
        raw.execute(stmt)
        raw.commit()

    def _in_scoped_root(self, path: Path) -> bool:
        """Whether `path` sits directly in the caller's projects root.

        Shared by _repoint_if_stale and ensure_dir_exists, which had the same
        containment block with the comparison inverted.
        """
        try:
            return path.resolve().parent == self._root_dir().resolve()
        except OSError:
            return False

    def _repoint_if_stale(self, project: Project) -> None:
        """Move a row off a pre-org-keyed path when that path holds nothing.

        Such rows point at `<root>/<name>`, which ensure_dir_exists won't
        recreate, so they resolve to a missing directory forever. A path with
        content stays put — swapping in an empty dir would strand the org's work.
        An empty directory is not content, so it moves.
        """
        current = Path(project.path)
        if self._in_scoped_root(current):
            return  # already org-keyed
        try:
            if current.is_dir() and any(current.iterdir()):
                return  # real content — leave it where it is
        except OSError:
            return
        new_path = str(self._project_path(project.name))
        # Core UPDATE, not session.add: the flush hook stamps created_by on any row
        # where it is None, which would attribute the org's system project to
        # whoever happened to trigger the heal.
        raw = unsafe_unscoped_session(self.session)
        raw.execute(
            sa.update(Project).where(Project.id == project.id).values(path=new_path)  # type: ignore[arg-type]
        )
        raw.commit()
        raw.refresh(project)
        logger.info("re-pointed %r off a pre-org-keyed path: %s", project.name, current)

    def ensure_dir_exists(self, project: Project) -> None:
        """Recreate a missing directory for a project the caller owns.

        Public because any project can lose its directory (a fresh pod, a wiped
        volume), not just the default one — see project_files._project_dir.

        The row is authoritative; a missing dir is unprovisioned state, not a
        reason to 404 an org out of its own project. Scoped root only, so a
        stale path from another deployment is left alone.
        """
        if Path(project.path).is_dir():
            return
        # Rebuild the target from the (sanitized, containment-checked) name rather
        # than mkdir-ing the stored string, so the value reaching the filesystem
        # is always one _project_path produced — never a raw DB/HTTP value. Only
        # create it when it matches what the row already claims, so a stale or
        # foreign path is left alone rather than silently re-homed here.
        try:
            safe = self._project_path(project.name)
        except ValueError:
            return
        if safe.resolve() != Path(project.path).resolve():
            return
        self._mkdir_in_root(safe, exist_ok=True)
        logger.info("provisioned missing project directory: %s", safe)

    def _root_dir(self) -> Path:
        """Projects root, org-keyed in org mode (same helper as skills/memory).

        Without the org segment all tenants shared one directory: two orgs using
        the same project name collided, and the second create hit an existing
        dir — a cross-org existence oracle.
        """
        return scoped_storage_root(
            Path(get_app_settings().project.root_dir), self.session.scope, store="projects"
        )

    # --- Symlink-safe filesystem ops on the projects root -------------------
    #
    # _project_path proves a path is contained, but the proof expires the moment
    # it returns: the caller's mkdir/rename/rmtree makes the kernel walk the
    # path again, and the agent can replace a component in between. Its pod
    # mounts this org's subtree read-write, so it can swap `projects` itself for
    # a symlink into another org and redirect the operation.
    #
    # These pin the root by descriptor once, with O_NOFOLLOW so opening it can't
    # traverse a symlink either, and act relative to that descriptor. The kernel
    # then resolves only the final component, against the inode we opened, so
    # there is nothing left to swap. Safe because _project_path already
    # guarantees a project dir is a DIRECT child of root.

    @contextmanager
    def _root_fd(self) -> "Iterator[PinnedDir]":
        """A handle for the projects root, refusing to follow a symlink (POSIX)."""
        with pinned_dir(self._root_dir(), create=True, nofollow_base=True) as d:
            yield d

    def _child_name(self, path: Path) -> str:
        """The single component *path* adds to the projects root.

        Refuses anything that is not a direct child, so a caller cannot smuggle
        a nested or absolute path into an operation that assumes one level:
        dir_fd only makes the FINAL component safe.
        """
        root = self._root_dir()
        if path.parent != root and path.parent.resolve() != root.resolve():
            raise ValueError(f"{path} is not a direct child of the projects root")
        return path.name

    def _mkdir_in_root(self, path: Path, *, exist_ok: bool = False) -> None:
        with self._root_fd() as root:
            try:
                dir_mkdir(root, self._child_name(path))
            except FileExistsError:
                if not exist_ok:
                    raise

    def _rename_in_root(self, old: Path, new: Path) -> None:
        """Rename *old* to *new*, where *new* is a direct child of the root.

        Only the DESTINATION is pinned. The source is passed absolute on
        purpose: a legacy row still on a pre-org-keyed path lives outside this
        root entirely, and that path is one we already hold rather than one an
        agent can redirect us into. The destination is the side an attacker
        would aim at another org, so that is the side that must not be
        re-walked.
        """
        with self._root_fd() as root:
            dir_rename_into(root, old, self._child_name(new))

    def _rmtree_in_root(self, path: Path) -> None:
        with self._root_fd() as root:
            dir_rmtree(root, self._child_name(path))

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
        # Ordered: `.first()` on an unordered select picks an arbitrary row, so a
        # pre-index duplicate would resolve differently per call. Oldest wins,
        # matching the migration's de-dupe.
        return self.session.exec(
            self.session.select(Project)
            .where(Project.name == name)
            .order_by(Project.created_at, Project.id)
        ).first()

    def get_or_provision_by_name_or_none(self, name: str) -> Project | None:
        """Resolve a project by name, provisioning the org's default on a `general` miss.

        The default project is created lazily per org (`ensure_general_for_scope`), so a
        plain by-name lookup of `general` can 404 before the org's first `GET /projects/`
        has provisioned its row — a send or task-create that names `general` would fail on
        a project that is supposed to always exist (ENG-1847). Provision the reserved name
        here instead of missing; every other name stays an exact match.

        The single home for the reserved-name self-heal: name-lookup callers that must
        not auto-provision (`get_project_by_name`, project-files, compat stubs) stay on the
        plain primitives. Can't fold into those primitives — `ensure_general_for_scope`
        calls `get_project_by_name_or_none`, so self-healing there would recurse.
        """
        if name == GENERAL_PROJECT:
            return self.ensure_general_for_scope()
        return self.get_project_by_name_or_none(name)

    def get_or_provision_by_name(self, name: str) -> Project:
        """`get_or_provision_by_name_or_none`, raising when the name is genuinely missing."""
        project = self.get_or_provision_by_name_or_none(name)
        if project is None:
            raise ValueError("Project not found")
        return project

    def _allocate_project_dir(self, base: str) -> tuple[str, Path]:
        """Claim a directory by creating it, bumping the name on collision."""
        candidate = self._unique_name(base)
        for attempt in range(2, 52):
            path = self._project_path(candidate)
            try:
                self._mkdir_in_root(path)  # raises if it already exists
                return candidate, path
            except FileExistsError:
                candidate = self._unique_name(f"{base}-{attempt}")
        raise ValueError("Could not allocate a project directory")

    def create_project(self, name: str) -> Project:
        sanitized = self._sanitize_name(name)
        # `general` belongs to the system row. A member creating it first would own
        # an undeletable, user-attributed project AND block the real default,
        # which is looked up by name.
        if sanitized == GENERAL_PROJECT:
            sanitized = f"{GENERAL_PROJECT}-2"
        # No exist_ok: `_unique_name` is a read-then-write with no unique
        # constraint behind it, so two concurrent creates can pick the same name.
        # Letting mkdir fail keeps them from sharing one directory (where deleting
        # either would rmtree the other's files) and stops a leftover directory
        # being adopted with stale contents. On collision, take the next name.
        final_name, path = self._allocate_project_dir(sanitized)
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
                    # The org's root may not exist yet (a legacy row still on an
                    # un-keyed path): rename would raise FileNotFoundError, which
                    # the endpoint's `except ValueError` turns into a 500.
                    self._rename_in_root(old_path, new_path)
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
                # Owner-AGNOSTIC: a project is org-shared and may hold several
                # members' conversations. Fetch org-scoped (session.get validates
                # org, not owner) and delete the row directly — the request-facing
                # delete_conversation is owner-scoped and would skip foreign rows,
                # re-orphaning them (ENG-701).
                conv = self.session.get(Conversation, cid)
                if conv is None:
                    continue
                conv_svc.delete_conversation_row(conv)
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
        # rmtree only a path re-derived from the sanitized name inside the
        # org-keyed root (same rebuild-and-compare as ensure_dir_exists). The
        # stored string can be stale (pre-org-keying) or tampered, and deleting
        # it verbatim would rmtree an arbitrary directory. A mismatched dir is
        # left behind instead — an orphaned directory beats a wrong-target rm.
        path = Path(project.path)
        try:
            safe = self._project_path(project.name)
        except ValueError:
            safe = None
        if path.exists():
            if safe is not None and safe.resolve() == path.resolve():
                self._rmtree_in_root(path)
            else:
                logger.warning(
                    "delete_project: stored path %s does not match the derived "
                    "project path; leaving the directory in place",
                    project.path,
                )
        was_active = project.is_active
        self.session.delete(project)
        self.session.commit()
        if was_active:
            # By name, not the fixed id: an org's default row has its own uuid, so
            # the constant resolves to None and deleting the active project left
            # the org with none active.
            general = self.get_project_by_name_or_none(GENERAL_PROJECT)
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
