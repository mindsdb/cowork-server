from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import logging
import os
import re
from pathlib import Path
from uuid import UUID, uuid4

import sqlalchemy as sa

from cowork.common.paths import (
    PinnedDir,
    dir_mkdir,
    dir_rename,
    dir_rmtree,
    pinned_dir,
    safe_join,
)
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession, scoped_storage_root, unsafe_unscoped_session
from cowork.models.project import Project

if TYPE_CHECKING:
    from cowork.services.skills import ProjectReferenceRewrite

logger = logging.getLogger(__name__)


GENERAL_PROJECT = "general"
GENERAL_PROJECT_ID = UUID("00000000-0000-0000-0000-000000000001")

_NAME_DISALLOWED = re.compile(r"[^A-Za-z0-9._-]+")
_NAME_HYPHEN_RUNS = re.compile(r"-{2,}")
_WIN_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}
_NAME_MAX_LEN = 48
_NAME_FALLBACK = "untitled-project"
_DISPLAY_NAME_MAX_LEN = 255


def display_label(project: "Project") -> str:
    """The name a person reads for this project.

    The ONE place that answers "which field do we show". `display_name` holds
    what the user typed; it is NULL on every row created before that column
    existed, and those fall back to the slug so they render exactly as they
    always have (ENG-1676).

    Never use this to address anything. `name` is the directory, the URL
    segment and the lookup key, and it stays that way.
    """
    # getattr, not attribute access: the harness passes project-shaped objects
    # (SimpleNamespace in tests, and any caller predating the column), and a
    # label helper must not be the thing that raises.
    return getattr(project, "display_name", None) or project.name


class ProjectNotFoundError(ValueError):
    """The requested project is absent from the caller's scoped view."""


class ProjectPathNotAllowedError(ValueError):
    """A caller chose a project folder on a deployment that does not allow one."""


@dataclass
class ProjectRenameStage:
    """Filesystem changes held open until the project transaction commits."""

    project_id: UUID
    old_name: str
    new_name: str
    old_path: Path
    new_path: Path
    skill_rewrites: list["ProjectReferenceRewrite"]
    directory_moved: bool = False
    skill_rewrites_applied: bool = False


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
        # winner's row. Core insert, not session.add, so created_by is written
        # explicitly rather than stamped by the flush hook: the member who first
        # reaches the org provisions General and becomes its recorded creator,
        # which is what makes its instructions editable without an admin. Rename
        # and delete stay refused for every role, creator included.
        raw = unsafe_unscoped_session(self.session)  # bootstrap op, not query path
        try:
            self._execute_general_insert(raw, path, scope.org_id, scope.user_id)
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

    def _execute_general_insert(
        self, raw, path: Path, org_id: str | None, created_by: str | None
    ) -> None:
        stmt = self._insert_stmt(raw).from_select(
            ["id", "name", "path", "is_active", "org_id", "created_by"],
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
                sa.literal(created_by),
            ).where(
                ~sa.exists().where(
                    Project.name == GENERAL_PROJECT,  # type: ignore[arg-type]
                    Project.org_id == org_id,  # type: ignore[arg-type]
                )
            ),
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
            Path(get_app_settings().project.root_dir),
            self.session.scope,
            store="projects",
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

    def _trusted_project_root(self, path: Path) -> tuple[Path, str]:
        """Return the trusted direct-child root and name for ``path``.

        Org rows may still point at the old unkeyed projects root. Both that
        root and the current org root are server configuration, but neither may
        be resolved here: following a swapped root symlink would turn the
        containment check into the cross-tenant traversal it is meant to stop.
        """
        absolute = Path(os.path.abspath(path))
        path_parent = Path(os.path.abspath(absolute.parent))
        canonical_parent = Path(os.path.realpath(path_parent.parent)) / path_parent.name
        roots = (self._root_dir(), Path(get_app_settings().project.root_dir))
        for candidate in roots:
            configured = Path(os.path.abspath(candidate))
            # Resolve stable ancestors (for macOS /var -> /private/var), but
            # append the agent-writable projects-root component without
            # following it. ``pinned_dir(..., nofollow_base=True)`` can then
            # reject a swapped root symlink at use time.
            root = Path(os.path.realpath(configured.parent)) / configured.name
            if canonical_parent == root and absolute.name not in {"", ".", ".."}:
                return root, absolute.name
        raise ValueError(f"{path} is not a direct child of a trusted projects root")

    def _rename_in_root(self, old: Path, new: Path) -> None:
        """Rename between trusted project roots without re-walking either path."""
        source_root, source_name = self._trusted_project_root(old)
        destination_root, destination_name = self._trusted_project_root(new)
        if source_root == destination_root:
            with pinned_dir(source_root, nofollow_base=True) as root:
                dir_rename(root, source_name, root, destination_name)
            return
        with pinned_dir(
            source_root,
            nofollow_base=True,
        ) as source, pinned_dir(
            destination_root,
            create=True,
            nofollow_base=True,
        ) as destination:
            dir_rename(source, source_name, destination, destination_name)

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
        if (
            candidate.is_absolute()
            or len(candidate.parts) != 1
            or candidate.name in {"", ".", ".."}
        ):
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
            p.name
            for p in self.session.exec(self.session.select(Project)).all()
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

    def _display_base(self, raw: str | None, fallback: str) -> str:
        """The label to start from, before de-duplication.

        Whitespace-only input does NOT become NULL: that would quietly expose
        the slug on a project the user just created, and NULL is reserved for
        "predates the column". Such input resolves through the existing
        empty-name fallback instead, and we store the label that produced.
        """
        cleaned = (raw or "").strip()
        return cleaned[:_DISPLAY_NAME_MAX_LEN] if cleaned else fallback

    def _unique_display_name(self, base: str, *, exclude_id: UUID | None = None) -> str:
        """`base`, or `base 2` / `base 3` when another project already reads that way.

        Compared against the RESOLVED label of every other project, so a new
        project typed literally `My-Project` still collides with an existing
        slug-labelled `My-Project`. Space-separated, unlike `_unique_name`'s
        hyphen: this suffix is read by humans, not used as a path.

        Known limitation, accepted. This is a read-then-write with no unique
        constraint behind it, so two concurrent creates of the same name can
        both pick the unsuffixed label. `_unique_name` has the same shape but
        `mkdir` settles it; there is no equivalent backstop for a label, and
        adding one means a unique index and a failure path on a column that
        addresses nothing. The cost of losing the race is two rows reading the
        same way in the sidebar -- which is where every non-Latin project
        already was before this ticket -- and renaming either one fixes it.
        """
        taken = {
            (p.display_name or p.name)
            for p in self.session.exec(self.session.select(Project)).all()
            if p.id != exclude_id
        }
        if base not in taken:
            return base
        i = 2
        while True:
            # Leave room for the suffix rather than overflowing the column.
            candidate = f"{base[: _DISPLAY_NAME_MAX_LEN - len(str(i)) - 1]} {i}"
            if candidate not in taken:
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
            raise ProjectNotFoundError("Project not found")
        return project

    def get_project_by_name(self, name: str) -> Project:
        project = self.session.exec(
            self.session.select(Project).where(Project.name == name)
        ).first()
        if project is None:
            raise ProjectNotFoundError("Project not found")
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
        """Resolve a name, provisioning the org's default on a `general` miss.

        `general` is lazily created per org, so a plain by-name lookup can 404 before the
        org's first `GET /projects/` provisions its row. Every other name stays exact.
        The single home for this self-heal: callers that must not auto-provision stay on
        the plain primitives, and it can't fold into those (`ensure_general_for_scope`
        calls `get_project_by_name_or_none`, so self-healing there would recurse).
        """
        if name == GENERAL_PROJECT:
            return self.ensure_general_for_scope()
        return self.get_project_by_name_or_none(name)

    def get_or_provision_by_name(self, name: str) -> Project:
        """`get_or_provision_by_name_or_none`, raising when the name is genuinely missing."""
        project = self.get_or_provision_by_name_or_none(name)
        if project is None:
            raise ProjectNotFoundError("Project not found")
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

    def directory_is_external(self, project: Project) -> bool:
        """Whether this project's directory sits outside the projects root.

        True only for a folder the user chose. Artifact discovery, skill link
        distribution and rename all find projects by scanning that root, so
        they need this to know when to consult the row instead.

        Always False in org mode. A folder can only be adopted on a local
        deployment, so a path outside the org-keyed root there is a stale or
        pre-org-keyed row -- `_repoint_if_stale`'s job, not a chosen folder.
        """
        if get_app_settings().tenancy_mode == "org" or self.session.scope.org_mode:
            return False
        try:
            resolved = Path(project.path).resolve(strict=False)
        except (OSError, RuntimeError):
            return False
        return not self._within_root(resolved)

    def _within_root(self, path: Path) -> bool:
        try:
            root = self._root_dir().resolve(strict=False)
        except (OSError, RuntimeError):
            return False
        return path == root or root in path.parents

    def _path_in_use(self, path: Path) -> bool:
        for project in self.session.exec(self.session.select(Project)).all():
            try:
                if Path(project.path).resolve(strict=False) == path:
                    return True
            except (OSError, RuntimeError):
                continue
        return False

    def _adopt_project_dir(self, base: str, path: Path) -> tuple[str, Path]:
        """Take a folder the user already has, creating nothing.

        The tenancy refusal comes before any filesystem access on purpose: an
        org deployment does not run on the caller's machine, so statting a
        path it chose would answer whether that server path exists.
        """
        if get_app_settings().tenancy_mode == "org":
            raise ProjectPathNotAllowedError(
                "Choosing a project folder is not available on this deployment"
            )
        try:
            resolved = path.expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("Choose an existing local folder") from exc
        if not resolved.is_dir():
            raise ValueError("Choose an existing local folder")
        # Inside the root a chosen folder can equal _project_path() for another
        # row, and delete_project would re-derive a match and rmtree it.
        if self._within_root(resolved):
            raise ValueError("Choose a folder outside the Cowork projects directory")
        if self._path_in_use(resolved):
            raise ValueError("Another project already uses this folder")
        # `_unique_name` still applies: `name` is the lookup key, the URL
        # segment and the basename, and nothing else keeps it unique.
        return self._unique_name(base), resolved

    def create_project(
        self,
        name: str,
        *,
        project_id: UUID | None = None,
        path: Path | None = None,
    ) -> Project:
        sanitized = self._sanitize_name(name)
        # `general` belongs to the system row. A member creating it first would own
        # an undeletable, user-attributed project AND block the real default,
        # which is looked up by name.
        if sanitized == GENERAL_PROJECT:
            sanitized = f"{GENERAL_PROJECT}-2"
        if path is not None:
            final_name, project_dir = self._adopt_project_dir(sanitized, path)
        else:
            # No exist_ok: `_unique_name` is a read-then-write with no unique
            # constraint behind it, so two concurrent creates can pick the same name.
            # Letting mkdir fail keeps them from sharing one directory (where deleting
            # either would rmtree the other's files) and stops a leftover directory
            # being adopted with stale contents. On collision, take the next name.
            final_name, project_dir = self._allocate_project_dir(sanitized)
        # self._scaffold(project_dir)
        # The literal input, kept verbatim; `final_name` stays the slug. A new
        # project always gets an explicit display_name -- NULL means "predates
        # the column", never "the user typed nothing" (ENG-1676).
        display = self._unique_display_name(self._display_base(name, final_name))
        project = (
            Project(
                id=project_id,
                name=final_name,
                display_name=display,
                path=str(project_dir),
                is_active=False,
            )
            if project_id is not None
            else Project(
                name=final_name,
                display_name=display,
                path=str(project_dir),
                is_active=False,
            )
        )
        self.session.add(project)
        self.session.commit()

        # Skill symlink distribution is desktop-only (see SkillService).
        if not self.session.scope.org_mode:
            from cowork.services.skill_links import reconcile_project
            from cowork.services.skills import SkillService

            reconcile_project(
                project_dir,
                SkillService(self.session.scope).list_skills(),
                project_name=final_name,
            )

        return project

    def resolve_display_label(self, project: Project, name: str) -> str:
        """The label this project would read as after being renamed to `name`.

        Exposed so the endpoint can tell a label-only rename from a true no-op
        without reimplementing the resolution. `_sanitize_name` is lossy for
        exactly the input ENG-1676 is about, so a rename can leave the slug
        untouched and still change what every member sees.
        """
        return self._unique_display_name(
            self._display_base(name, project.name), exclude_id=project.id
        )

    def resolve_update_name(self, project: Project, name: str) -> str:
        """Return the canonical collision-resolved name for an update."""
        sanitized = self._sanitize_name(name)
        if sanitized == GENERAL_PROJECT and project.name != GENERAL_PROJECT:
            sanitized = f"{GENERAL_PROJECT}-2"
        return self._unique_name(sanitized, exclude=project.name)

    def _stage_project_rename(
        self,
        project: Project,
        final_name: str,
        *,
        skill_rewrites: list["ProjectReferenceRewrite"] | None = None,
    ) -> ProjectRenameStage:
        from cowork.services.skills import SkillService

        old_name = project.name
        old_path = Path(project.path)
        new_path = self._project_path(final_name)
        skill_service = SkillService(self.session.scope)
        rewrites = (
            skill_rewrites
            if skill_rewrites is not None
            else skill_service.prepare_project_reference_rewrites(
                old_name,
                final_name,
            )
        )
        stage = ProjectRenameStage(
            project_id=project.id,
            old_name=old_name,
            new_name=final_name,
            old_path=old_path,
            new_path=new_path,
            skill_rewrites=rewrites,
        )
        try:
            if old_path.exists():
                # The org's root may not exist yet for a legacy row. The
                # destination remains pinned inside the scoped root.
                self._rename_in_root(old_path, new_path)
                stage.directory_moved = True
            skill_service.apply_project_reference_rewrites(rewrites)
            stage.skill_rewrites_applied = True
            project.name = final_name
            project.path = str(new_path)
            self.session.add(project)
        except Exception:
            self.session.rollback()
            try:
                self.rollback_project_rename(stage)
            except Exception:
                logger.exception(
                    "Could not fully restore project %s after rename staging failed",
                    project.id,
                )
            raise
        return stage

    def rollback_project_rename(self, stage: ProjectRenameStage) -> None:
        """Restore every filesystem component of an uncommitted rename."""
        first_error: Exception | None = None
        if stage.skill_rewrites_applied:
            from cowork.services.skills import SkillService

            try:
                SkillService(self.session.scope).restore_project_reference_rewrites(
                    stage.skill_rewrites
                )
                stage.skill_rewrites_applied = False
            except Exception as exc:
                first_error = exc
                logger.exception(
                    "Could not restore skill references for project %s",
                    stage.project_id,
                )
        if stage.directory_moved:
            try:
                self._rename_in_root(stage.new_path, stage.old_path)
                stage.directory_moved = False
            except Exception as exc:
                first_error = first_error or exc
                logger.exception(
                    "Could not restore the directory for project %s",
                    stage.project_id,
                )
        if first_error is not None:
            raise RuntimeError(
                "Project rename compensation was incomplete"
            ) from first_error

    def _apply_active_selection(
        self,
        project: Project,
        is_active: bool | None,
    ) -> None:
        if is_active is None:
            return
        if is_active:
            for other in self.session.exec(self.session.select(Project)).all():
                if other.id != project.id and other.is_active:
                    other.is_active = False
                    self.session.add(other)
        project.is_active = is_active

    def stage_project_update(
        self,
        project_id: UUID,
        *,
        resolved_name: str | None,
        is_active: bool | None,
        # No default on purpose. Both endpoint branches call this, and only one
        # of them passed a label -- so an org rename moved the directory and
        # rewrote every skill reference while `display_name` stayed frozen at
        # the old name. Requiring it makes that omission a TypeError instead of
        # a stale label, and matches the two parameters above.
        display_label: str | None,
        skill_rewrites: list["ProjectReferenceRewrite"] | None = None,
    ) -> tuple[Project, ProjectRenameStage | None]:
        """Stage a project update without committing its DB transaction.

        `display_label` is the raw name the user typed, not the slug. It is a
        separate parameter because `resolved_name` has already been through
        `_sanitize_name`, which is lossy for exactly the input ENG-1676 is
        about: every Cyrillic name sanitizes to `untitled-project`.
        """
        project = self.session.get(Project, project_id)
        if project is None:
            raise ProjectNotFoundError("Project not found")
        stage = None
        try:
            if resolved_name is not None and resolved_name != project.name:
                if project.name == GENERAL_PROJECT:
                    raise ValueError("Cannot rename the General project")
                if self.directory_is_external(project):
                    # Refused here rather than inside the move: _rename_in_root
                    # would raise "not a direct child of a trusted projects
                    # root", which is true and unusable as a message.
                    raise ValueError(
                        "This project points at a folder you chose, so it cannot "
                        "be renamed. Rename the folder instead."
                    )
                stage = self._stage_project_rename(
                    project,
                    resolved_name,
                    skill_rewrites=skill_rewrites,
                )
            if display_label is not None:
                # Outside the rename branch on purpose: two different names can
                # sanitize to the same slug -- every pair of Cyrillic names
                # does -- and the user still renamed the project, so the label
                # must follow even when nothing moved on disk (ENG-1676).
                project.display_name = self._unique_display_name(
                    self._display_base(display_label, project.name),
                    exclude_id=project.id,
                )
            self._apply_active_selection(project, is_active)
            self.session.add(project)
            self.session.flush()
        except Exception:
            self.session.rollback()
            if stage is not None:
                try:
                    self.rollback_project_rename(stage)
                except Exception:
                    logger.exception(
                        "Could not fully restore project %s after update staging failed",
                        project_id,
                    )
            raise
        return project, stage

    def commit_staged_project_update(
        self,
        project: Project,
        stage: ProjectRenameStage | None,
    ) -> Project:
        """Commit a staged update and compensate its filesystem on failure."""
        try:
            self.session.commit()
        except Exception:
            self.session.rollback()
            if stage is not None:
                try:
                    self.rollback_project_rename(stage)
                except Exception:
                    logger.exception(
                        "Could not fully restore project %s after commit failed",
                        project.id,
                    )
            raise
        if stage is not None and not self.session.scope.org_mode:
            # These links are derived desktop state, so a reconciliation failure
            # is logged after the canonical project/skill commit rather than
            # turning a successful rename into a false API failure.
            try:
                from cowork.services.skill_links import reconcile_project
                from cowork.services.skills import SkillService

                skill_service = SkillService(self.session.scope)
                skill_service.finalize_project_reference_rewrites(stage.skill_rewrites)
                reconcile_project(
                    stage.new_path,
                    skill_service.list_skills(),
                    project_name=stage.new_name,
                )
            except Exception:
                logger.exception(
                    "Could not reconcile desktop links for renamed project %s",
                    project.id,
                )
        return project

    def update_project(
        self,
        project_id: UUID,
        name: str | None = None,
        is_active: bool | None = None,
    ) -> Project:
        project = self.session.get(Project, project_id)
        if project is None:
            raise ProjectNotFoundError("Project not found")
        resolved_name = (
            self.resolve_update_name(project, name) if name is not None else None
        )
        updated, stage = self.stage_project_update(
            project_id,
            resolved_name=resolved_name,
            is_active=is_active,
            display_label=name,
        )
        return self.commit_staged_project_update(updated, stage)

    def delete_project(
        self,
        project_id: UUID,
        *,
        before_commit: Callable[[], None] | None = None,
    ) -> bool:
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
                self.session.select(Conversation).where(
                    Conversation.project_id == project_id
                )
            ).all()
        ]
        conversation_stages = []
        try:
            for cid in conv_ids:
                # Owner-agnostic: a project is org-shared and may hold several
                # members' conversations. Fetch org-scoped (session.get validates
                # org, not owner) and stage the row directly. No attachment or
                # workspace bytes are removed until the project/audit commit.
                conv = self.session.get(Conversation, cid)
                if conv is None:
                    continue
                conversation_stages.append(
                    # A project delete is the org-wide cascade, so it is the one
                    # caller allowed to drop every member's attachment rows.
                    conv_svc.stage_delete_conversation_row(
                        conv, include_org_attachments=True
                    )
                )
        except Exception:
            # Atomicity is stricter than the old skip-on-error behavior. A
            # failed child stage rolls back every earlier child and aborts the
            # project delete, so no surviving conversation becomes a ghost.
            self.session.rollback()
            raise
        # rmtree only a path re-derived from the sanitized name inside the
        # org-keyed root (same rebuild-and-compare as ensure_dir_exists). The
        # stored string can be stale (pre-org-keying) or tampered, and deleting
        # it verbatim would rmtree an arbitrary directory. A mismatched dir is
        # left behind instead — an orphaned directory beats a wrong-target rm.
        path = Path(project.path)
        staged_path: Path | None = None
        try:
            safe = self._project_path(project.name)
        except ValueError:
            safe = None
        if path.exists():
            if safe is not None and safe.resolve() == path.resolve():
                staged_path = self._root_dir() / (f".delete-{project.id}-{uuid4().hex}")
                try:
                    self._rename_in_root(path, staged_path)
                except Exception:
                    self.session.rollback()
                    raise
            elif self.directory_is_external(project):
                # The expected outcome for a folder the user chose, not an
                # anomaly: it is theirs, so the project row goes and the
                # directory stays. `directoryIsExternal` tells the client to
                # say so before it asks for confirmation.
                logger.info(
                    "delete_project: %r points at a folder outside the projects "
                    "root; leaving it in place",
                    project.name,
                )
            else:
                logger.warning(
                    "delete_project: stored path %s does not match the derived "
                    "project path; leaving the directory in place",
                    project.path,
                )
        try:
            if project.is_active:
                # By name, not the fixed id: an org's default row has its own
                # uuid. Keep this selection change in the project-delete commit.
                general = self.get_project_by_name_or_none(GENERAL_PROJECT)
                if general is not None and not general.is_active:
                    general.is_active = True
                    self.session.add(general)
            if before_commit is not None:
                before_commit()
            self.session.delete(project)
            self.session.commit()
        except Exception:
            self.session.rollback()
            if staged_path is not None and staged_path.exists():
                try:
                    self._rename_in_root(staged_path, path)
                except Exception:
                    logger.exception(
                        "Could not restore project directory after delete failed: %s",
                        project_id,
                    )
            raise
        for conversation_stage in conversation_stages:
            conv_svc.finalize_staged_conversation_delete(
                conversation_stage,
                cleanup_project_files=staged_path is None,
            )
        if staged_path is not None and staged_path.exists():
            try:
                self._rmtree_in_root(staged_path)
            except Exception:
                # The row and required audit already committed together. A
                # hidden tombstone is safe to reap later and must not turn the
                # successful delete into a retry that records another event.
                logger.exception(
                    "Could not finalize staged project directory deletion: %s",
                    staged_path,
                )
        return True

    def get_active_project(self) -> Project:
        project = self.session.exec(
            self.session.select(Project).where(Project.is_active)
        ).first()
        if project is None:
            raise ValueError("No active project")
        return project
