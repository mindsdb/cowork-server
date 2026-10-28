"""Distribute canonical skills into per-project ``skills/`` folders.

Each enabled skill is symlinked from the canonical store
(``COWORK_SKILLS_DIR/<slug>``) into ``<projects_root>/<project>/skills/<slug>``.

That fan-out runs only where skills and projects stay unpartitioned
(``links_apply_to``). Where they are partitioned, nothing fans links out:
each in-process turn links its own skills folder from the asker's store
instead (``sync_turn_skill_links``).

Scoping rules:
  - ``enabled=false`` → no links anywhere.
  - ``metadata.projects`` is empty → link to **all** discovered projects (global skill).
  - ``metadata.projects`` lists specific projects → link only to those projects.

Projects inside ``project.root_dir`` are discovered by scanning it, and a
project pointed at a folder the user chose is read from its row, because the
scan cannot see it. A skill's ``metadata.projects`` entries are matched
against a project's **name**, which is its folder name for a scanned project
and only that: once a folder is adopted the two are different strings.

Directory links only. We prefer a real symlink; on Windows, where creating a
symlink needs admin rights or Developer Mode (otherwise ``os.symlink`` raises
``WinError 1314``), we fall back to a directory **junction**, which any user
can create. A single unlinkable skill is logged and skipped by
``reconcile_all`` rather than aborting server startup.
"""
from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from cowork.common.paths import dir_rmdir, dir_unlink, pinned_dir, safe_join_lexical
from cowork.common.settings import get_app_settings
from cowork.db.personal_resources import PersonalResource
from cowork.db.scoped import TenantScope, tenant_scope_required
from cowork.managed import managed_folder
from cowork.models.skill import Skill

logger = logging.getLogger(__name__)

_IS_WINDOWS = os.name == "nt"

if _IS_WINDOWS:
    import _winapi  # Windows-only stdlib; provides CreateJunction.


def skill_partitions_enabled() -> bool:
    """Whether this deployment keeps skills per org or per person.

    True in org tenancy, and on a deployment that trusts identity headers and
    lists skills in COWORK_PERSONAL_RESOURCES. There the unkeyed store belongs
    to the operator view alone, so nothing fans its skills out into projects,
    not even at boot.
    """
    settings = get_app_settings()
    return settings.tenancy_mode == "org" or (
        settings.trust_identity_headers
        and PersonalResource.SKILLS in settings.personal_resources
    )


def links_apply_to(scope: TenantScope | None) -> bool:
    """Whether a skill or project change under ``scope`` redistributes links.

    Every link points from the unkeyed skills root into a project folder found
    under the unkeyed projects root, so distribution belongs only to a scope
    that keeps both skills and projects unpartitioned, on a deployment that
    keeps skills unpartitioned: the desktop, and a trusted-identity deployment
    that leaves skills shared (its operator view, and a person whose projects
    are shared too). Anywhere else a link would name another person's or org's
    skill, or land in a project that is not the caller's; those turns link
    their own skills instead (sync_turn_skill_links). ``None`` is an unscoped
    boot or migration path.
    """
    if skill_partitions_enabled():
        return False
    if scope is None:
        return True
    return not (
        scope.partitioned_for(PersonalResource.SKILLS)
        or scope.partitioned_for(PersonalResource.PROJECTS)
    )


def _canon_root() -> Path:
    return Path(get_app_settings().skill.root_dir)


def _project_dirs() -> list[Path]:
    root = Path(get_app_settings().project.root_dir)
    if not root.exists():
        return []
    return [p for p in root.iterdir() if p.is_dir()]


@dataclass(frozen=True)
class _ProjectDir:
    """A project as skill distribution addresses it: its name and its folder."""

    name: str
    path: Path


def _external_project_dirs() -> dict[str, Path]:
    """Adopted-folder projects, keyed by row name.

    Symlink distribution is desktop-only (see ``SkillService``), so a
    local-scope read is the whole picture: there is no other tenant to scope
    to. Opening a session here keeps the module's callers unchanged — several
    hold only a ``TenantScope``.

    A failure degrades to the scan alone, which is the behaviour before
    adopted folders existed, so it is logged rather than raised.
    """
    if tenant_scope_required():
        # LOCAL_SCOPE below disables partition filtering, so this query would
        # read every org's or person's rows. A folder can only be adopted on an
        # unpartitioned deployment, so there is nothing here to find.
        return {}
    try:
        from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
        from cowork.db.session import get_open_session
        from cowork.services.projects import ProjectService

        with get_open_session() as raw:
            service = ProjectService(ScopedSession(raw, LOCAL_SCOPE))
            return {
                project.name: Path(project.path)
                for project in service.list_projects()
                if service.directory_is_external(project)
            }
    except Exception:
        logger.warning("Could not read adopted project folders", exc_info=True)
        return {}


def _all_projects() -> list[_ProjectDir]:
    """Every project skills can be distributed into, scanned and adopted."""
    projects = [_ProjectDir(name=path.name, path=path) for path in _project_dirs()]
    seen = {project.name for project in projects}
    for name, path in _external_project_dirs().items():
        if name not in seen:
            projects.append(_ProjectDir(name=name, path=path))
    return projects


def _select_project(project_dir: Path, project_name: str | None) -> _ProjectDir | None:
    """The project to link into, always taken from discovery.

    Never the caller's path. A request-controlled parent that shares a
    project's folder name would otherwise receive the links, so the name
    selects and discovery supplies the directory.

    An adopted folder's basename is not its name, and could even be another
    project's, so its caller has to pass `project_name`.
    """
    requested = project_dir.name if project_name is None else project_name
    safe_name = os.path.basename(requested)
    if (
        safe_name != requested
        or safe_name in {"", ".", ".."}
        or "\\" in safe_name
        or "\0" in safe_name
    ):
        raise ValueError("Project link target must be a direct-child name")
    return next(
        (
            candidate
            for candidate in _all_projects()
            if secrets.compare_digest(candidate.name, safe_name)
        ),
        None,
    )


def _is_dir_link(path: Path) -> bool:
    """True if *path* is a symlink or (on Windows) a directory junction.

    Junctions are reparse points that ``Path.is_symlink()`` reports as ``False``,
    so we additionally check the reparse tag on Windows.
    """
    try:
        if path.is_symlink():
            return True
    except OSError:
        return False
    if _IS_WINDOWS:
        try:
            return bool(getattr(os.lstat(path), "st_reparse_tag", 0))
        except OSError:
            return False
    return False


def _make_dir_link(link: Path, target: Path) -> None:
    """Create a directory link at *link* pointing to *target*.

    Prefers a real symlink. On Windows a symlink needs admin rights or
    Developer Mode (else ``WinError 1314``), so fall back to a directory
    junction — creatable by any user and equivalent for our read-only fan-out.
    """
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if not _IS_WINDOWS:
            raise
        # CreateJunction(target, junction); target must be absolute.
        _winapi.CreateJunction(os.path.abspath(target), str(link))


def _unlink_dir_link(link: Path) -> None:
    """Remove a symlink or junction without touching the target.

    On Windows a directory reparse point (symlink-to-dir or junction) is
    removed with ``rmdir``, not ``unlink``.
    """
    name = link.name
    safe_name = os.path.basename(name)
    if (
        safe_name != name
        or safe_name in {"", ".", ".."}
        or "\\" in safe_name
        or "\0" in safe_name
    ):
        raise ValueError("Skill link must be a direct-child name")
    with pinned_dir(link.parent, nofollow_base=True) as parent:
        if _IS_WINDOWS:
            try:
                dir_rmdir(parent, safe_name)
                return
            except OSError:
                pass
        dir_unlink(parent, safe_name)


def _ensure_symlink(link: Path, target: Path) -> None:
    if _is_dir_link(link):
        try:
            if link.resolve() == target.resolve():
                return
        except OSError:
            pass  # dangling link → recreate below
        _unlink_dir_link(link)
    elif link.exists():
        raise RuntimeError(f"{link} exists and is not a symlink; refusing to replace it.")
    link.parent.mkdir(parents=True, exist_ok=True)
    _make_dir_link(link, target)


def _remove_link(link: Path) -> None:
    if _is_dir_link(link):
        _unlink_dir_link(link)
    elif link.exists():
        raise RuntimeError(f"{link} exists and is not a symlink; refusing to remove it.")


def reconcile_skill_links(skill: Skill) -> None:
    """Make each project's ``skills/<slug>`` link match the skill's metadata."""
    canon = safe_join_lexical(_canon_root(), skill.name)
    all_projects = _all_projects()
    managed = managed_folder()
    if not skill.enabled or (managed is not None and skill.name in managed.skill_slugs()):
        # A slug the operator's managed folder holds is loaded from that folder,
        # so a same-named store skill must link into no project, or the agent
        # would recall the store copy over the managed one. Treated like a
        # disabled skill, which also removes a link left from before the
        # managed copy on the next fan-out.
        desired: set[str] = set()
    elif skill.projects:
        desired = set(skill.projects)
    else:
        desired = {p.name for p in all_projects}
    for project in all_projects:
        link = safe_join_lexical(project.path / "skills", skill.name)
        try:
            if project.name in desired and canon.exists():
                _ensure_symlink(link, canon)
            else:
                _remove_link(link)
        except Exception:
            # Per project, so one folder cannot abort the fan-out. An adopted
            # folder is the user's: it can already hold a real `skills/<slug>`
            # directory, or be read-only, and neither should fail the skill
            # write that triggered this.
            logger.warning(
                "Could not reconcile skill %r into project %r",
                skill.name,
                project.name,
                exc_info=True,
            )


def remove_skill_links(slug: str) -> None:
    """Drop the skill's link from every project (used on delete / rename)."""
    for project in _all_projects():
        try:
            _remove_link(safe_join_lexical(project.path / "skills", slug))
        except Exception:
            logger.warning(
                "Could not remove skill %r from project %r",
                slug,
                project.name,
                exc_info=True,
            )


def reconcile_project(
    project_dir: Path, skills: list[Skill], *, project_name: str | None = None
) -> None:
    """Link all applicable skills into a single newly-created project."""
    selected_project = _select_project(project_dir, project_name)
    if selected_project is None:
        return
    canon_root = _canon_root()
    managed = managed_folder()
    managed_slugs = frozenset() if managed is None else managed.skill_slugs()
    for skill in skills:
        if not skill.enabled:
            continue
        # A slug the managed folder holds is loaded from there; a same-named
        # store skill must never link into the new project.
        if skill.name in managed_slugs:
            continue
        if skill.projects and selected_project.name not in skill.projects:
            continue
        safe_skill_name = os.path.basename(skill.name)
        if (
            safe_skill_name != skill.name
            or safe_skill_name in {"", ".", ".."}
            or "\\" in safe_skill_name
            or "\0" in safe_skill_name
        ):
            raise ValueError("Skill link target must be a direct-child name")
        canon = safe_join_lexical(canon_root, safe_skill_name)
        if canon.exists():
            _ensure_symlink(
                safe_join_lexical(selected_project.path / "skills", safe_skill_name),
                canon,
            )


def sync_turn_skill_links(*, skills_dir: Path, wanted: dict[str, Path]) -> None:
    """Make ``skills_dir`` hold exactly one link per wanted skill, for one turn.

    ``wanted`` maps each skill's slug to its directory: in the asker's own
    store (select_turn_skills), or, for the folder of skills others share with
    the asker, in each creator's own store (shared_turn_skills). Every other
    link is removed, including one
    that points into another store under a wanted slug, and links left by an
    earlier fan-out from the unkeyed store; a link already pointing at its
    wanted directory is kept. A real directory is left alone, so a skill the
    agent wrote there is still relocated into a draft after the turn
    (task_objects.finalize_turn_skill_drafts) rather than deleted. One link
    that cannot be made or removed is logged and skipped, so the turn still
    starts with the rest.
    """
    skills_dir.mkdir(parents=True, exist_ok=True)
    for entry in sorted(skills_dir.iterdir()):
        if not _is_dir_link(entry):
            continue
        target = wanted.get(entry.name)
        try:
            if target is not None and entry.resolve() == target.resolve():
                continue
        except OSError:
            pass  # dangling: removed below, relinked if wanted
        try:
            _remove_link(entry)
        except Exception:
            logger.warning("Could not remove stale skill link %r", entry.name, exc_info=True)
    for slug, target in sorted(wanted.items()):
        try:
            _ensure_symlink(safe_join_lexical(skills_dir, slug), target)
        except Exception:
            logger.warning("Could not link skill %r for this turn", slug, exc_info=True)


def reconcile_all(skills: list[Skill]) -> None:
    """Full reconcile of all skills across all projects (boot / seed).

    Best-effort: a failure to (re)link one skill is logged and skipped rather
    than raised, so a single unlinkable skill can never abort server startup.
    Per-project skill links are a convenience; the server must still boot
    without them.
    """
    for skill in skills:
        try:
            reconcile_skill_links(skill)
        except Exception:
            logger.warning(
                "Failed to reconcile skill links for %r; skipping",
                skill.name,
                exc_info=True,
            )
