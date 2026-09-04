"""Distribute canonical skills into per-project ``skills/`` folders.

Each enabled skill is symlinked from the canonical store
(``COWORK_SKILLS_DIR/<slug>``) into ``<projects_root>/<project>/skills/<slug>``.

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
from cowork.models.skill import Skill

logger = logging.getLogger(__name__)

_IS_WINDOWS = os.name == "nt"

if _IS_WINDOWS:
    import _winapi  # Windows-only stdlib; provides CreateJunction.


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
    if not skill.enabled:
        desired: set[str] = set()
    elif skill.projects:
        desired = set(skill.projects)
    else:
        desired = {p.name for p in all_projects}
    for project in all_projects:
        link = safe_join_lexical(project.path / "skills", skill.name)
        if project.name in desired and canon.exists():
            _ensure_symlink(link, canon)
        else:
            _remove_link(link)


def remove_skill_links(slug: str) -> None:
    """Drop the skill's link from every project (used on delete / rename)."""
    for project in _all_projects():
        _remove_link(safe_join_lexical(project.path / "skills", slug))


def reconcile_project(
    project_dir: Path, skills: list[Skill], *, project_name: str | None = None
) -> None:
    """Link all applicable skills into a single newly-created project."""
    selected_project = _select_project(project_dir, project_name)
    if selected_project is None:
        return
    canon_root = _canon_root()
    for skill in skills:
        if not skill.enabled:
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
