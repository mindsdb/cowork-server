"""Distribute canonical skills into per-project ``skills/`` folders.

Each enabled skill is symlinked from the canonical store
(``COWORK_SKILLS_DIR/<slug>``) into ``<projects_root>/<project>/skills/<slug>``.

Scoping rules:
  - ``enabled=false`` → no links anywhere.
  - ``metadata.projects`` is empty → link to **all** discovered projects (global skill).
  - ``metadata.projects`` lists specific projects → link only to those projects.

Projects are discovered by scanning ``project.root_dir`` (no DB), so a skill's
``metadata.projects`` entries are matched against project **folder names**.

Directory links only. We prefer a real symlink; on Windows, where creating a
symlink needs admin rights or Developer Mode (otherwise ``os.symlink`` raises
``WinError 1314``), we fall back to a directory **junction**, which any user
can create. A single unlinkable skill is logged and skipped by
``reconcile_all`` rather than aborting server startup.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

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
    if _IS_WINDOWS:
        try:
            os.rmdir(link)
            return
        except OSError:
            pass
    link.unlink()


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
    canon = _canon_root() / skill.name
    all_projects = _project_dirs()
    if not skill.enabled:
        desired: set[str] = set()
    elif skill.projects:
        desired = set(skill.projects)
    else:
        desired = {p.name for p in all_projects}
    for project_dir in all_projects:
        link = project_dir / "skills" / skill.name
        if project_dir.name in desired and canon.exists():
            _ensure_symlink(link, canon)
        else:
            _remove_link(link)


def remove_skill_links(slug: str) -> None:
    """Drop the skill's link from every project (used on delete / rename)."""
    for project_dir in _project_dirs():
        _remove_link(project_dir / "skills" / slug)


def reconcile_project(project_dir: Path, skills: list[Skill]) -> None:
    """Link all applicable skills into a single newly-created project."""
    canon_root = _canon_root()
    for skill in skills:
        if not skill.enabled:
            continue
        if skill.projects and project_dir.name not in skill.projects:
            continue
        canon = canon_root / skill.name
        if canon.exists():
            _ensure_symlink(project_dir / "skills" / skill.name, canon)


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
