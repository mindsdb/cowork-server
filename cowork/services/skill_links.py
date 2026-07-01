"""Distribute canonical skills into per-project ``skills/`` folders.

Each enabled skill is symlinked from the canonical store
(``COWORK_SKILLS_DIR/<slug>``) into ``<projects_root>/<project>/skills/<slug>``.

Scoping rules:
  - ``enabled=false`` → no links anywhere.
  - ``metadata.projects`` is empty → link to **all** discovered projects (global skill).
  - ``metadata.projects`` lists specific projects → link only to those projects.

Projects are discovered by scanning ``project.root_dir`` (no DB), so a skill's
``metadata.projects`` entries are matched against project **folder names**.

Symlinks only — if one can't be created we raise rather than fall back to a
copy, so a misconfigured filesystem fails loudly.
"""
from __future__ import annotations

from pathlib import Path

from cowork.common.settings import get_app_settings
from cowork.models.skill import Skill


def _canon_root() -> Path:
    return Path(get_app_settings().skill.root_dir)


def _project_dirs() -> list[Path]:
    root = Path(get_app_settings().project.root_dir)
    if not root.exists():
        return []
    return [p for p in root.iterdir() if p.is_dir()]


def _ensure_symlink(link: Path, target: Path) -> None:
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"{link} exists and is not a symlink; refusing to replace it.")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)


def _remove_link(link: Path) -> None:
    if link.is_symlink():
        link.unlink()
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


def reconcile_all(skills: list[Skill]) -> None:
    """Full reconcile of all skills across all projects (boot / seed)."""
    for skill in skills:
        reconcile_skill_links(skill)
