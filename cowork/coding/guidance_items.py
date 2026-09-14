from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

_INSTRUCTION_FILES = {"agents.md", "claude.md", "instructions.md"}
_WORKFLOW_SUFFIXES = {".yml", ".yaml"}
_IGNORED_DIRECTORIES = {".git", ".next", ".venv", "__pycache__", "build", "dist", "node_modules", "venv"}
_FRONTMATTER_DESCRIPTION = re.compile(r"^description:\s*['\"]?(.*?)['\"]?\s*$", re.MULTILINE)


@dataclass(frozen=True)
class GuidanceItemSpec:
    kind: str
    name: str
    path: str
    description: str = ""


def discover_guidance_items(cache: Path, *, limit: int = 250) -> list[GuidanceItemSpec]:
    """Discover portable agent guidance without following repository symlinks."""
    items: list[GuidanceItemSpec] = []
    for directory, directories, files in os.walk(cache, followlinks=False):
        current = Path(directory)
        directories[:] = sorted(
            name
            for name in directories
            if name.casefold() not in _IGNORED_DIRECTORIES
            and not (current / name).is_symlink()
        )
        for filename in sorted(files):
            path = current / filename
            if path.is_symlink():
                continue
            relative_path = path.relative_to(cache)
            relative = relative_path.as_posix()
            lower_name = path.name.casefold()
            if lower_name == "skill.md":
                items.append(
                    GuidanceItemSpec(
                        kind="skill",
                        name=path.parent.name,
                        path=relative,
                        description=_skill_description(path),
                    )
                )
            elif lower_name in _INSTRUCTION_FILES:
                items.append(GuidanceItemSpec(kind="instructions", name=path.name, path=relative))
            elif _is_github_workflow(relative_path):
                items.append(GuidanceItemSpec(kind="workflow", name=path.stem, path=relative))
            if len(items) >= limit:
                return items
    return items


def read_guidance_text(cache: Path, relative: str, *, limit: int) -> str:
    """Read a discovered item while preserving the managed-cache boundary."""
    root = cache.resolve()
    candidate = root / relative
    if candidate.is_symlink():
        return ""
    path = candidate.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return ""
    try:
        with path.open("rb") as stream:
            return stream.read(limit).decode("utf-8", errors="ignore")
    except (OSError, UnicodeError):
        return ""


def _skill_description(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            text = stream.read(8_000).decode("utf-8", errors="ignore")
        match = _FRONTMATTER_DESCRIPTION.search(text)
        return match.group(1).strip() if match else ""
    except (OSError, UnicodeError):
        return ""


def _is_github_workflow(path: Path) -> bool:
    parts = tuple(part.casefold() for part in path.parts)
    return (
        len(parts) == 3
        and parts[:2] == (".github", "workflows")
        and path.suffix.casefold() in _WORKFLOW_SUFFIXES
    )
