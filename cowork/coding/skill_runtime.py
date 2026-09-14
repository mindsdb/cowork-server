from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from anton.core.tools.skill_format import normalize_name

from cowork.coding.contracts import ResolvedSkill
from cowork.coding.guidance_items import discover_guidance_items, read_guidance_text
from cowork.coding.project_models import CodeProject
from cowork.coding.skill_library import SkillLibraryService
from cowork.coding.skill_models import SkillResolution
from cowork.services.skills import CodeSkillService

_MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024
_MAX_INSTRUCTION_BYTES = 120_000


class SkillRuntimeResolver:
    """Resolve project and organisation skills into an immutable agent bundle."""

    def __init__(self, library: SkillLibraryService) -> None:
        self.library = library
        self.snapshots = library.root / "snapshots"
        self.snapshots.mkdir(parents=True, exist_ok=True)

    def resolve(
        self,
        session_id: str,
        project: CodeProject | None,
        code_skills: CodeSkillService | None = None,
    ) -> SkillResolution:
        target = self.snapshots / session_id
        staging = self.snapshots / f".{session_id}.next"
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(target, ignore_errors=True)
        staging.mkdir(parents=True)

        items: list[ResolvedSkill] = []
        instruction_sections: list[str] = []
        instruction_bytes = 0
        used_names: set[str] = set()
        try:
            if project:
                for binding in project.skill_sources:
                    source, cache = self.library.cache_for(binding.source_id)
                    selected = set(binding.enabled_paths)
                    for spec in discover_guidance_items(cache):
                        if spec.path not in selected:
                            continue
                        source_file = cache / spec.path
                        content = ""
                        if spec.kind == "skill":
                            digest = self._content_hash(source_file.parent)
                        else:
                            content = read_guidance_text(cache, spec.path, limit=48_000)
                            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                        item_id = f"{source.id}:{spec.path}"
                        items.append(
                            ResolvedSkill(
                                id=item_id,
                                kind=spec.kind,
                                name=spec.name,
                                description=spec.description,
                                origin="team",
                                source_id=source.id,
                                source_name=source.name,
                                source_path=spec.path,
                                version=source.applied_revision,
                                content_hash=digest,
                            )
                        )
                        if spec.kind == "skill":
                            self._copy_skill(source_file.parent, staging, spec.name, item_id, used_names)
                        else:
                            if content:
                                section = (
                                    f"## {spec.kind.title()}: {spec.name}\n"
                                    f"Source: {source.name}@{source.applied_revision[:12]} · {spec.path}\n{content}"
                                )
                                section_bytes = len(section.encode("utf-8"))
                                if instruction_bytes + section_bytes > _MAX_INSTRUCTION_BYTES:
                                    raise ValueError(
                                        "The selected shared instructions are too large for one coding task"
                                    )
                                instruction_sections.append(section)
                                instruction_bytes += section_bytes

            if code_skills:
                code_skills.ensure_builtin_skills()
                project_keys = {project.id, project.name} if project else set()
                builtin_names = code_skills.builtin_skill_names()
                for skill in code_skills.list_skills():
                    if not skill.enabled:
                        continue
                    if skill.projects and not project_keys.intersection(skill.projects):
                        continue
                    key = skill.name.casefold()
                    if key in used_names:
                        continue
                    source_dir = code_skills.root / skill.name
                    item_id = f"personal:{skill.name}"
                    origin = "built_in" if skill.name in builtin_names else "personal"
                    items.append(
                        ResolvedSkill(
                            id=item_id,
                            kind="skill",
                            name=skill.display_name,
                            description=skill.description,
                            origin=origin,
                            source_name="MindsHub" if origin == "built_in" else "Yours",
                            source_path=skill.name,
                            version=skill.updated_at.isoformat() if skill.updated_at else None,
                            content_hash=self._content_hash(source_dir),
                        )
                    )
                    self._copy_skill(source_dir, staging, skill.name, item_id, used_names)

            staging.rename(target)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(target, ignore_errors=True)
            raise

        team_count = sum(item.origin == "team" for item in items)
        skill_count = sum(item.kind == "skill" for item in items)
        summary = None
        if items:
            summary = f"{skill_count} skill{'s' if skill_count != 1 else ''}"
            if team_count:
                summary += f" · {team_count} team item{'s' if team_count != 1 else ''} version-pinned"
        instructions = ""
        if instruction_sections:
            instructions = "\n\n".join([
                "# MindsHub shared engineering guidance",
                "These project-selected items are pinned to the task's creation-time revisions.",
                *instruction_sections,
            ])
        roots = [str(target)] if any(item.kind == "skill" for item in items) else []
        return SkillResolution(items=items, roots=roots, developer_instructions=instructions, summary=summary)

    def clone(self, source_session_id: str, target_session_id: str) -> list[str]:
        source = self.snapshots / source_session_id
        target = self.snapshots / target_session_id
        if not source.is_dir():
            return []
        shutil.copytree(source, target)
        return [str(target)]

    def cleanup(self, session_id: str) -> None:
        shutil.rmtree(self.snapshots / session_id, ignore_errors=True)

    def _copy_skill(
        self,
        source: Path,
        target_root: Path,
        preferred_name: str,
        identity: str,
        used_names: set[str],
    ) -> None:
        slug = normalize_name(preferred_name) or f"skill-{hashlib.sha256(identity.encode()).hexdigest()[:8]}"
        base = slug
        suffix = 2
        while slug.casefold() in used_names:
            slug = f"{base}-{suffix}"
            suffix += 1
        used_names.add(slug.casefold())
        destination = target_root / slug
        destination.mkdir()
        total = 0
        root = source.resolve()
        for child in sorted(source.rglob("*")):
            if child.is_symlink() or not child.is_file():
                continue
            resolved = child.resolve()
            if not resolved.is_relative_to(root):
                continue
            size = child.stat().st_size
            total += size
            if total > _MAX_SNAPSHOT_BYTES:
                raise ValueError(f"Skill {preferred_name!r} is too large to include in a coding task")
            relative = child.relative_to(source)
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, output)

    @staticmethod
    def _content_hash(path: Path) -> str:
        digest = hashlib.sha256()
        total = 0
        if path.is_file():
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(64 * 1024), b""):
                    total += len(chunk)
                    if total > _MAX_SNAPSHOT_BYTES:
                        raise ValueError(f"Skill {path.name!r} is too large to include in a coding task")
                    digest.update(chunk)
            return digest.hexdigest()
        root = path.resolve()
        for child in sorted(path.rglob("*")):
            if child.is_symlink() or not child.is_file():
                continue
            resolved = child.resolve()
            if not resolved.is_relative_to(root):
                continue
            digest.update(child.relative_to(path).as_posix().encode())
            with child.open("rb") as stream:
                for chunk in iter(lambda: stream.read(64 * 1024), b""):
                    total += len(chunk)
                    if total > _MAX_SNAPSHOT_BYTES:
                        raise ValueError(f"Skill {path.name!r} is too large to include in a coding task")
                    digest.update(chunk)
        return digest.hexdigest()
