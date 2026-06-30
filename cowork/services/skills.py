from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from anton.core.tools.skill_format import (
    SKILL_FILE,
    dump_skill,
    normalize_name,
    parse_skill_dir,
    validate_name,
)

from cowork.common.settings import get_app_settings
from cowork.models.skill import (
    META_CREATED_AT,
    META_DISPLAY_NAME,
    Skill,
)


def _skill_from_dir(skill_dir: Path) -> Skill | None:
    """Read a ``SKILL.md`` folder into a ``Skill``.
    """
    agent = parse_skill_dir(skill_dir)
    if agent is None:
        return None
    skill = Skill.model_construct(**dict(agent))

    return skill


class SkillService:
    """File-backed skill store using the agentskills.io ``SKILL.md`` format."""



    def __init__(self) -> None:
        self.root = Path(get_app_settings().skill.root_dir)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _skill_dir(self, slug: str) -> Path:
        validate_name(slug)
        skill_dir = (self.root / slug).resolve()
        if not skill_dir.is_relative_to(self.root.resolve()):
            raise ValueError(f"Invalid skill name: {slug!r}")
        return skill_dir

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _build_metadata(
        slug: str,
        name: str | None,
        created_at: datetime,
    ) -> dict[str, str]:
        metadata: dict[str, str] = {}
        if name and name != slug:
            metadata[META_DISPLAY_NAME] = name
        metadata[META_CREATED_AT] = created_at.replace(tzinfo=None).isoformat()
        return metadata

    # ── reads ────────────────────────────────────────────────────────────────
    def list_skills(self) -> list[Skill]:
        if not self.root.exists():
            return []
        skills: list[Skill] = []
        for entry in self.root.iterdir():
            if entry.is_dir() and (entry / SKILL_FILE).exists():
                skill = _skill_from_dir(entry)
                if skill is not None:
                    skills.append(skill)
        skills.sort(key=lambda s: (s.created_at is None, s.created_at), reverse=True)
        return skills

    def get_skill(self, slug: str) -> Skill:
        skill_dir = self._skill_dir(slug)
        skill = _skill_from_dir(skill_dir) if (skill_dir / SKILL_FILE).exists() else None
        if skill is None:
            raise ValueError(f"Skill {slug!r} not found.")
        return skill

    # ── writes ───────────────────────────────────────────────────────────────
    def create_skill(
        self,
        label: str,
        name: str,
        instructions: str,
        description: str | None = None,
    ) -> Skill:
        slug = normalize_name(label)
        if not slug:
            raise ValueError(
                f"Skill name {label!r} is empty"
            )
        if self._skill_dir(slug).exists():
            raise ValueError(f"A skill named '{slug}' already exists.")

        skill = Skill(
            name=slug,
            instructions=instructions or "",
            # description is required and non-empty by spec; fall back to the
            # display name / slug so we never write an empty value.
            description=(description or "").strip() or name or slug,
            metadata=self._build_metadata(slug, name, datetime.now(timezone.utc)),
        )
        self._write(skill)
        return self.get_skill(slug)

    def update_skill(
        self,
        skill_id: str,
        label: str | None = None,
        name: str | None = None,
        description: str | None = None,
        instructions: str | None = None,
    ) -> Skill:
        skill = self.get_skill(skill_id)
        metadata = dict(skill.metadata)

        new_slug = skill.name
        if label is not None:
            new_slug = normalize_name(label)

        if name is not None:
            if name and name != new_slug:
                metadata[META_DISPLAY_NAME] = name
            else:
                metadata.pop(META_DISPLAY_NAME, None)
        if description is not None:
            skill.description = (description.strip() or skill.display_name or new_slug)
        if instructions is not None:
            skill.instructions = instructions

        if new_slug != skill.name:
            if self._skill_dir(new_slug).exists():
                raise ValueError(f"A skill named '{new_slug}' already exists.")
            self._rename_dir(skill.name, new_slug)
            skill.name = new_slug

        skill.metadata = metadata
        self._write(skill)
        return self.get_skill(skill.name)

    def delete_skill(self, slug: str) -> bool:
        skill_dir = self._skill_dir(slug)
        if not skill_dir.exists():
            return False
        shutil.rmtree(skill_dir)
        return True

    # ── low-level fs ─────────────────────────────────────────────────────────
    def _write(self, skill: Skill) -> None:
        self._ensure_root()
        skill_dir = self._skill_dir(skill.name)
        skill_dir.mkdir(parents=True, exist_ok=True)
        target = skill_dir / SKILL_FILE
        tmp = skill_dir / f".{SKILL_FILE}.tmp"
        tmp.write_text(dump_skill(skill), encoding="utf-8")
        os.replace(tmp, target)  # atomic within the same directory

    def _rename_dir(self, old_slug: str, new_slug: str) -> None:
        self._ensure_root()
        os.replace(self._skill_dir(old_slug), self._skill_dir(new_slug))
