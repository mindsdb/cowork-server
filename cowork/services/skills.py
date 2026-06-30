from __future__ import annotations

import io
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from anton.core.tools.skill_format import (
    SKILL_FILE,
    dump_skill,
    normalize_name,
    parse_skill_dir,
    validate_name,
)
from sqlmodel import Session

from cowork.common.settings import get_app_settings
from cowork.services.skill_links import reconcile_skill_links, remove_skill_links
from cowork.models.skill import (
    META_CREATED_AT,
    META_DISPLAY_NAME,
    META_ENABLED,
    META_PROJECTS,
    META_UPDATED_AT,
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



    def __init__(self, session: Session) -> None:
        self.session = session
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
    def _slug_from_label(label: str) -> str:
        """Normalize a user-supplied label into a slug, rejecting empties.

        A label made only of symbols/whitespace normalizes to "" — surface a
        clear validation error instead of letting it resolve to the root dir.
        """
        slug = normalize_name(label)
        if not slug:
            raise ValueError(
                f"Skill name {label!r} must contain at least one letter or digit."
            )
        return slug

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

    @staticmethod
    def _apply_metadata_flags(
        metadata: dict[str, str],
        enabled: bool | None,
        projects: list[str] | None,
    ) -> None:
        """Write enabled/projects into ``metadata`` (kept clean: omit defaults)."""
        if enabled is not None:
            if enabled:
                metadata.pop(META_ENABLED, None)  # default-on
            else:
                metadata[META_ENABLED] = "false"
        if projects is not None:
            joined = ",".join(p.strip() for p in projects if p.strip())
            if joined:
                metadata[META_PROJECTS] = joined
            else:
                metadata.pop(META_PROJECTS, None)

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
        instructions: str,
        name: str | None = None,
        description: str | None = None,
        enabled: bool | None = None,
        projects: list[str] | None = None,
    ) -> Skill:
        label = self._slug_from_label(label)
        if self._skill_dir(label).exists():
            raise ValueError(f"A skill named '{label}' already exists.")

        metadata = self._build_metadata(label, name, datetime.now(timezone.utc))
        self._apply_metadata_flags(metadata, enabled, projects)
        skill = Skill(
            name=label,
            instructions=instructions or "",
            # description is required and non-empty by spec; fall back to the
            # display name / slug so we never write an empty value.
            description=(description or "").strip() or name or label,
            metadata=metadata,
        )
        self._write(skill)
        return self.get_skill(label)

    def update_skill(
        self,
        skill_id: str,
        label: str | None = None,
        name: str | None = None,
        description: str | None = None,
        instructions: str | None = None,
        enabled: bool | None = None,
        projects: list[str] | None = None,
    ) -> Skill:
        skill = self.get_skill(skill_id)
        metadata = dict(skill.metadata)
        self._apply_metadata_flags(metadata, enabled, projects)

        new_slug = skill.name
        if label is not None:
            new_slug = self._slug_from_label(label)

        if name is not None:
            if name and name != new_slug:
                metadata[META_DISPLAY_NAME] = name
            else:
                metadata.pop(META_DISPLAY_NAME, None)
        if description is not None:
            skill.description = (description.strip() or skill.display_name or new_slug)
        if instructions is not None:
            skill.instructions = instructions

        renaming = new_slug != skill.name
        if renaming and self._skill_dir(new_slug).exists():
            raise ValueError(f"A skill named '{new_slug}' already exists.")

        # Write the updated content into the current dir first, then rename the
        # whole dir last. A failed _write leaves the old dir intact; the
        # destructive os.replace only runs once content is safely persisted.
        skill.metadata = metadata
        self._write(skill)
        if renaming:
            self._rename_dir(skill.name, new_slug)
            remove_skill_links(skill.name)
            skill.name = new_slug
            reconcile_skill_links(skill)

        return self.get_skill(skill.name)

    def import_skill(self, data: bytes, filename: str | None = None) -> Skill:
        """Import a skill from an uploaded file.

        Supported formats (by extension):
          - ``.md`` / ``.skill`` — a text ``SKILL.md``.
          - ``.zip`` — the contents of a skill folder, extracted as-is.

        Validation = "does its ``SKILL.md`` parse via skill_format". Raises
        ``ValueError`` for an unparseable/unsafe file, ``FileExistsError`` on
        slug collision.
        """
        if Path(filename or "").suffix.lower() == ".zip":
            return self._import_zip(data)
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("File must be UTF-8 encoded text.")
        # parse_skill_dir needs a dir holding a file literally named SKILL.md.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp) / "skill"
            tmp_dir.mkdir(parents=True)
            (tmp_dir / SKILL_FILE).write_text(content, encoding="utf-8")
            return self._persist_imported(tmp_dir, copy_tree=False)

    def _import_zip(self, data: bytes) -> Skill:
        with tempfile.TemporaryDirectory() as tmp:
            extract_dir = Path(tmp) / "skill"
            extract_dir.mkdir(parents=True)
            self._safe_extract_zip(data, extract_dir)
            return self._persist_imported(extract_dir, copy_tree=True)

    def _persist_imported(self, src_dir: Path, *, copy_tree: bool) -> Skill:
        """Validate a parsed skill folder and persist it into the canon.

        ``copy_tree`` copies the whole ``src_dir`` (zip: keep sibling files);
        otherwise only ``SKILL.md`` is written by ``_write``.
        """
        self._normalize_skill_dir(src_dir)
        skill = _skill_from_dir(src_dir)
        if skill is None:
            raise ValueError("Could not find a parseable SKILL.md in the upload.")
        if not skill.name:
            raise ValueError("Skill name is missing or invalid.")
        if self._skill_dir(skill.name).exists():
            raise FileExistsError(f"A skill named '{skill.name}' already exists.")

        metadata = dict(skill.metadata)
        metadata.setdefault(META_CREATED_AT, datetime.now(timezone.utc).isoformat())
        skill.metadata = metadata
        if not skill.description.strip():
            skill.description = skill.display_name or skill.name

        if copy_tree:
            self._ensure_root()
            dest = self._skill_dir(skill.name)
            shutil.copytree(src_dir, dest)
            try:
                # _write (re)writes SKILL.md canonically, stamps updated_at,
                # reconciles links; sibling files are already in place.
                self._write(skill)
            except Exception:
                shutil.rmtree(dest, ignore_errors=True)
                raise
        else:
            self._write(skill)
        return self.get_skill(skill.name)

    @staticmethod
    def _normalize_skill_dir(src_dir: Path) -> None:
        """Unwrap a single-element upload so ``SKILL.md`` sits at ``src_dir`` root.

        - A lone wrapping folder (zip packed with its folder) → hoist its
          contents up one level (repeats for nested wrapping).
        - A lone ``*.md`` file → rename it to ``SKILL.md``.
        """

        entries = list(src_dir.iterdir())
        if len(entries) != 1:
            return
        only = entries[0]
        if only.is_dir():
            for item in list(only.iterdir()):
                shutil.move(str(item), str(src_dir / item.name))
            only.rmdir()
            return
        if only.suffix.lower() == ".md" and only.name != SKILL_FILE:
            only.rename(src_dir / SKILL_FILE)

    @staticmethod
    def _safe_extract_zip(data: bytes, dest: Path) -> None:
        """Extract a zip into ``dest``, rejecting paths that escape it or are symlinks."""
        import stat

        dest_resolved = dest.resolve()
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for info in zf.infolist():
                    # Upper 16 bits of external_attr are Unix mode bits (0 on Windows zips).
                    unix_mode = info.external_attr >> 16
                    if unix_mode and stat.S_ISLNK(unix_mode):
                        raise ValueError(f"Archive contains a symlink: {info.filename!r}")
                    target = (dest / info.filename).resolve()
                    if target != dest_resolved and dest_resolved not in target.parents:
                        raise ValueError(f"Unsafe path in archive: {info.filename!r}")
                zf.extractall(dest)
        except zipfile.BadZipFile:
            raise ValueError("Uploaded file is not a valid zip archive.")

    def delete_skill(self, slug: str) -> bool:
        skill_dir = self._skill_dir(slug)
        if not skill_dir.exists():
            return False
        shutil.rmtree(skill_dir)
        remove_skill_links(slug)
        return True

    # ── low-level fs ─────────────────────────────────────────────────────────
    def _write(self, skill: Skill) -> None:
        self._ensure_root()
        skill.metadata[META_UPDATED_AT] = datetime.now(timezone.utc).isoformat()
        skill_dir = self._skill_dir(skill.name)
        skill_dir.mkdir(parents=True, exist_ok=True)
        target = skill_dir / SKILL_FILE
        tmp = skill_dir / f".{SKILL_FILE}.tmp"
        tmp.write_text(dump_skill(skill), encoding="utf-8")
        os.replace(tmp, target)  # atomic within the same directory
        # Project per-project links to match the skill's metadata.
        reconcile_skill_links(skill)

    def _rename_dir(self, old_slug: str, new_slug: str) -> None:
        self._ensure_root()
        os.replace(self._skill_dir(old_slug), self._skill_dir(new_slug))
