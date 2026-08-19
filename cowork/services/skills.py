from __future__ import annotations

import io
import json
import logging
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
from cowork.common.paths import safe_join
from cowork.common.settings import get_app_settings
from cowork.db.scoped import TenantScope, scoped_storage_root
from cowork.services.skill_links import reconcile_skill_links, remove_skill_links
from cowork.models.skill import (
    META_CREATED_AT,
    META_DISPLAY_NAME,
    META_ENABLED,
    META_PROJECTS,
    META_UPDATED_AT,
    Skill,
)


logger = logging.getLogger(__name__)

# Per-file cap (matches the skill-draft cap): filters mispackaged data blobs and
# stops one bad skill from bloating the payload. The aggregate request size is
# bounded downstream by the producer's _fit_request against the real stdin cap.
_TURN_SKILL_FILE_MAX = 200_000

# Packaged skills shipped with cowork. Bump BUILTIN_SKILLS_VERSION when the set
# changes; seeding re-runs only when the stored version is lower, so a future
# release can ship more skills without touching ones the user edited or deleted.
# Desktop seeding (unkeyed store, DB sentinel) lives in migrations.py and shares
# this version; org seeding (per-org store, file marker) is a SkillService method.
BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills_builtin"
BUILTIN_SKILLS_VERSION = 1
#: Org-mode version marker, a file in the org's own store. See
#: ``SkillService.ensure_builtin_skills`` for why this is not a Setting row.
BUILTIN_SKILLS_MARKER = ".builtins_seeded"


def _wire_len(text: str) -> int:
    """Bytes `text` occupies on the wire. The controller uses ensure_ascii JSON,
    so counting chars would under-count non-ASCII ~6x against the byte cap."""
    return len(json.dumps(text))


def _skill_wire_files(skill_dir: Path, root: Path) -> dict[str, str] | None:
    """One skill directory as ``{posix relpath: text}`` for the wire, or None
    to drop it (no parseable/fitting SKILL.md).

    Text files only; ``stats.json`` (private sidecar), hidden files, and
    symlinks are excluded. Containment is checked against ``root``, not
    ``skill_dir`` — resolving against a symlinked skill_dir would resolve to
    its foreign target and find everything "inside" it.
    """
    files: dict[str, str] = {}
    root_resolved = root.resolve()
    for child in sorted(skill_dir.rglob("*")):
        if not child.is_file() or child.is_symlink():
            continue
        try:
            # rglob not following dir symlinks is just a stdlib default; ensure
            # no child resolves into another org's store.
            child.resolve().relative_to(root_resolved)
        except (OSError, ValueError):
            logger.warning("turn skills: skipping out-of-tree path %r in %r",
                           str(child), skill_dir.name)
            continue
        rel = child.relative_to(skill_dir).as_posix()
        if rel == "stats.json" or any(p.startswith(".") for p in rel.split("/")):
            continue
        try:
            text = child.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            logger.warning("turn skills: skipping unreadable or non-text file %r in %r",
                           rel, skill_dir.name)
            continue
        if _wire_len(text) > _TURN_SKILL_FILE_MAX:
            if rel == SKILL_FILE:
                logger.warning("turn skills: dropping %r (SKILL.md over %d wire bytes)",
                               skill_dir.name, _TURN_SKILL_FILE_MAX)
                return None
            logger.warning("turn skills: skipping oversized file %r in %r (%d wire bytes)",
                           rel, skill_dir.name, _wire_len(text))
            continue
        files[rel] = text
    if SKILL_FILE not in files:
        return None
    return files


def build_turn_skills(scope: TenantScope | None, project_path: str | None = None) -> dict[str, dict]:
    """Skills for a remote turn: ``{slug: {"files": {relpath: text}}}``.

    Org-keyed through the passed `scope` (the producer binds no ambient scope).
    Selection mirrors skill_links: enabled skills whose ``metadata.projects`` is
    empty or names this project's folder.

    Slug and files both come from the DIRECTORY, not frontmatter ``name`` — the
    two can drift on a hand-edited store, and resolving by frontmatter could
    ship another (e.g. disabled) skill's files. Every drop is logged.
    """
    svc = SkillService(scope)
    # A fresh org's store starts empty and there is no org-creation hook, so the
    # builtins are seeded on first read. Seeded here for an org that chats before
    # it ever opens the skills menu; the menu seeds too, so whichever comes first
    # wins. No-op after that, and in local mode.
    svc.ensure_builtin_skills()

    project_name = Path(project_path).name if project_path else None

    out: dict[str, dict] = {}
    if not svc.root.exists():
        return out
    root = svc.root
    for skill_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        slug = skill_dir.name
        if skill_dir.is_symlink():
            # is_dir() follows the link, so a symlinked dir could point into
            # another org's store — reject it.
            logger.warning("turn skills: dropping %r (symlinked skill directory)", slug)
            continue
        if not (skill_dir / SKILL_FILE).exists():
            continue
        try:
            validate_name(slug)
        except ValueError:
            logger.warning("turn skills: dropping %r (directory name is not a valid slug)", slug)
            continue
        skill = _skill_from_dir(skill_dir)
        if skill is None:
            logger.warning("turn skills: dropping %r (unparseable SKILL.md)", slug)
            continue
        if not skill.enabled:
            continue
        if skill.projects and (project_name is None or project_name not in skill.projects):
            continue
        files = _skill_wire_files(skill_dir, root)
        if files is None:
            logger.warning("turn skills: dropping %r (no SKILL.md within size bounds)", slug)
            continue
        out[slug] = {"files": files}
    return out


def _skill_from_dir(skill_dir: Path) -> Skill | None:
    """Read a ``SKILL.md`` folder into a ``Skill``.
    """
    agent = parse_skill_dir(skill_dir)
    if agent is None:
        return None
    skill = Skill.model_construct(**dict(agent))

    return skill


class SkillService:
    """File-backed skill store using the agentskills.io ``SKILL.md`` format.

    Org mode keys the store per org (``<shared_root>/<org_id>/skills``); local
    mode uses the shared root unchanged."""

    def __init__(self, scope: TenantScope | None = None) -> None:
        settings = get_app_settings()
        self._scope = scope
        self.root = scoped_storage_root(Path(settings.skill.root_dir), scope, store="skills")
        # Symlink distribution is desktop-only (skill_links resolves the unkeyed
        # root and scans all project dirs). Keyed on deployment mode, not just
        # scope — an unscoped service (migration, seeding) must not fan symlinks
        # out of the unkeyed root in org mode either.
        self._link_projects = settings.tenancy_mode != "org" and (
            scope is None or not scope.org_mode
        )

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
        metadata[META_CREATED_AT] = created_at.isoformat()
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
            if self._link_projects:
                remove_skill_links(skill.name)
            skill.name = new_slug
            if self._link_projects:
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
        metadata.pop(META_PROJECTS, None)
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
            return SkillService._normalize_skill_dir(src_dir)
        if only.suffix.lower() == ".md" and only.name != SKILL_FILE:
            only.rename(src_dir / SKILL_FILE)

    # A small archive can expand to many times its size — bound the expansion.
    _ZIP_MAX_UNCOMPRESSED = 200 * 1024 * 1024  # 200 MB

    @staticmethod
    def _safe_extract_zip(data: bytes, dest: Path) -> None:
        """Extract a zip into ``dest``, rejecting escaping paths, symlinks, and
        archives that would expand past the size bound."""
        import stat

        dest_resolved = dest.resolve()
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                infos = zf.infolist()
                if sum(i.file_size for i in infos) > SkillService._ZIP_MAX_UNCOMPRESSED:
                    raise ValueError("Archive would expand beyond the allowed size.")
                for info in infos:
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
        if self._link_projects:
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
        # Project per-project links to match the skill's metadata (desktop only).
        if self._link_projects:
            reconcile_skill_links(skill)

    def _rename_dir(self, old_slug: str, new_slug: str) -> None:
        self._ensure_root()
        os.replace(self._skill_dir(old_slug), self._skill_dir(new_slug))

    # ── builtin seeding ──────────────────────────────────────────────────────
    def _copy_builtin_skills(self) -> int:
        """Copy packaged builtins into this store, skipping slugs it already has.

        Returns how many were copied. Never overwrites, so a skill the user
        edited or deleted stays as they left it.
        """
        if not BUILTIN_SKILLS_DIR.exists():
            return 0
        self._ensure_root()
        copied = 0
        for src in sorted(BUILTIN_SKILLS_DIR.iterdir()):
            if not src.is_dir() or not (src / SKILL_FILE).exists():
                continue
            try:
                dest = self._skill_dir(src.name)
                if dest.exists():
                    continue  # keep the user-editable copy untouched
                # Copied file by file, each destination re-checked for containment
                # with safe_join. copy2, not copyfile: copytree preserved mode, and a
                # future builtin shipping an executable helper must keep its +x.
                for child in sorted(p for p in src.rglob("*") if p.is_file()):
                    target = safe_join(dest, *child.relative_to(src).parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(child, target)
            except ValueError:
                # Either the slug resolves outside the store, or safe_join rejected a
                # destination. The agent writes into its own org's tree on shared
                # storage, so it can plant a symlink under `dest` between the exists()
                # check above and the write. Skip this builtin: propagating would 500
                # every skills read and every turn for the org, which a tenant must
                # not be able to do to itself.
                logger.warning("Skipping builtin %r: it does not resolve inside %s",
                               src.name, self.root, exc_info=True)
                continue
            copied += 1
        return copied

    def ensure_builtin_skills(self) -> bool:
        """Seed this org's skill store with the packaged builtins, on first use.

        Org mode only: desktop seeds once at boot via ``migrations.seed_builtin_skills``,
        and its store is unkeyed so there is nothing per-tenant to do. There is no
        org-creation hook to hang this on, so it runs lazily where skills are read.

        The version marker is a FILE in the org's store rather than a Setting row,
        so marker and skills share fate:
        - a DB row would outlive a lost volume and leave the org permanently empty,
          because seeding would believe it had already run
        - a deliberate deletion of every skill is still respected, since the marker
          survives it and blocks a re-seed

        Returns True if seeding ran. Fail-soft: a filesystem problem leaves the org
        unseeded and retries on the next read rather than failing the request.
        """
        if self._scope is None or not self._scope.org_mode:
            return False
        try:
            marker = self.root / BUILTIN_SKILLS_MARKER
            current = 0
            if marker.is_file():
                raw = marker.read_text(encoding="utf-8").strip()
                current = int(raw) if raw.isdigit() else 0
            if current >= BUILTIN_SKILLS_VERSION:
                return False
            if not BUILTIN_SKILLS_DIR.exists():
                # Nothing to seed from — a packaging fault, not a seeded org. Writing
                # the marker here would record "done" against an empty store, and the
                # org would stay empty forever once the image is fixed.
                logger.warning("Builtin skills are missing from this build (%s); not marking %s seeded",
                               BUILTIN_SKILLS_DIR, self.root)
                return False
            copied = self._copy_builtin_skills()
            marker.write_text(f"{BUILTIN_SKILLS_VERSION}\n", encoding="utf-8")
        except OSError:
            logger.warning("Could not seed builtin skills for org %s",
                           getattr(self._scope, "org_id", None), exc_info=True)
            return False
        logger.info("Seeded %d builtin skill(s) into %s", copied, self.root)
        return True
