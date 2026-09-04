from __future__ import annotations

import io
import json
import logging
import os
import shutil
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from anton.core.tools.skill_format import (
    SKILL_FILE,
    dump_skill,
    normalize_name,
    parse_skill_dir,
    validate_name,
)

from cowork.common.paths import (
    dir_rename,
    dir_replace,
    dir_rmtree,
    dir_unlink,
    pinned_dir,
    safe_join,
)
from cowork.common.settings import get_app_settings
from cowork.db.scoped import TenantScope, scoped_storage_root
from cowork.models.skill import (
    META_CREATED_AT,
    META_DISPLAY_NAME,
    META_ENABLED,
    META_PROJECTS,
    META_UPDATED_AT,
    Skill,
)
from cowork.services.skill_links import reconcile_skill_links, remove_skill_links

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
# Versioned identity manifest. Authorization must not depend on the packaged
# directory being readable: an incomplete image must not make an existing
# shared-volume copy mutable.
BUILTIN_SKILL_SLUGS = frozenset(
    {
        "documents",
        "docx",
        "gmail",
        "google-sheets",
        "google-slides",
        "pdf",
        "presentations",
        "skill-creator",
    }
)
#: Org-mode version marker, a file in the org's own store. See
#: ``SkillService.ensure_builtin_skills`` for why this is not a Setting row.
BUILTIN_SKILLS_MARKER = ".builtins_seeded"

# Skills in this set belong to the coding product, not the general Cowork
# assistant. They remain packaged beside the other builtins for now, but are
# seeded into a separate store and are never exposed by ``SkillService``.
CODE_ONLY_BUILTIN_SKILL_NAMES = frozenset({
    "thermo-nuclear-code-quality-review",
})
CODE_BUILTIN_SKILLS_VERSION = 1
CODE_BUILTIN_SKILLS_MARKER = ".builtins_seeded"


def is_builtin_skill(slug: str) -> bool:
    """Whether ``slug`` is reserved by the packaged, immutable skill set."""
    return slug in BUILTIN_SKILL_SLUGS


def builtin_skill_refusal(slug: str) -> str:
    """Why every change to ``slug`` is refused, in terms a user can act on.

    Identity is the directory slug, so a skill somebody wrote before the
    packaged set claimed that name is frozen by the same rule as the packaged
    copy, and "this is a built-in" reads as a lie to whoever wrote it. Naming
    the reservation instead tells them the way out is a different name.
    """
    return (
        f"Skill name {slug!r} is reserved by a packaged built-in skill and is "
        "immutable for every role. Copy its contents into a new skill under a "
        "different name."
    )


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
            logger.warning(
                "turn skills: skipping out-of-tree path %r in %r",
                str(child),
                skill_dir.name,
            )
            continue
        rel = child.relative_to(skill_dir).as_posix()
        if rel == "stats.json" or any(p.startswith(".") for p in rel.split("/")):
            continue
        try:
            text = child.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            logger.warning(
                "turn skills: skipping unreadable or non-text file %r in %r",
                rel,
                skill_dir.name,
            )
            continue
        if _wire_len(text) > _TURN_SKILL_FILE_MAX:
            if rel == SKILL_FILE:
                logger.warning(
                    "turn skills: dropping %r (SKILL.md over %d wire bytes)",
                    skill_dir.name,
                    _TURN_SKILL_FILE_MAX,
                )
                return None
            logger.warning(
                "turn skills: skipping oversized file %r in %r (%d wire bytes)",
                rel,
                skill_dir.name,
                _wire_len(text),
            )
            continue
        files[rel] = text
    if SKILL_FILE not in files:
        return None
    return files


def build_turn_skills(
    scope: TenantScope | None, project_path: str | None = None
) -> dict[str, dict]:
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
    # builtins are seeded on first read. This function has no production caller
    # today — cloud turns read skills off the shared mount directly rather than
    # through this payload (see `_stage_remote_workspace_files`, which is the
    # actual seed trigger for a chat-first org) — but it seeds too, so it stays
    # correct if a future caller reappears. No-op after the first run, and in
    # local mode.
    svc.ensure_builtin_skills()

    project_name = Path(project_path).name if project_path else None

    out: dict[str, dict] = {}
    if not svc.root.exists():
        return out
    root = svc.root
    for skill_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        slug = skill_dir.name
        if not svc.allows_skill(slug):
            continue
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
            logger.warning(
                "turn skills: dropping %r (directory name is not a valid slug)", slug
            )
            continue
        skill = _skill_from_dir(skill_dir)
        if skill is None:
            logger.warning("turn skills: dropping %r (unparseable SKILL.md)", slug)
            continue
        if not skill.enabled:
            continue
        if skill.projects and (
            project_name is None or project_name not in skill.projects
        ):
            continue
        files = _skill_wire_files(skill_dir, root)
        if files is None:
            logger.warning(
                "turn skills: dropping %r (no SKILL.md within size bounds)", slug
            )
            continue
        out[slug] = {"files": files}
    return out


def _skill_from_dir(
    skill_dir: Path,
    *,
    canonicalize_name: bool = True,
) -> Skill | None:
    """Read a ``SKILL.md`` folder into a ``Skill``.

    Persisted skill identity is its directory slug. Frontmatter can drift after
    a hand edit or partial filesystem mutation, but reads must not let that
    content select another authorization or builtin identity.
    """
    agent = parse_skill_dir(skill_dir)
    if agent is None:
        return None
    skill = Skill.model_construct(**dict(agent))
    if canonicalize_name:
        skill.name = skill_dir.name

    return skill


@dataclass(frozen=True)
class ProjectReferenceRewrite:
    """One exact, reversible skill edit caused by a project rename."""

    slug: str
    previous_content: bytes
    updated_skill: Skill


class SkillService:
    """File-backed skill store using the agentskills.io ``SKILL.md`` format.

    Org mode keys the store per org (``<shared_root>/<org_id>/skills``); local
    mode uses the shared root unchanged."""

    store_name = "skills"
    seed_builtins_in_local_mode = False

    def __init__(self, scope: TenantScope | None = None) -> None:
        settings = get_app_settings()
        self._scope = scope
        base = Path(settings.skill.root_dir)
        if self.store_name != "skills":
            base = base.parent / self.store_name
        self.root = scoped_storage_root(base, scope, store=self.store_name)
        # Symlink distribution is desktop-only (skill_links resolves the unkeyed
        # root and scans all project dirs). Keyed on deployment mode, not just
        # scope — an unscoped service (migration, seeding) must not fan symlinks
        # out of the unkeyed root in org mode either.
        self._link_projects = self.store_name == "skills" and settings.tenancy_mode != "org" and (
            scope is None or not scope.org_mode
        )

    @property
    def builtin_skills_dir(self) -> Path:
        return BUILTIN_SKILLS_DIR

    @property
    def builtin_skills_version(self) -> int:
        return BUILTIN_SKILLS_VERSION

    @property
    def builtin_skills_marker(self) -> str:
        return BUILTIN_SKILLS_MARKER

    def allows_skill(self, slug: str) -> bool:
        """Whether this store owns ``slug`` at its product boundary."""
        return slug not in CODE_ONLY_BUILTIN_SKILL_NAMES

    def includes_packaged_builtin(self, slug: str) -> bool:
        return self.allows_skill(slug)

    @property
    def packaged_builtin_slugs(self) -> frozenset[str]:
        """Identities this store ships as immutable packaged skills.

        Immutability is per store. The general store owns
        ``BUILTIN_SKILL_SLUGS`` and the Code store owns
        ``CODE_ONLY_BUILTIN_SKILL_NAMES``, so a store must not consult the
        other's manifest and conclude its own packaged skill is editable.
        """
        return BUILTIN_SKILL_SLUGS

    def builtin_skill_names(self) -> set[str]:
        root = self.builtin_skills_dir
        if not root.is_dir():
            return set()
        return {
            path.name
            for path in root.iterdir()
            if path.is_dir()
            and (path / SKILL_FILE).exists()
            and self.includes_packaged_builtin(path.name)
        }

    # ── helpers ──────────────────────────────────────────────────────────────
    def _skill_entry(self, slug: str) -> Path:
        """Return the validated lexical entry without following a slug symlink."""
        validate_name(slug)
        safe_slug = os.path.basename(slug)
        if (
            safe_slug != slug
            or safe_slug in {"", ".", ".."}
            or "\\" in safe_slug
            or "\0" in safe_slug
        ):
            raise ValueError(f"Invalid skill name: {slug!r}")
        root = os.path.abspath(self.root)
        entry = os.path.abspath(os.path.join(root, safe_slug))
        root_prefix = root if root.endswith(os.sep) else f"{root}{os.sep}"
        if not entry.startswith(root_prefix):
            raise ValueError(f"Invalid skill name: {slug!r}")
        return Path(entry)

    def _skill_dir(self, slug: str) -> Path:
        entry = self._skill_entry(slug)
        root = os.path.realpath(self.root)
        skill_dir = os.path.realpath(entry)
        root_prefix = root if root.endswith(os.sep) else f"{root}{os.sep}"
        if not skill_dir.startswith(root_prefix):
            raise ValueError(f"Invalid skill name: {slug!r}")
        return Path(skill_dir)

    def _is_immutable_builtin(self, slug: str) -> bool:
        return bool(
            self._scope is not None
            and self._scope.org_mode
            and slug in self.packaged_builtin_slugs
        )

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _replace_direct_child(directory: Path, source: str, destination: str) -> None:
        """Swap a temp file onto its final name without re-resolving by path.

        The durable write is the last step of every skill mutation, so the
        destination must not be re-opened by name here: a link planted between
        the write and the swap would otherwise receive the bytes.
        """
        with pinned_dir(directory, nofollow_base=True) as pinned:
            dir_replace(pinned, source, destination)

    @staticmethod
    def _rmtree_direct_child(
        root: Path,
        name: str,
        *,
        ignore_errors: bool = False,
    ) -> None:
        try:
            with pinned_dir(root, nofollow_base=True) as directory:
                dir_rmtree(directory, name)
        except OSError:
            if not ignore_errors:
                raise

    @staticmethod
    def _unlink_direct_child(
        root: Path,
        name: str,
        *,
        ignore_errors: bool = False,
    ) -> None:
        """Unlink a direct child of ``root`` against a pinned descriptor."""
        try:
            with pinned_dir(root, nofollow_base=True) as directory:
                dir_unlink(directory, name)
        except OSError:
            if not ignore_errors:
                raise

    @staticmethod
    def _rename_direct_child(
        source_root: Path,
        source_name: str,
        destination_root: Path,
        destination_name: str,
    ) -> None:
        """Move a direct child between trusted roots, pinning both by descriptor.

        A path-based rename re-walks every component, so a link planted at
        either end after the containment check redirects the move onto whatever
        it points at. Pinning the roots leaves the kernel resolving only the
        final name against the inode already opened, and a link squatting the
        destination name is replaced rather than followed.
        """
        if source_root == destination_root:
            with pinned_dir(source_root, nofollow_base=True) as root:
                dir_rename(root, source_name, root, destination_name)
            return
        with pinned_dir(source_root, nofollow_base=True) as source, pinned_dir(
            destination_root,
            create=True,
            nofollow_base=True,
        ) as destination:
            dir_rename(source, source_name, destination, destination_name)

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
            if (
                self.allows_skill(entry.name)
                and entry.is_dir()
                and (entry / SKILL_FILE).exists()
            ):
                skill = _skill_from_dir(entry, canonicalize_name=True)
                if skill is not None:
                    skills.append(skill)
        skills.sort(key=lambda s: (s.created_at is None, s.created_at), reverse=True)
        return skills

    def get_skill(self, slug: str) -> Skill:
        if not self.allows_skill(slug):
            raise ValueError(f"Skill {slug!r} not found.")
        skill_dir = self._skill_dir(slug)
        skill = (
            _skill_from_dir(skill_dir, canonicalize_name=True)
            if (skill_dir / SKILL_FILE).exists()
            else None
        )
        if skill is None:
            raise ValueError(f"Skill {slug!r} not found.")
        return skill

    def has_complete_skill(self, slug: str) -> bool:
        """Whether ``slug`` has a regular, parseable canonical skill file.

        A pending create may outlive the process that reserved it.  The directory
        alone is not evidence that the filesystem mutation survived: ``_write``
        creates it before replacing ``SKILL.md``.  Recovery must therefore reject
        empty directories, temporary files, symlinks, malformed frontmatter, and
        a file whose declared identity does not match its directory.
        """
        skill_dir = self._skill_entry(slug)
        skill_file = skill_dir / SKILL_FILE
        try:
            if skill_dir.is_symlink() or not skill_dir.is_dir():
                return False
            if skill_file.is_symlink() or not skill_file.is_file():
                return False
            parsed = _skill_from_dir(skill_dir, canonicalize_name=False)
        except (OSError, UnicodeError, ValueError):
            return False
        return parsed is not None and parsed.name == slug

    def discard_incomplete_skill(self, slug: str) -> bool:
        """Atomically move aside and remove an invalid stale-create directory.

        Moving to a unique staging path first means a later create can claim the
        canonical slug without a delayed cleanup deleting its new directory.
        Symlinks and non-directory entries are moved as entries and then unlinked;
        they are never followed by ``rmtree``.
        """
        if self.has_complete_skill(slug):
            return False
        source = self._skill_entry(slug)
        staging_root = self.root / ".incomplete-staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        staged = staging_root / f"{source.name}-{uuid4()}"
        try:
            self._rename_direct_child(
                self.root,
                source.name,
                staging_root,
                staged.name,
            )
        except FileNotFoundError:
            return False
        if staged.is_symlink() or not staged.is_dir():
            self._unlink_direct_child(
                staging_root,
                staged.name,
                ignore_errors=True,
            )
        else:
            self._rmtree_direct_child(staging_root, staged.name)
        return True

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
        if self._is_immutable_builtin(label):
            raise PermissionError(builtin_skill_refusal(label))
        if not self.allows_skill(label):
            raise ValueError(f"Skill name {label!r} is reserved for MindsHub Code.")
        if self._skill_dir(label).exists():
            raise ValueError(f"A skill named '{label}' already exists.")

        metadata = self._build_metadata(label, name, datetime.now(UTC))
        self._apply_metadata_flags(metadata, enabled, projects)
        skill = Skill(
            name=label,
            instructions=instructions or "",
            # description is required and non-empty by spec; fall back to the
            # display name / slug so we never write an empty value.
            description=(description or "").strip() or name or label,
            metadata=metadata,
        )
        try:
            self._write(skill, create=True)
        except FileExistsError:
            raise ValueError(f"A skill named '{label}' already exists.")
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
        if self._is_immutable_builtin(skill_id):
            raise PermissionError(builtin_skill_refusal(skill_id))
        skill = self.get_skill(skill_id)
        # ``skill_id`` may be a same-root alias, so the reservation is
        # re-checked against the directory this edit would rewrite.
        if self._is_immutable_builtin(skill.name):
            raise PermissionError(builtin_skill_refusal(skill.name))
        original_state = (
            skill.name,
            skill.display_name,
            skill.description,
            skill.instructions,
            skill.enabled,
            skill.projects,
        )
        metadata = dict(skill.metadata)
        self._apply_metadata_flags(metadata, enabled, projects)

        new_slug = skill.name
        if label is not None:
            new_slug = self._slug_from_label(label)
            if self._is_immutable_builtin(new_slug):
                raise PermissionError(builtin_skill_refusal(new_slug))

        if name is not None:
            if name and name != new_slug:
                metadata[META_DISPLAY_NAME] = name
            else:
                metadata.pop(META_DISPLAY_NAME, None)
        if description is not None:
            skill.description = description.strip() or skill.display_name or new_slug
        if instructions is not None:
            skill.instructions = instructions

        renaming = new_slug != skill.name
        if renaming and self._skill_dir(new_slug).exists():
            raise ValueError(f"A skill named '{new_slug}' already exists.")

        candidate = skill.model_copy(deep=True)
        candidate.name = new_slug
        candidate.metadata = metadata
        if (
            candidate.name,
            candidate.display_name,
            candidate.description,
            candidate.instructions,
            candidate.enabled,
            candidate.projects,
        ) == original_state:
            return skill

        # Write the updated content into the current dir first, then rename the
        # whole dir last. A failed _write leaves the old dir intact; the
        # destructive swap only runs once content is safely persisted.
        old_slug = skill.name
        original_skill_file = (self._skill_dir(old_slug) / SKILL_FILE).read_bytes()
        skill.metadata = metadata
        if renaming:
            # Frontmatter and directory identity must change together. Write
            # the new slug into the old directory first, then atomically rename
            # that directory; parsing the result must never return the old id.
            skill.name = new_slug
        self._write(
            skill,
            directory_slug=old_slug,
            reconcile=not renaming,
        )
        if renaming:
            try:
                self._rename_dir(old_slug, new_slug)
            except Exception:
                try:
                    self._restore_skill_file(old_slug, original_skill_file)
                except Exception:
                    logger.exception(
                        "Could not restore %r after its directory rename failed",
                        old_slug,
                    )
                raise
            if self._link_projects:
                remove_skill_links(old_slug)
                reconcile_skill_links(skill)

        return self.get_skill(skill.name)

    def import_skill(
        self,
        data: bytes,
        filename: str | None = None,
        *,
        before_persist: Callable[[str], None] | None = None,
    ) -> Skill:
        """Import a skill from an uploaded file.

        Supported formats (by extension):
          - ``.md`` / ``.skill`` — a text ``SKILL.md``.
          - ``.zip`` — the contents of a skill folder, extracted as-is.

        Validation = "does its ``SKILL.md`` parse via skill_format". Raises
        ``ValueError`` for an unparseable/unsafe file, ``FileExistsError`` on
        slug collision.
        """
        if Path(filename or "").suffix.lower() == ".zip":
            return self._import_zip(data, before_persist=before_persist)
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("File must be UTF-8 encoded text.")
        # parse_skill_dir needs a dir holding a file literally named SKILL.md.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp) / "skill"
            tmp_dir.mkdir(parents=True)
            (tmp_dir / SKILL_FILE).write_text(content, encoding="utf-8")
            return self._persist_imported(
                tmp_dir,
                copy_tree=False,
                before_persist=before_persist,
            )

    def _import_zip(
        self,
        data: bytes,
        *,
        before_persist: Callable[[str], None] | None = None,
    ) -> Skill:
        with tempfile.TemporaryDirectory() as tmp:
            extract_dir = Path(tmp) / "skill"
            extract_dir.mkdir(parents=True)
            self._safe_extract_zip(data, extract_dir)
            return self._persist_imported(
                extract_dir,
                copy_tree=True,
                before_persist=before_persist,
            )

    def _persist_imported(
        self,
        src_dir: Path,
        *,
        copy_tree: bool,
        before_persist: Callable[[str], None] | None = None,
    ) -> Skill:
        """Validate a parsed skill folder and persist it into the canon.

        ``copy_tree`` copies the whole ``src_dir`` (zip: keep sibling files);
        otherwise only ``SKILL.md`` is written by ``_write``.
        """
        self._normalize_skill_dir(src_dir)
        skill = _skill_from_dir(src_dir, canonicalize_name=False)
        if skill is None:
            raise ValueError("Could not find a parseable SKILL.md in the upload.")
        if not skill.name:
            raise ValueError("Skill name is missing or invalid.")
        if self._is_immutable_builtin(skill.name):
            raise PermissionError(builtin_skill_refusal(skill.name))
        if not self.allows_skill(skill.name):
            raise ValueError(
                f"Skill name {skill.name!r} is reserved for MindsHub Code."
            )
        if before_persist is not None:
            before_persist(skill.name)
        if self._skill_dir(skill.name).exists():
            raise FileExistsError(f"A skill named '{skill.name}' already exists.")

        metadata = dict(skill.metadata)
        metadata.setdefault(META_CREATED_AT, datetime.now(UTC).isoformat())
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
                self._rmtree_direct_child(
                    self.root,
                    dest.name,
                    ignore_errors=True,
                )
                raise
        else:
            self._write(skill, create=True)
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
            SkillService._rmtree_direct_child(src_dir, only.name)
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
                        raise ValueError(
                            f"Archive contains a symlink: {info.filename!r}"
                        )
                    target = (dest / info.filename).resolve()
                    if target != dest_resolved and dest_resolved not in target.parents:
                        raise ValueError(f"Unsafe path in archive: {info.filename!r}")
                zf.extractall(dest)
        except zipfile.BadZipFile:
            raise ValueError("Uploaded file is not a valid zip archive.")

    def delete_skill(self, slug: str) -> bool:
        if self._is_immutable_builtin(slug):
            raise PermissionError(builtin_skill_refusal(slug))
        skill_dir = self._skill_dir(slug)
        if not skill_dir.exists():
            return False
        # A same-root alias resolves to another slug's directory, so the
        # reservation is re-checked against the directory actually removed.
        if self._is_immutable_builtin(skill_dir.name):
            raise PermissionError(builtin_skill_refusal(skill_dir.name))
        self._rmtree_direct_child(self.root, skill_dir.name)
        if self._link_projects:
            remove_skill_links(slug)
        return True

    def stage_delete(self, slug: str) -> Path | None:
        """Atomically hide a skill while its deletion audit is committed."""
        if self._is_immutable_builtin(slug):
            raise PermissionError(builtin_skill_refusal(slug))
        skill_dir = self._skill_dir(slug)
        if not skill_dir.exists():
            return None
        # A same-root alias resolves to another slug's directory, so the
        # reservation is re-checked against the directory actually staged.
        if self._is_immutable_builtin(skill_dir.name):
            raise PermissionError(builtin_skill_refusal(skill_dir.name))
        trash = self._delete_staging_root()
        trash.mkdir(parents=True, exist_ok=True)
        staged = trash / f"{skill_dir.name}-{uuid4()}"
        self._rename_direct_child(self.root, skill_dir.name, trash, staged.name)
        return staged

    def restore_staged_delete(self, slug: str, staged: Path) -> None:
        self._rename_direct_child(
            self._delete_staging_root(),
            self._staged_delete_name(staged),
            self.root,
            self._skill_entry(slug).name,
        )

    def finalize_staged_delete(self, staged: Path) -> None:
        self._rmtree_direct_child(
            self._delete_staging_root(),
            self._staged_delete_name(staged),
            ignore_errors=True,
        )

    def _delete_staging_root(self) -> Path:
        return Path(os.path.abspath(self.root / ".delete-staging"))

    def _staged_delete_name(self, staged: Path) -> str:
        """The direct-child name of a staged deletion, refusing a foreign path."""
        absolute = Path(os.path.abspath(staged))
        if absolute.parent != self._delete_staging_root():
            raise ValueError("Invalid staged skill deletion path")
        return absolute.name

    def project_reference_slugs(self, project_name: str) -> list[str]:
        """Canonical skill identities whose metadata names ``project_name``."""
        return sorted(
            skill.name for skill in self.list_skills() if project_name in skill.projects
        )

    def prepare_project_reference_rewrites(
        self,
        old_name: str,
        new_name: str,
        *,
        slugs: list[str] | None = None,
    ) -> list[ProjectReferenceRewrite]:
        """Build exact-byte rollback records without changing the skill store."""
        candidates = (
            slugs if slugs is not None else self.project_reference_slugs(old_name)
        )
        rewrites: list[ProjectReferenceRewrite] = []
        for slug in sorted(set(candidates)):
            skill = self.get_skill(slug)
            if old_name not in skill.projects:
                continue
            metadata = dict(skill.metadata)
            self._apply_metadata_flags(
                metadata,
                enabled=None,
                projects=[
                    new_name if project == old_name else project
                    for project in skill.projects
                ],
            )
            updated_skill = skill.model_copy(deep=True)
            updated_skill.metadata = metadata
            rewrites.append(
                ProjectReferenceRewrite(
                    slug=slug,
                    previous_content=(self._skill_dir(slug) / SKILL_FILE).read_bytes(),
                    updated_skill=updated_skill,
                )
            )
        return rewrites

    def apply_project_reference_rewrites(
        self,
        rewrites: list[ProjectReferenceRewrite],
    ) -> None:
        """Apply a prepared set, restoring every attempted file on failure."""
        attempted: list[ProjectReferenceRewrite] = []
        try:
            for rewrite in rewrites:
                attempted.append(rewrite)
                # Project links are derived desktop state. Reconcile them only
                # after every canonical skill file and the project row commit.
                self._write(rewrite.updated_skill, reconcile=False)
        except Exception:
            self.restore_project_reference_rewrites(attempted)
            raise

    def restore_project_reference_rewrites(
        self,
        rewrites: list[ProjectReferenceRewrite],
    ) -> None:
        """Restore prepared skill files byte-for-byte in reverse write order."""
        for rewrite in reversed(rewrites):
            self._restore_skill_file(rewrite.slug, rewrite.previous_content)

    def finalize_project_reference_rewrites(
        self,
        rewrites: list[ProjectReferenceRewrite],
    ) -> None:
        """Refresh desktop-only derived links after the canonical commit."""
        if not self._link_projects:
            return
        for rewrite in rewrites:
            reconcile_skill_links(self.get_skill(rewrite.slug))

    def replace_project_reference(self, old_name: str, new_name: str) -> int:
        """Repair project-name links after an authorized project rename.

        This is a consistency cascade, not a user-facing skill edit: it must
        update skills regardless of their creator and may repair an immutable
        packaged copy.  Shared-resource ownership/audit therefore stays on the
        project rename that caused it.
        """
        rewrites = self.prepare_project_reference_rewrites(old_name, new_name)
        self.apply_project_reference_rewrites(rewrites)
        self.finalize_project_reference_rewrites(rewrites)
        return len(rewrites)

    # ── low-level fs ─────────────────────────────────────────────────────────
    def _write(
        self,
        skill: Skill,
        *,
        create: bool = False,
        directory_slug: str | None = None,
        reconcile: bool = True,
    ) -> None:
        self._ensure_root()
        skill.metadata[META_UPDATED_AT] = datetime.now(UTC).isoformat()
        skill_dir = self._skill_dir(directory_slug or skill.name)
        skill_dir.mkdir(parents=True, exist_ok=not create)
        target = skill_dir / SKILL_FILE
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{SKILL_FILE}.",
            suffix=".tmp",
            dir=skill_dir,
            text=True,
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(dump_skill(skill))
            self._replace_direct_child(skill_dir, tmp.name, target.name)
        except Exception:
            self._unlink_direct_child(skill_dir, tmp.name, ignore_errors=True)
            if create:
                self._rmtree_direct_child(
                    self.root,
                    skill_dir.name,
                    ignore_errors=True,
                )
            raise
        # Project per-project links to match the skill's metadata (desktop only).
        if self._link_projects and reconcile:
            reconcile_skill_links(skill)

    def _rename_dir(self, old_slug: str, new_slug: str) -> None:
        self._ensure_root()
        self._rename_direct_child(
            self.root,
            self._skill_entry(old_slug).name,
            self.root,
            self._skill_entry(new_slug).name,
        )

    def _restore_skill_file(self, slug: str, content: bytes) -> None:
        """Atomically restore the exact canonical bytes after a failed rename."""
        skill_dir = self._skill_dir(slug)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{SKILL_FILE}.",
            suffix=".restore",
            dir=skill_dir,
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
            self._replace_direct_child(skill_dir, tmp.name, SKILL_FILE)
        except Exception:
            self._unlink_direct_child(skill_dir, tmp.name, ignore_errors=True)
            raise

    # ── builtin seeding ──────────────────────────────────────────────────────
    def _copy_builtin_skill(self, src: Path) -> bool:
        """Copy one packaged builtin without replacing an existing skill."""
        dest = self._skill_dir(src.name)
        if dest.exists():
            return False
        # Copied file by file, each destination re-checked for containment
        # with safe_join. copy2, not copyfile: a packaged executable helper
        # must keep its +x bit.
        for child in sorted(p for p in src.rglob("*") if p.is_file()):
            target = safe_join(dest, *child.relative_to(src).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)
        return True

    def _copy_builtin_skills(self) -> int:
        """Copy packaged builtins into this store, skipping slugs it already has.

        Returns how many were copied. Never overwrites, so a skill the user
        edited or deleted stays as they left it.
        """
        builtin_dir = self.builtin_skills_dir
        if not builtin_dir.exists():
            return 0
        self._ensure_root()
        copied = 0
        for src in sorted(builtin_dir.iterdir()):
            if not src.is_dir() or not (src / SKILL_FILE).exists():
                continue
            if not self.includes_packaged_builtin(src.name):
                continue
            try:
                if not self._copy_builtin_skill(src):
                    continue  # keep the user-editable copy untouched
            except ValueError:
                # Either the slug resolves outside the store, or safe_join rejected a
                # destination. The agent writes into its own org's tree on shared
                # storage, so it can plant a symlink under `dest` between the exists()
                # check above and the write. Skip this builtin: propagating would 500
                # every skills read and every turn for the org, which a tenant must
                # not be able to do to itself.
                logger.warning(
                    "Skipping builtin %r: it does not resolve inside %s",
                    src.name,
                    self.root,
                    exc_info=True,
                )
                continue
            copied += 1
        return copied

    def ensure_builtin_skills(self) -> bool:
        """Seed this product-specific skill store with its packaged builtins.

        Cowork seeds lazily only in org mode; desktop Cowork still seeds once at
        boot via ``migrations.seed_builtin_skills``. Code uses a separate store and
        seeds lazily in both local and org mode so its engineering catalogue never
        depends on, or leaks into, the general Cowork catalogue.

        The version marker is a FILE in the org's store rather than a Setting row,
        so marker and skills share fate:
        - a DB row would outlive a lost volume and leave the org permanently empty,
          because seeding would believe it had already run
        - a deliberate deletion of every skill is still respected, since the marker
          survives it and blocks a re-seed

        Returns True if seeding ran. Fail-soft: a filesystem problem leaves the org
        unseeded and retries on the next read rather than failing the request.
        """
        if (
            not self.seed_builtins_in_local_mode
            and (self._scope is None or not self._scope.org_mode)
        ):
            return False
        try:
            marker = self.root / self.builtin_skills_marker
            current = 0
            if marker.is_file():
                raw = marker.read_text(encoding="utf-8").strip()
                current = int(raw) if raw.isdigit() else 0
            if current >= self.builtin_skills_version:
                return False
            if not self.builtin_skills_dir.exists():
                # Nothing to seed from — a packaging fault, not a seeded org. Writing
                # the marker here would record "done" against an empty store, and the
                # org would stay empty forever once the image is fixed.
                logger.warning(
                    "Builtin skills are missing from this build (%s); not marking %s seeded",
                    self.builtin_skills_dir,
                    self.root,
                )
                return False
            copied = self._copy_builtin_skills()
            marker.write_text(f"{self.builtin_skills_version}\n", encoding="utf-8")
        except OSError:
            logger.warning(
                "Could not seed builtin skills for org %s",
                getattr(self._scope, "org_id", None),
                exc_info=True,
            )
            return False
        logger.info("Seeded %d builtin skill(s) into %s", copied, self.root)
        return True


class CodeSkillService(SkillService):
    """Code-only skill store, isolated from the general Cowork catalogue."""

    store_name = "code-skills"
    seed_builtins_in_local_mode = True

    @property
    def builtin_skills_version(self) -> int:
        return CODE_BUILTIN_SKILLS_VERSION

    @property
    def builtin_skills_marker(self) -> str:
        return CODE_BUILTIN_SKILLS_MARKER

    def allows_skill(self, slug: str) -> bool:
        return True

    def includes_packaged_builtin(self, slug: str) -> bool:
        return slug in CODE_ONLY_BUILTIN_SKILL_NAMES

    @property
    def packaged_builtin_slugs(self) -> frozenset[str]:
        return CODE_ONLY_BUILTIN_SKILL_NAMES

    def ensure_builtin_skills(self) -> bool:
        """Seed Code builtins and repair directories left empty by an interrupted copy.

        A current marker still represents a deliberate deletion when the whole
        skill directory is absent. Only an existing, empty directory is known to
        be an incomplete install, so repairing it does not resurrect deleted or
        user-edited skills.
        """
        repaired = 0
        try:
            for slug in sorted(CODE_ONLY_BUILTIN_SKILL_NAMES):
                src = self.builtin_skills_dir / slug
                dest = self._skill_dir(slug)
                if (
                    src.is_dir()
                    and (src / SKILL_FILE).is_file()
                    and dest.is_dir()
                    and not dest.is_symlink()
                    and next(dest.iterdir(), None) is None
                ):
                    dest.rmdir()
                    repaired += int(self._copy_builtin_skill(src))
        except (OSError, ValueError):
            logger.warning("Could not repair incomplete Code builtin skills in %s",
                           self.root, exc_info=True)
        return super().ensure_builtin_skills() or repaired > 0
