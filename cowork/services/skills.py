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
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4

from anton.core.tools.skill_format import (
    SKILL_FILE,
    dump_skill,
    normalize_name,
    parse_skill_dir,
    validate_name,
)

from cowork import shared_skills
from cowork.common.paths import (
    dir_rename,
    dir_replace,
    dir_rmtree,
    dir_unlink,
    pinned_dir,
    safe_join,
)
from cowork.common.settings import get_app_settings
from cowork.creators import changer_for
from cowork.db.personal_resources import PersonalResource
from cowork.db.scoped import TenantScope, scope_for_identity, scoped_storage_root
from cowork.managed import (
    ManagedFolder,
    SharedSkill,
    SharedSkills,
    managed_folder,
    reads_shared_registry,
    readonly_skill_dirs,
)
from cowork.models.skill import (
    META_CREATED_AT,
    META_CREATED_BY,
    META_DISPLAY_NAME,
    META_ENABLED,
    META_PROJECTS,
    META_UPDATED_AT,
    Skill,
)
from cowork.services.skill_links import (
    links_apply_to,
    reconcile_skill_links,
    remove_skill_links,
    sync_turn_skill_links,
)
# Re-exported beside the other sharing texts and errors, for the routes and the
# operator command.
from cowork.shared_skills import SKILL_SHARING_BUSY, SharedSkillsBusyError  # noqa: F401

logger = logging.getLogger(__name__)


class SkillNotFoundError(ValueError):
    """A skill is unavailable; remains compatible with ValueError callers."""


class SkillAlreadyExistsError(ValueError):
    """A create or rename conflicts with an existing skill identity."""


class SkillImmutableError(PermissionError):
    """A change to a skill its reader can't change was refused: a built-in, a
    managed or shared skill, or a shared skill's rename. Not a filesystem
    permission failure."""


class SkillSharingError(Exception):
    """A share of one of the owner's skills was refused, with a reason they
    can act on. Neither a ValueError nor a PermissionError, so no route
    handler for those remaps it."""


class SkillSharingUnavailableError(SkillSharingError):
    """Sharing one skill at a time is unavailable for the owner's scope
    (SkillService.sharing_available)."""


# Every change to a skill its reader can't change: one someone else shares
# with them, one of the operator's managed skills, and where skills are shared
# across the deployment, one somebody else created (cowork.creators).
SHARED_SKILL_REFUSAL = "This skill is shared with you and can't be changed here."
# A shared skill's list entry names it by slug, its only identity, so a rename
# would quietly stop sharing it. The person stops sharing it first.
SHARED_SKILL_RENAME_REFUSAL = "Stop sharing this skill before you rename it."
# Where sharing one skill at a time is unavailable: skills left shared, hosted
# org tenancy, the desktop, and a scope with no person. The sentence holds in
# each of those: it names what is unavailable and nothing about who already
# reads the skills.
SKILL_SHARING_UNAVAILABLE = "Sharing one skill at a time isn't available here."


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


@dataclass(frozen=True)
class TurnSkill:
    """One skill a turn may use: its slug and its directory in the store."""

    slug: str
    path: Path


def _used_in(skill: Skill, *, project_name: str | None) -> bool:
    """The selection rule every turn applies to a skill: enabled, and its
    ``metadata.projects`` empty or naming this project."""
    if not skill.enabled:
        return False
    return not skill.projects or (project_name is not None and project_name in skill.projects)


def select_turn_skills(
    *, service: SkillService, project_name: str | None
) -> list[TurnSkill]:
    """The skills of ``service``'s store that a turn in ``project_name`` uses.

    Selection mirrors skill_links: enabled skills whose ``metadata.projects`` is
    empty or names this project.

    Slug and files both come from the DIRECTORY, not frontmatter ``name`` — the
    two can drift on a hand-edited store, and resolving by frontmatter could
    ship another (e.g. disabled) skill's files. Every drop is logged.
    """
    root = service.root
    if not root.exists():
        return []
    selected: list[TurnSkill] = []
    for skill_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        slug = skill_dir.name
        if not service.allows_skill(slug):
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
        if not _used_in(skill, project_name=project_name):
            continue
        selected.append(TurnSkill(slug=slug, path=skill_dir))
    return selected


def shared_turn_skills(
    *, service: SkillService, project_name: str | None
) -> list[TurnSkill]:
    """The skills others share with ``service``'s owner that a turn in
    ``project_name`` uses, by slug.

    SkillService.shared_with_owner's skills, with the selection rule the
    owner's own get (select_turn_skills), applied to each skill's own
    metadata: its creator's disable or project scope reaches everyone. Each
    path is the skill's folder in its creator's own store, read live.
    """
    selected: list[TurnSkill] = []
    for slug, skill_dir in sorted(service.shared_with_owner().items()):
        skill = _skill_from_dir(skill_dir)
        if skill is not None and _used_in(skill, project_name=project_name):
            selected.append(TurnSkill(slug=slug, path=skill_dir))
    return selected


def agent_skill_names() -> frozenset[str]:
    """The skills every person's agent already has from a read-only root
    beside the managed folder: the harness's own (HOST_SKILLS_DIR) and
    anton's packaged ones (anton's SkillStore.builtin_root).

    anton reads the harness's root ahead of the shared skills and its
    packaged ones after them, so a shared skill by one of these names would
    reach nobody or replace anton's for everyone. Neither is shared
    (SkillService._sharing, cowork.ops.skills.share_skill).
    """
    from anton.core.memory.skills import SkillStore

    from cowork.harnesses.anton_harness.harness import HOST_SKILLS_DIR

    return frozenset(
        {*readonly_skill_dirs(HOST_SKILLS_DIR).reserved, *readonly_skill_dirs(SkillStore().builtin_root).reserved}
    )


def reads_shared_skills(scope: TenantScope | None) -> bool:
    """Whether ``scope`` reads the skills other people share with it, from the
    shared-skill list (cowork.shared_skills): a person's scope that keeps
    skills personal (cowork.managed.reads_shared_registry)."""
    return reads_shared_registry(scope, resource=PersonalResource.SKILLS)


def build_turn_skills(
    scope: TenantScope | None, project_path: str | None = None
) -> dict[str, dict]:
    """Skills for a remote turn: ``{slug: {"files": {relpath: text}}}``.

    Org-keyed through the passed `scope` (the producer binds no ambient scope).
    Selection is select_turn_skills's, by this project's folder name.
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
    for skill in select_turn_skills(service=svc, project_name=project_name):
        files = _skill_wire_files(skill.path, svc.root)
        if files is None:
            logger.warning(
                "turn skills: dropping %r (no SKILL.md within size bounds)", skill.slug
            )
            continue
        out[skill.slug] = {"files": files}
    return out


def link_turn_skills(
    *,
    scope: TenantScope | None,
    project_name: str,
    skills_dir: Path,
    shared_skills_dir: Path | None = None,
) -> tuple[Path, ...]:
    """Link the skills one in-process turn uses into ``skills_dir``, from the
    asker's own store, and the ones others share with the asker into
    ``shared_skills_dir``. Returns the folders the links point at, own skills
    first, which the turn's code sandbox grants read-only
    (cowork.code_sandbox.build_cell_sandbox).

    For a turn whose project receives no fanned-out links
    (skill_links.links_apply_to). The packaged builtins are seeded into the
    store first, read-only wherever it is partitioned, as a remote turn's
    staging seeds them (ensure_builtin_skills), so a person's first turn
    already has them. Selection is linked_turn_skills's.

    ``shared_skills_dir`` is for a scope that reads shared skills
    (reads_shared_skills): it takes one link per shared_turn_skills skill,
    each into its creator's own store. The caller hands it to anton as a
    read-only skill root after the managed one. anton writes a skill's usage
    counters only under its own skills root (SkillStore.increment_recommended
    finds the skill there alone), so a recall never writes the creator's
    store, and it takes a name from the first read-only root that has one
    (SkillStore._readonly_roots), so a shared skill never hides a host or
    managed skill of the same name.
    """
    service = SkillService(scope)
    service.ensure_builtin_skills()
    own = {
        skill.slug: skill.path
        for skill in linked_turn_skills(service=service, project_name=project_name)
    }
    sync_turn_skill_links(skills_dir=skills_dir, wanted=own)
    shared: dict[str, Path] = {}
    if shared_skills_dir is not None:
        shared = {
            skill.slug: skill.path
            for skill in shared_turn_skills(service=service, project_name=project_name)
        }
        sync_turn_skill_links(skills_dir=shared_skills_dir, wanted=shared)
    return (*own.values(), *shared.values())


def linked_turn_skills(
    *, service: SkillService, project_name: str | None
) -> list[TurnSkill]:
    """The store's skills link_turn_skills links into a turn's skills folder.

    select_turn_skills's selection, without a slug the managed folder
    reserves, readable or not (SkillService.managed_skill_slugs): anton ranks
    the turn's skills folder above every read-only root, so a linked
    same-named skill would hide the operator's, and the same reserved set
    keeps it out of every fan-out (reconcile_skill_links, reconcile_project).
    """
    reserved = service.managed_skill_slugs()
    return [
        skill
        for skill in select_turn_skills(service=service, project_name=project_name)
        if skill.slug not in reserved
    ]


def _newest_first(skills: list[Skill]) -> list[Skill]:
    return sorted(skills, key=_creation_order, reverse=True)


def _creation_order(skill: Skill) -> tuple[bool, datetime | None]:
    return (skill.created_at is None, skill.created_at)


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


class SkillOrigin(StrEnum):
    """Where a skill its reader sees comes from."""

    MANAGED = "managed"  # the operator's managed folder: nobody created it
    OWN = "own"  # the reader's own store
    SHARED = "shared"  # someone else's own store, shared with the reader


@dataclass(frozen=True)
class VisibleSkill:
    """One skill as its reader lists or reads it."""

    skill: Skill
    origin: SkillOrigin
    # In the shared-skill list: every SHARED skill, and an OWN one its
    # reader shares.
    shared: bool = False


class _Sharing(NamedTuple):
    # The store's own skills its owner shares, by slug.
    by_owner: frozenset[str]
    # Each skill someone else shares that the owner sees, by slug: its folder
    # in its creator's own store.
    with_owner: dict[str, Path]


_NOTHING_SHARED = _Sharing(by_owner=frozenset(), with_owner={})


def _shared_skill_dir(entry: SharedSkill, *, strict: bool = False) -> Path | None:
    """The folder of the skill ``entry`` shares, in its creator's own store,
    or None when that store holds no complete skill by that name. ``strict``
    re-raises a read error other than a missing path (has_complete_skill),
    so a transient failure never reads as "gone"."""
    creator = SkillService(scope_for_identity(org_id=entry.partition_id, user_id=entry.user_id))
    if not creator.allows_skill(entry.skill) or not creator.has_complete_skill(entry.skill, strict=strict):
        return None
    return creator._skill_dir(entry.skill)


@dataclass(frozen=True)
class ProjectReferenceRewrite:
    """One exact, reversible skill edit caused by a project rename."""

    slug: str
    previous_content: bytes
    updated_skill: Skill


class SkillService:
    """File-backed skill store using the agentskills.io ``SKILL.md`` format.

    Org mode keys the store per org (``<shared_root>/<org_id>/skills``), and a
    personal scope that keeps skills per person keys it per partition the same
    way; local mode uses the shared root unchanged. Wherever the store is
    partitioned, the packaged builtins are seeded into it lazily and stay
    immutable.

    On a deployment that trusts identity headers, every skill created here
    records its creator in ``metadata.created_by`` (META_CREATED_BY): the
    person's sub, never a value the request or upload carries, and nobody for
    the operator view. An edit, rename or project-rename rewrite keeps it.
    Where skills are personal, a person shares or hides each of their own
    skills (share, stop_sharing, or a create with ``shared``), and reads the
    skills others share with them (shared_with_owner), live from each
    creator's own store and read-only; the creator still changes theirs,
    except its name while it is shared. Where skills are shared, a person
    changes only a skill they created, and the operator view any
    (may_change). All of it is the general store's alone: the Code store
    records no creator and is never in the shared-skill list. With skills
    left shared it is one store with no creator rule, which only a caller on
    the server's own host reaches: its routes are LoopbackDesktopOnly.
    """

    store_name = "skills"
    seed_builtins_in_local_mode = False
    immutable_builtins_in_local_mode = False

    def __init__(self, scope: TenantScope | None = None) -> None:
        settings = get_app_settings()
        self._scope = scope
        base = Path(settings.skill.root_dir)
        if self.store_name != "skills":
            base = base.parent / self.store_name
        self.root = scoped_storage_root(base, scope, store=self.store_name)
        # Symlink distribution resolves the unkeyed root and scans all project
        # dirs, so it runs only where skills and projects stay unpartitioned
        # (links_apply_to). Keyed on deployment mode too, not just scope — an
        # unscoped service (migration, seeding) must not fan symlinks out of
        # the unkeyed root in org mode either.
        self._link_projects = (
            self.store_name == "skills"
            and settings.tenancy_mode != "org"
            and links_apply_to(scope)
        )
        # Who creates and changes this store's skills, where creators are
        # recorded (cowork.creators).
        self._changer = (
            changer_for(scope, resource=PersonalResource.SKILLS)
            if self.store_name == "skills"
            else None
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
        protected = self.immutable_builtins_in_local_mode or (
            self._scope is not None
            and self._scope.partitioned_for(PersonalResource.SKILLS)
        )
        return bool(protected and slug in self.packaged_builtin_slugs)

    def _managed_for_store(self) -> ManagedFolder | None:
        """The operator's managed folder where this store sits beside it: the
        general catalogue, never the Code store. None without a managed folder
        (cowork.managed)."""
        if self.store_name != "skills":
            return None
        return managed_folder()

    def managed_skill_dirs(self) -> dict[str, Path]:
        """The operator's readable managed skills beside this store, by slug
        (ManagedFolder.skill_dirs): what the listing shows."""
        managed = self._managed_for_store()
        return {} if managed is None else managed.skill_dirs()

    def managed_skill_slugs(self) -> frozenset[str]:
        """Every slug the managed folder beside this store reserves, readable
        or not (ManagedFolder.skill_slugs): an unreadable managed skill keeps
        its name, so a create under it is refused rather than clobbering the
        operator's skill, and a same-named skill of the owner's is never
        linked over it."""
        managed = self._managed_for_store()
        return frozenset() if managed is None else managed.skill_slugs()

    def is_managed_skill(self, slug: str) -> bool:
        return slug in self.managed_skill_slugs()

    @cached_property
    def _sharing(self) -> _Sharing:
        """What this store's owner shares and is shared, read live from the
        shared-skill list (cowork.shared_skills); nothing for a scope that
        reads no shared skills (reads_shared_skills) or another store.
        Read once per instance, which is one request or turn (_registry);
        after this instance changes the list, built again from the list the
        change left (_forget_sharing).

        A skill someone else shares is skipped when its creator's store no
        longer holds it, and hidden by a same-named skill earlier in the
        owner's order: a packaged built-in, one the agent already has
        (agent_skill_names), a managed skill, then any entry of the owner's
        own store. For a name two people share, the first entry wins. A
        malformed list shares nothing and logs a warning naming the file
        (read_shared_skills).
        """
        if self.store_name != "skills" or not reads_shared_skills(self._scope):
            return _NOTHING_SHARED
        registry = self._registry.root
        partition = self._scope.org_id
        managed = managed_folder()
        taken = {
            *self.packaged_builtin_slugs,
            *agent_skill_names(),
            *(() if managed is None else managed.skill_slugs()),
            *self._own_entry_names(),
        }
        with_owner: dict[str, Path] = {}
        for entry in registry:
            if entry.partition_id == partition or entry.skill in taken or entry.skill in with_owner:
                continue
            skill_dir = _shared_skill_dir(entry)
            if skill_dir is not None:
                with_owner[entry.skill] = skill_dir
        return _Sharing(
            by_owner=frozenset(entry.skill for entry in registry if entry.partition_id == partition),
            with_owner=with_owner,
        )

    def _own_entry_names(self) -> set[str]:
        return {entry.name for entry in self.root.iterdir()} if self.root.is_dir() else set()

    @cached_property
    def _registry(self) -> SharedSkills:
        """The shared-skill list as this instance last saw it: read once
        (read_shared_skills), or the list this instance's last change left."""
        return shared_skills.read_shared_skills()

    def _forget_sharing(self, *, left: SharedSkills | None = None) -> None:
        """Drop the cached _sharing after a change to the list, so this
        instance's next read sees the change. ``left`` is the list the change
        left, read under the list's lock, so it holds every other writer's
        change too, and the next read takes it rather than the file; without
        it, the next read reads the file again."""
        self.__dict__.pop("_sharing", None)
        if left is None:
            self.__dict__.pop("_registry", None)
        else:
            self.__dict__["_registry"] = left

    def shared_with_owner(self) -> dict[str, Path]:
        """Each skill someone else shares with this store's owner that they
        see, by slug: its folder in its creator's own store (_sharing)."""
        return self._sharing.with_owner

    def shared_by_owner(self) -> frozenset[str]:
        """The slugs of this store's own skills that its owner shares."""
        return self._sharing.by_owner

    def refuse_unchangeable(self, *slugs: str, creating: bool = False) -> None:
        """Refuse a change that would write, rename into or remove any of
        ``slugs`` when the owner can't change that name: a slug the managed
        folder holds, in every mode and scope, or one someone else shares with
        the owner (shared_with_owner). Both answer SHARED_SKILL_REFUSAL. Needs
        no read of the owner's own store, which may not hold the name.

        ``creating`` is for a create, upload or rename into ``slugs``, where
        the caller is making a new skill of their own rather than changing a
        skill someone shares: a name someone else shares then answers
        SkillAlreadyExistsError, which names the way out, a different name. A
        managed name keeps SHARED_SKILL_REFUSAL even on a create, since nobody
        may hold the operator's reserved name.
        """
        if any(self.is_managed_skill(slug) for slug in slugs):
            raise SkillImmutableError(SHARED_SKILL_REFUSAL)
        shared = self.shared_with_owner()
        taken = [slug for slug in slugs if slug in shared]
        if taken:
            if creating:
                raise SkillAlreadyExistsError(
                    f"Someone else shares a skill named '{taken[0]}'. Choose another name."
                )
            raise SkillImmutableError(SHARED_SKILL_REFUSAL)

    def may_change(self, skill: Skill) -> bool:
        """Whether the owner may change ``skill``, one of the store's own:
        always, except in the deployment's base store every person reads
        (skills left shared on a deployment that trusts identity headers),
        where a person may change only a skill they created and the operator
        view any (cowork.creators.Changer.may_change)."""
        return self._changer is None or self._changer.may_change(skill.created_by)

    def _refuse_unless_changeable(self, skill: Skill | None) -> None:
        """Refuse a change the owner may not make to one of the store's own
        skills (may_change); ``None``, a folder with no parseable SKILL.md,
        records no creator."""
        creator = None if skill is None else skill.created_by
        if self._changer is not None and not self._changer.may_change(creator):
            raise SkillImmutableError(SHARED_SKILL_REFUSAL)

    def _stamp_creator(self, metadata: dict[str, str]) -> None:
        """Record in ``metadata`` who creates the skill, where creators are
        recorded: the changer's (cowork.creators.Changer.creator), never a
        value the caller passed."""
        if self._changer is None:
            return
        metadata.pop(META_CREATED_BY, None)
        if self._changer.creator is not None:
            metadata[META_CREATED_BY] = self._changer.creator

    # ── sharing one skill ────────────────────────────────────────────────────
    @property
    def sharing_available(self) -> bool:
        """Whether this store's owner may share their own skills one at a
        time: a person's scope, with their sub, that keeps skills personal.
        A personal scope already means local tenancy that trusts identity
        headers. Never for hosted org tenancy, the desktop, the operator
        view, skills left shared or another store."""
        scope = self._scope
        return (
            self.store_name == "skills"
            and scope is not None
            and scope.personal
            and scope.user_id is not None
            and scope.partitioned_for(PersonalResource.SKILLS)
        )

    def refuse_unless_sharing_available(self) -> None:
        """Answers SKILL_SHARING_UNAVAILABLE where sharing_available is false."""
        if not self.sharing_available:
            raise SkillSharingUnavailableError(SKILL_SHARING_UNAVAILABLE)

    def _refuse_shared_name(self, slug: str) -> None:
        """Refuse to share ``slug`` when everyone already has a skill by that
        name ahead of a shared one: a packaged built-in, one the agent
        already has (agent_skill_names), or a managed skill. A share of
        any of them would reach nobody, or replace anton's own for
        everyone."""
        if is_builtin_skill(slug):
            raise SkillSharingError(
                f"{slug} is a built-in skill that everyone already has, so it can't be shared."
            )
        if slug in agent_skill_names():
            raise SkillSharingError(
                f"Everyone's agent already has a skill named {slug}, so sharing yours would "
                "reach nobody. Rename your skill before you share it."
            )
        if self.is_managed_skill(slug):
            raise SkillSharingError(
                f"A built-in skill is already named {slug}, so sharing yours would reach "
                "nobody. Rename your skill before you share it."
            )

    def _refuse_links(self, slug: str) -> None:
        """Refuse to share a skill that holds a link: everyone would read
        whatever it points at. Walks the folder without following one."""
        skill_dir = self._skill_entry(slug)
        for directory, subdirectories, files in os.walk(skill_dir):
            for name in [*subdirectories, *files]:
                entry = Path(directory) / name
                if entry.is_symlink():
                    raise SkillSharingError(
                        f"This skill holds a link, {entry.relative_to(skill_dir).as_posix()}. "
                        "Remove it before you share the skill."
                    )

    def share(self, slug: str) -> bool:
        """Share the owner's own skill ``slug`` with everyone, read-only.

        Refused, in this order: where sharing is unavailable
        (SkillSharingUnavailableError), for a name everyone already has
        (_refuse_shared_name), for a skill the store doesn't hold
        (SkillNotFoundError), for one it can't read in full, for one holding
        a link, and for a name someone else already shares a live skill
        under (_add_own_entry). Returns whether it added an entry: False when
        the skill is already shared, with nothing written.
        """
        self.refuse_unless_sharing_available()
        self._refuse_shared_name(slug)
        self.get_skill(slug)
        if not self.has_complete_skill(slug):
            raise SkillSharingError(f"Skill {slug} can't be read in full, so it can't be shared.")
        self._refuse_links(slug)
        return self._add_own_entry(slug)

    def stop_sharing(self, slug: str) -> bool:
        """Remove every entry of the owner's for ``slug`` from the list, and
        return whether there was one. Needs no skill folder: an entry left
        by a skill that is gone is removed too."""
        self.refuse_unless_sharing_available()
        partition = self._scope.org_id
        removed = False

        def without_own_entry(current: SharedSkills) -> SharedSkills:
            nonlocal removed
            kept = tuple(
                entry
                for entry in current.root
                if not (entry.partition_id == partition and entry.skill == slug)
            )
            removed = len(kept) != len(current.root)
            return SharedSkills(kept)

        try:
            left = shared_skills.change_shared_skills(change=without_own_entry)
        except BaseException:
            self._forget_sharing()
            raise
        self._forget_sharing(left=left)
        return removed

    def _add_own_entry(self, slug: str) -> bool:
        """Add the owner's entry for ``slug`` under the list's lock, and
        return whether it was added: False when the owner already shares it.
        Refused when someone else already shares a live skill by that name,
        since everyone reads the first; an entry whose skill is gone doesn't
        count."""
        scope = self._scope
        added = False

        def with_own_entry(current: SharedSkills) -> SharedSkills:
            nonlocal added
            # Scan every entry for the slug before deciding: a dead own entry
            # listed ahead of someone else's live share must not short-circuit
            # the live-share check, or a duplicate would slip through.
            own_exists = False
            for entry in current.root:
                if entry.skill != slug:
                    continue
                if entry.partition_id == scope.org_id:
                    own_exists = True
                    continue
                try:
                    live = _shared_skill_dir(entry, strict=True) is not None
                except OSError as error:
                    raise SharedSkillsBusyError(SKILL_SHARING_BUSY) from error
                if live:
                    raise SkillSharingError(
                        f"Someone else already shares a skill named {slug}. Rename your "
                        "skill before you share it."
                    )
            if own_exists:
                return current
            added = True
            own = SharedSkill(user_id=scope.user_id, partition_id=scope.org_id, skill=slug)
            return SharedSkills((*current.root, own))

        try:
            left = shared_skills.change_shared_skills(change=with_own_entry)
        except BaseException:
            self._forget_sharing()
            raise
        self._forget_sharing(left=left)
        return added

    def _refuse_unless_shareable_on_create(self, slug: str) -> None:
        """The checks a skill created shared passes before any file is
        written: sharing is available, and its name is one a share reaches
        (_refuse_shared_name)."""
        self.refuse_unless_sharing_available()
        self._refuse_shared_name(slug)

    def _clear_leftover_entry(self, slug: str) -> None:
        """Before a new skill of the owner's is written at ``slug``, which the
        store doesn't hold, remove the owner's entry the list still has for
        a skill of theirs by that name that is gone, so the new skill is
        never shared unless they chose it.

        Reads under the list's lock (stop_sharing), not the forgiving cached
        _sharing: a list the filesystem can't read answers busy here, rather
        than letting a leftover entry survive and silently share the new
        skill once the read recovers. stop_sharing writes nothing when no
        entry matches, unless the list is malformed, which it replaces.
        Skipped when the list file doesn't exist, so a create where nobody
        ever shared anything never creates the shared folder."""
        if self.sharing_available and shared_skills.shared_skills_file().exists():
            self.stop_sharing(slug)

    def _share_new_skill(self, slug: str) -> None:
        """Share the skill a create just wrote at ``slug``. When adding the
        entry fails, for any reason, the new folder is removed first: no
        skill created shared is ever left behind unshared."""
        try:
            self._add_own_entry(slug)
        except Exception:
            self._rmtree_direct_child(self.root, slug, ignore_errors=True)
            raise

    def as_own(self, skill: Skill) -> VisibleSkill:
        """``skill``, one of the store's own, as its owner sees it."""
        return VisibleSkill(
            skill=skill, origin=SkillOrigin.OWN, shared=skill.name in self.shared_by_owner()
        )

    def _refuse_if_reserved(self, slug: str, *, creating: bool = False) -> None:
        """Refuse a change that would write, rename into or remove ``slug``.

        A slug the owner can't change, managed or shared with them, is refused
        with SHARED_SKILL_REFUSAL (refuse_unchangeable); on a create, upload or
        rename into ``slug`` (``creating``), a name someone else shares answers
        SkillAlreadyExistsError instead. A packaged builtin slug is refused
        wherever builtins are immutable (_is_immutable_builtin).
        """
        self.refuse_unchangeable(slug, creating=creating)
        if self._is_immutable_builtin(slug):
            raise SkillImmutableError(builtin_skill_refusal(slug))

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
        return _newest_first(skills)

    def _managed_skills(self) -> dict[str, Skill]:
        managed: dict[str, Skill] = {}
        for slug, skill_dir in self.managed_skill_dirs().items():
            skill = _skill_from_dir(skill_dir, canonicalize_name=True)
            if skill is not None and self.allows_skill(slug):
                managed[slug] = skill
        return managed

    def list_visible_skills(self) -> list[VisibleSkill]:
        """The skills this store's owner sees, newest first: the managed
        skills, the store's own except one whose slug a managed skill holds,
        and the ones others share with the owner (shared_with_owner).

        For the skills listing only. list_skills and get_skill stay store-only,
        so the project-rename cascade and the link reconciliation, which write
        what they read back into the store, never reach a managed or shared
        skill.
        """
        managed = self._managed_skills()
        sharing = self._sharing
        visible = [
            VisibleSkill(skill=skill, origin=SkillOrigin.MANAGED) for skill in managed.values()
        ]
        visible.extend(
            VisibleSkill(skill=skill, origin=SkillOrigin.OWN, shared=skill.name in sharing.by_owner)
            for skill in self.list_skills()
            if skill.name not in managed
        )
        for skill_dir in sharing.with_owner.values():
            skill = _skill_from_dir(skill_dir)
            if skill is not None:
                visible.append(VisibleSkill(skill=skill, origin=SkillOrigin.SHARED, shared=True))
        return sorted(visible, key=lambda seen: _creation_order(seen.skill), reverse=True)

    def get_visible_skill(self, slug: str) -> VisibleSkill:
        """The skill ``slug`` names for its owner: the managed one when the
        managed folder holds it, one someone else shares with them, else the
        store's own (get_skill)."""
        managed = self._managed_skills().get(slug)
        if managed is not None:
            return VisibleSkill(skill=managed, origin=SkillOrigin.MANAGED)
        shared_dir = self.shared_with_owner().get(slug)
        shared = None if shared_dir is None else _skill_from_dir(shared_dir)
        if shared is not None:
            return VisibleSkill(skill=shared, origin=SkillOrigin.SHARED, shared=True)
        return self.as_own(self.get_skill(slug))

    def get_skill(self, slug: str) -> Skill:
        if not self.allows_skill(slug):
            raise SkillNotFoundError(f"Skill {slug!r} not found.")
        skill_dir = self._skill_dir(slug)
        skill = (
            _skill_from_dir(skill_dir, canonicalize_name=True)
            if (skill_dir / SKILL_FILE).exists()
            else None
        )
        if skill is None:
            raise SkillNotFoundError(f"Skill {slug!r} not found.")
        return skill

    def has_complete_skill(self, slug: str, *, strict: bool = False) -> bool:
        """Whether ``slug`` has a regular, parseable canonical skill file.

        A pending create may outlive the process that reserved it.  The directory
        alone is not evidence that the filesystem mutation survived: ``_write``
        creates it before replacing ``SKILL.md``.  Recovery must therefore reject
        empty directories, temporary files, symlinks, malformed frontmatter, and
        a file whose declared identity does not match its directory.

        ``strict`` separates "the skill is gone" from "the filesystem couldn't
        answer": a read error other than a missing path then raises instead of
        reading as absent, so the locked duplicate-share check never counts a
        live skill it merely failed to read as gone.
        """
        skill_dir = self._skill_entry(slug)
        skill_file = skill_dir / SKILL_FILE
        try:
            if skill_dir.is_symlink() or not skill_dir.is_dir():
                return False
            if skill_file.is_symlink() or not skill_file.is_file():
                return False
            parsed = _skill_from_dir(skill_dir, canonicalize_name=False)
        except (UnicodeError, ValueError):
            return False
        except FileNotFoundError:
            return False
        except OSError:
            if strict:
                raise
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
        *,
        shared: bool = False,
    ) -> Skill:
        """Create a skill, shared with everyone, read-only, when ``shared``.

        A shared create runs every share check that needs no file
        (_refuse_unless_shareable_on_create) before writing anything, and
        removes the new folder when adding its entry then fails
        (_share_new_skill). A create that isn't shared first removes an
        entry the owner's list still has for a gone skill by that name.
        """
        label = self._slug_from_label(label)
        self._refuse_if_reserved(label, creating=True)
        if not self.allows_skill(label):
            raise ValueError(f"Skill name {label!r} is reserved for MindsHub Code.")
        if self._skill_dir(label).exists():
            raise SkillAlreadyExistsError(f"A skill named '{label}' already exists.")
        if shared:
            self._refuse_unless_shareable_on_create(label)
        else:
            self._clear_leftover_entry(label)

        metadata = self._build_metadata(label, name, datetime.now(UTC))
        self._apply_metadata_flags(metadata, enabled, projects)
        self._stamp_creator(metadata)
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
            raise SkillAlreadyExistsError(f"A skill named '{label}' already exists.")
        if shared:
            self._share_new_skill(label)
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
        self._refuse_if_reserved(skill_id)
        skill = self.get_skill(skill_id)
        # ``skill_id`` may be a same-root alias, so the reservation is
        # re-checked against the directory this edit would rewrite.
        self._refuse_if_reserved(skill.name)
        self._refuse_unless_changeable(skill)
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
            self._refuse_if_reserved(new_slug, creating=new_slug != skill.name)
            if new_slug != skill.name and skill.name in self.shared_by_owner():
                raise SkillImmutableError(SHARED_SKILL_RENAME_REFUSAL)

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
            raise SkillAlreadyExistsError(f"A skill named '{new_slug}' already exists.")
        if renaming:
            self._clear_leftover_entry(new_slug)

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
        validate_skill: Callable[[Skill], None] | None = None,
        shared: bool = False,
    ) -> Skill:
        """Import a skill from an uploaded file, shared with everyone,
        read-only, when ``shared`` (as create_skill shares one).

        Supported formats (by extension):
          - ``.md`` / ``.skill`` — a text ``SKILL.md``.
          - ``.zip`` — the contents of a skill folder, extracted as-is.

        Validation = "does its ``SKILL.md`` parse via skill_format". Raises
        ``ValueError`` for an unparseable/unsafe file, ``FileExistsError`` on
        slug collision.
        ``validate_skill`` can enforce a caller's edit constraints on the parsed
        skill before any files are persisted; other callers remain lenient.
        """
        if Path(filename or "").suffix.lower() == ".zip":
            return self._import_zip(
                data,
                before_persist=before_persist,
                validate_skill=validate_skill,
                shared=shared,
            )
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
                validate_skill=validate_skill,
                shared=shared,
            )

    def _import_zip(
        self,
        data: bytes,
        *,
        before_persist: Callable[[str], None] | None = None,
        validate_skill: Callable[[Skill], None] | None = None,
        shared: bool = False,
    ) -> Skill:
        with tempfile.TemporaryDirectory() as tmp:
            extract_dir = Path(tmp) / "skill"
            extract_dir.mkdir(parents=True)
            self._safe_extract_zip(data, extract_dir)
            return self._persist_imported(
                extract_dir,
                copy_tree=True,
                before_persist=before_persist,
                validate_skill=validate_skill,
                shared=shared,
            )

    def _persist_imported(
        self,
        src_dir: Path,
        *,
        copy_tree: bool,
        before_persist: Callable[[str], None] | None = None,
        validate_skill: Callable[[Skill], None] | None = None,
        shared: bool = False,
    ) -> Skill:
        """Validate a parsed skill folder and persist it into the canon.

        ``copy_tree`` copies the whole ``src_dir`` (zip: keep sibling files);
        otherwise only ``SKILL.md`` is written by ``_write``. ``shared`` runs
        the share checks that need no file before ``before_persist``, so a
        refused share reserves nothing, and shares the skill once written.
        """
        self._normalize_skill_dir(src_dir)
        skill = _skill_from_dir(src_dir, canonicalize_name=False)
        if skill is None:
            raise ValueError("Could not find a parseable SKILL.md in the upload.")
        if not skill.name:
            raise ValueError("Skill name is missing or invalid.")
        self._refuse_if_reserved(skill.name, creating=True)
        if not self.allows_skill(skill.name):
            raise ValueError(
                f"Skill name {skill.name!r} is reserved for MindsHub Code."
            )
        if shared:
            self._refuse_unless_shareable_on_create(skill.name)
        if before_persist is not None:
            before_persist(skill.name)
        if self._skill_dir(skill.name).exists():
            raise FileExistsError(f"A skill named '{skill.name}' already exists.")
        if not shared:
            self._clear_leftover_entry(skill.name)

        metadata = dict(skill.metadata)
        metadata.setdefault(META_CREATED_AT, datetime.now(UTC).isoformat())
        metadata.pop(META_PROJECTS, None)
        self._stamp_creator(metadata)
        skill.metadata = metadata
        if not skill.description.strip():
            skill.description = skill.display_name or skill.name
        if validate_skill is not None:
            validate_skill(skill)

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
        if shared:
            self._share_new_skill(skill.name)
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
        self._refuse_if_reserved(slug)
        skill_dir = self._skill_dir(slug)
        if not skill_dir.exists():
            return False
        # A same-root alias resolves to another slug's directory, so the
        # reservation is re-checked against the directory actually removed.
        self._refuse_if_reserved(skill_dir.name)
        self._refuse_unless_changeable(_skill_from_dir(skill_dir))
        self._rmtree_direct_child(self.root, skill_dir.name)
        if self._link_projects:
            remove_skill_links(slug)
        if self.sharing_available and shared_skills.shared_skills_file().exists():
            # The folder goes first. An entry left behind shares nothing:
            # readers skip an entry whose skill is gone, and the owner's next
            # create, upload or rename into the name removes it. Cleanup is
            # best-effort and always attempted where sharing is available, so
            # a list the filesystem can't read logs a possible leftover entry.
            try:
                self.stop_sharing(skill_dir.name)
            except Exception:
                logger.warning(
                    "couldn't confirm deleted skill %s is no longer listed as shared in %s",
                    skill_dir.name,
                    shared_skills.shared_skills_file(),
                    exc_info=True,
                )
        return True

    def stage_delete(self, slug: str) -> Path | None:
        """Atomically hide a skill while its deletion audit is committed."""
        self._refuse_if_reserved(slug)
        skill_dir = self._skill_dir(slug)
        if not skill_dir.exists():
            return None
        # A same-root alias resolves to another slug's directory, so the
        # reservation is re-checked against the directory actually staged.
        self._refuse_if_reserved(skill_dir.name)
        self._refuse_unless_changeable(_skill_from_dir(skill_dir))
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

        Cowork seeds lazily wherever the store is partitioned (org mode, and a
        personal scope that keeps skills per person); desktop Cowork, and a
        shared store, still seed once at boot via
        ``migrations.seed_builtin_skills``. Code uses a separate store and
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
            and (
                self._scope is None
                or not self._scope.partitioned_for(PersonalResource.SKILLS)
            )
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
    immutable_builtins_in_local_mode = True

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
