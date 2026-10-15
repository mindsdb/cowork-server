"""The operator's managed folder, COWORK_MANAGED_DIR.

One folder the operator fills from a shell on the host, and Cowork only reads.
Every turn gets what it holds, whoever asked it:

- ``memory/`` holds ``rules.md`` and ``lessons.md``, and an empty
  ``profile.md``. anton's Cortex reads it as a read-only layer ahead of the
  turn's own tiers (``managed_hc``, passed by the anton harness's
  ``_turn_cortex``) and never writes it.
- ``skills/<slug>/SKILL.md`` are skills the operator adds, which nobody
  created. The anton harness adds them to anton's read-only skill roots,
  after its own. SkillService lists them as built-in, and their names are
  reserved: every change to one is refused with
  cowork.services.skills.SHARED_SKILL_REFUSAL.
- ``data-vault/`` holds the operator's own saved connections, in the vault's
  own file format. ``cowork.services.connectors.persist.vault_for_scope``
  layers them over the reader's own connections (LayeredDataVault), and every
  change to one is refused with
  cowork.services.connectors.layered_vault.SHARED_CONNECTION_REFUSAL.
- ``shared/connections.json`` lists the connections people share
  (SharedConnections), and ``shared/skills.json`` the skills (SharedSkills).
  Each entry names one person's own connection or skill, which everyone else
  reads live from that person's own store and can't change, while its
  creator still can. ``python -m cowork.ops.connections share|unshare`` and
  ``python -m cowork.ops.skills share|unshare`` write them; the server only
  reads them, and only for a scope that keeps that resource personal
  (reads_shared_registry). A missing file, or a missing ``shared/`` folder,
  shares nothing.

create_app refuses to start when the setting names a folder that is missing,
or one of memory/, skills/ and data-vault/ in it, so a missing mount can never
free the reserved names, and when either registry is malformed.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Generic, Self, TypeVar
from uuid import UUID

from anton.core.tools.skill_format import SKILL_FILE, validate_name
from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError, field_validator

from cowork.common.settings.app_settings import get_app_settings

if TYPE_CHECKING:
    from cowork.db.personal_resources import PersonalResource
    from cowork.db.scoped import TenantScope


class SharedEntry(BaseModel):
    """One registry entry: something one person created and shares with
    everyone on the deployment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The person who created it: their sub, and their partition
    # (cowork.principal.personal_partition_id), whose own store holds it.
    user_id: str
    partition_id: str

    @field_validator("user_id", "partition_id")
    @classmethod
    def _canonical_uuid(cls, value: str) -> str:
        return str(UUID(value))


class SharedConnection(SharedEntry):
    """One person's saved connection, shared: one entry of
    ``shared/connections.json``."""

    engine: str = Field(min_length=1)
    name: str = Field(min_length=1)


class SharedSkill(SharedEntry):
    """One person's skill, shared: one entry of ``shared/skills.json``."""

    # The skill's slug, its folder's name in its creator's store and its only
    # identity. A rename would leave this entry naming nothing, so SkillService
    # refuses its creator's rename while the skill is shared.
    skill: str = Field(min_length=1)

    @field_validator("skill")
    @classmethod
    def _slug(cls, value: str) -> str:
        validate_name(value)
        return value


class SharedRegistryError(ValueError):
    """A registry file exists but can't be read as its registry."""


EntryT = TypeVar("EntryT", bound=SharedEntry)


class SharedRegistry(RootModel[tuple[EntryT, ...]], Generic[EntryT]):
    """A registry of shared entries, in file order. For a name two people
    share, everyone else reads the first entry."""

    model_config = ConfigDict(frozen=True)

    # What one entry is, and its fields, for the errors a broken file raises.
    entry_noun: ClassVar[str]
    entry_fields: ClassVar[str]

    @classmethod
    def read(cls, path: Path) -> Self:
        """The registry at ``path``, or none when the file is missing.

        Raises SharedRegistryError for a file that can't be read or isn't a
        JSON list of entries, naming the file and what is wrong with it.
        """
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return cls(())
        except OSError as error:
            raise SharedRegistryError(f"{path} can't be read: {error}") from error
        try:
            return cls.model_validate_json(raw)
        except ValidationError as error:
            raise SharedRegistryError(
                f"{path} is not a list of shared {cls.entry_noun}s ({cls.entry_fields}): {error}"
            ) from error

    @classmethod
    def read_live(cls, path: Path, *, logger: logging.Logger) -> Self:
        """The registry at ``path``, read for one request or turn.

        A malformed registry shares nothing and logs an error on ``logger``
        naming the file: every request goes on with the reader's own, and
        nobody reads anyone else's. create_app refuses to start on one, so this
        is a registry broken after the server started.
        """
        try:
            return cls.read(path)
        except SharedRegistryError as error:
            logger.error("no %s is shared until the registry is fixed: %s", cls.entry_noun, error)
            return cls(())


class SharedConnections(SharedRegistry[SharedConnection]):
    """The shared-connection registry, ``shared/connections.json``."""

    entry_noun: ClassVar[str] = "connection"
    entry_fields: ClassVar[str] = "user_id, partition_id, engine, name"


class SharedSkills(SharedRegistry[SharedSkill]):
    """The shared-skill registry, ``shared/skills.json``."""

    entry_noun: ClassVar[str] = "skill"
    entry_fields: ClassVar[str] = "user_id, partition_id, skill"


def reads_shared_registry(scope: TenantScope | None, *, resource: PersonalResource) -> bool:
    """Whether ``scope`` reads what other people share of ``resource`` from
    the managed folder's registry, when there is one: only a person's scope
    that keeps ``resource`` personal. Where ``resource`` is shared, its base
    store already holds everyone's, and the operator view reads that."""
    return scope is not None and scope.personal and scope.partitioned_for(resource)


class ManagedFolder(BaseModel):
    """The managed folder and the folders in it."""

    model_config = ConfigDict(frozen=True)

    root: Path
    memory_dir: Path
    skills_dir: Path
    vault_dir: Path
    shared_dir: Path

    @classmethod
    def at(cls, root: Path) -> ManagedFolder:
        return cls(
            root=root,
            memory_dir=root / "memory",
            skills_dir=root / "skills",
            vault_dir=root / "data-vault",
            shared_dir=root / "shared",
        )

    def missing_dirs(self) -> list[Path]:
        """The root and each of memory/, skills/ and data-vault/ that is not a
        directory, in that order. ``shared/`` is not required: an operator
        who shares nothing needs none."""
        return [
            path
            for path in (self.root, self.memory_dir, self.skills_dir, self.vault_dir)
            if not path.is_dir()
        ]

    @property
    def shared_connections_file(self) -> Path:
        return self.shared_dir / "connections.json"

    def shared_connections(self) -> SharedConnections:
        """The shared-connection registry; raises SharedRegistryError when
        the file is malformed (SharedRegistry.read)."""
        return SharedConnections.read(self.shared_connections_file)

    @property
    def shared_skills_file(self) -> Path:
        return self.shared_dir / "skills.json"

    def shared_skills(self) -> SharedSkills:
        """The shared-skill registry; raises SharedRegistryError when the file
        is malformed (SharedRegistry.read)."""
        return SharedSkills.read(self.shared_skills_file)

    def skill_dirs(self) -> dict[str, Path]:
        """Each managed skill's folder, by slug, sorted.

        The folders anton's SkillStore reads from a read-only root: every
        directory holding a ``SKILL.md`` file. So the names reserved here are
        exactly the skills a turn can load from this folder.
        """
        if not self.skills_dir.is_dir():
            return {}
        return {
            entry.name: entry
            for entry in sorted(self.skills_dir.iterdir())
            if entry.is_dir() and (entry / SKILL_FILE).is_file()
        }

    def skill_slugs(self) -> frozenset[str]:
        return frozenset(self.skill_dirs())


def managed_folder() -> ManagedFolder | None:
    """The deployment's managed folder, or None when COWORK_MANAGED_DIR is unset."""
    root = get_app_settings().managed_dir
    return None if root is None else ManagedFolder.at(root)


def cortex_takes_managed_layer() -> bool:
    """Whether the installed anton's Cortex takes ``managed_hc``.

    Cortex is a plain class, so cowork.build_info.supported_kwargs, which reads
    dataclass fields, would drop the argument on every anton. The signature is
    read instead, and create_app refuses a managed folder without it: a
    silently dropped layer would lose the operator's rules on every turn.
    """
    from anton.core.memory.cortex import Cortex

    return "managed_hc" in inspect.signature(Cortex).parameters
