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
  (SharedConnections). Each entry names one person's own connection, which
  everyone else reads live from that person's own vault and can't change,
  while its creator still can. ``python -m cowork.ops.connections
  share|unshare`` writes it; the server only reads it, and only for a scope
  that keeps connections personal (reads_shared_registry). A missing file,
  or a missing ``shared/`` folder, shares nothing.

The skills people share are listed outside this folder, in the shared root,
which Cowork writes itself (cowork.shared_skills). Their models live here,
beside the connections'.

create_app refuses to start when the setting names a folder that is missing,
or one of memory/, skills/ and data-vault/ in it, so a missing mount can never
free the reserved names, and when ``shared/connections.json`` is malformed.
"""

from __future__ import annotations

import inspect
import logging
import os
import stat
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Generic, NamedTuple, Self, TypeVar
from uuid import UUID

from anton.core.tools.skill_format import SKILL_FILE, validate_name
from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError, field_validator

from cowork.common.paths import O_NOFOLLOW, O_NONBLOCK, open_fd, read_bounded
from cowork.common.settings.app_settings import get_app_settings

# A shared registry is a short JSON list. A file past this cap is treated as
# malformed rather than read into memory, so a symlink to a huge file or
# /dev/zero can't be read in full.
MAX_REGISTRY_BYTES = 1 << 20  # 1 MiB

logger = logging.getLogger(__name__)

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
    """One person's skill, shared: one entry of the shared-skill list
    (cowork.shared_skills)."""

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


class SharedRegistryUnreadableError(SharedRegistryError):
    """A registry file exists but the filesystem refused to read it (an
    OSError), so nothing about its content is known."""


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

        Opens the file non-blocking and without following a final symlink,
        then checks it is a regular file no larger than MAX_REGISTRY_BYTES
        before reading: a FIFO, a device, a directory or a symlink at ``path``
        never blocks the read or exhausts memory. A non-regular file, a read
        the filesystem refuses, and a symlink all raise
        SharedRegistryUnreadableError; a file past the size cap or one that
        isn't a JSON list of entries raises SharedRegistryError, each naming
        the file and what is wrong with it.
        """
        try:
            fd = open_fd(path, os.O_RDONLY | O_NONBLOCK | O_NOFOLLOW)
        except FileNotFoundError:
            return cls(())
        except OSError as error:  # ELOOP for a symlink, EACCES, ...
            raise SharedRegistryUnreadableError(f"{path} can't be read: {error}") from error
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise SharedRegistryUnreadableError(f"{path} can't be read: not a regular file")
            if info.st_size > MAX_REGISTRY_BYTES:
                raise SharedRegistryError(
                    f"{path} is not a list of shared {cls.entry_noun}s: "
                    f"larger than {MAX_REGISTRY_BYTES} bytes"
                )
            raw = b"".join(read_bounded(fd, info.st_size))
        finally:
            os.close(fd)
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

    def dump_json(self) -> str:
        """The registry file's whole text, as write writes it."""
        return self.model_dump_json(indent=2) + "\n"

    def write(self, path: Path) -> None:
        """Replace the registry file at ``path`` at once, so no reader ever
        reads half of it: a temporary file in the same folder, readable by
        the server's user whatever the writer's umask, flushed to disk and
        moved over it, so a crash never leaves a torn list in its place."""
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.stem}-", suffix=".json", delete=False
        )
        temporary = Path(handle.name)
        try:
            with handle:
                handle.write(self.dump_json())
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o644)
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


class SharedConnections(SharedRegistry[SharedConnection]):
    """The shared-connection registry, ``shared/connections.json``."""

    entry_noun: ClassVar[str] = "connection"
    entry_fields: ClassVar[str] = "user_id, partition_id, engine, name"


class SharedSkills(SharedRegistry[SharedSkill]):
    """The shared-skill list (cowork.shared_skills)."""

    entry_noun: ClassVar[str] = "skill"
    entry_fields: ClassVar[str] = "user_id, partition_id, skill"


def reads_shared_registry(scope: TenantScope | None, *, resource: PersonalResource) -> bool:
    """Whether ``scope`` reads what other people share of ``resource`` from
    its registry, when there is one: only a person's scope that keeps
    ``resource`` personal. Where ``resource`` is shared, its base store
    already holds everyone's, and the operator view reads that."""
    return scope is not None and scope.personal and scope.partitioned_for(resource)


class ManagedSkills(NamedTuple):
    """The skill folders a read-only skill root holds."""

    # The folders a turn can load and the listing shows, by slug: a directory
    # with a readable ``SKILL.md``.
    readable: dict[str, Path]
    # Every slug the root reserves, readable or not. A folder whose ``SKILL.md``
    # the server's user can't stat still reserves its name, so a create under it
    # is refused rather than clobbering the operator's skill.
    reserved: frozenset[str]


def readonly_skill_dirs(root: Path) -> ManagedSkills:
    """Each skill folder anton's SkillStore reads from the read-only skill
    root ``root``, by slug, sorted: every directory holding a ``SKILL.md``
    file, and every slug the root reserves.

    A folder the server's user can't list or stat never crashes a listing, a
    turn or a share change: it is logged with its path and the ``chmod``
    remedy, left out of ``readable``, and kept in ``reserved`` so its name
    stays taken. Empty, with nothing logged, when ``root`` is missing or is
    not a directory."""
    try:
        entries = sorted(root.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return ManagedSkills({}, frozenset())
    except OSError as error:
        logger.error(
            "read-only skill root %s can't be listed (%s); run `chmod -R a+rX` on it", root, error
        )
        return ManagedSkills({}, frozenset())
    readable: dict[str, Path] = {}
    reserved: set[str] = set()
    for entry in entries:
        try:
            if not entry.is_dir():
                continue
            has_skill = (entry / SKILL_FILE).is_file()
        except OSError as error:
            logger.error(
                "skill folder %s can't be read (%s); run `chmod -R a+rX` on it; "
                "its name stays reserved",
                entry,
                error,
            )
            reserved.add(entry.name)
            continue
        if has_skill:
            readable[entry.name] = entry
            reserved.add(entry.name)
    return ManagedSkills(readable, frozenset(reserved))


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

    def skill_dirs(self) -> dict[str, Path]:
        """Each managed skill's readable folder, by slug, sorted
        (readonly_skill_dirs): what a turn can load and the listing shows."""
        return readonly_skill_dirs(self.skills_dir).readable

    def skill_slugs(self) -> frozenset[str]:
        """Every slug the managed folder reserves, readable or not, so an
        unreadable skill's name is still refused for a create."""
        return readonly_skill_dirs(self.skills_dir).reserved


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
