"""The operator's managed folder, COWORK_MANAGED_DIR.

One folder the operator fills from a shell on the host, and Cowork only reads.
Every turn gets what it holds, whoever asked it:

- ``memory/`` holds ``rules.md`` and ``lessons.md``, and an empty
  ``profile.md``. anton's Cortex reads it as a read-only layer ahead of the
  turn's own tiers (``managed_hc``, passed by the anton harness's
  ``_turn_cortex``) and never writes it.
- ``skills/<slug>/SKILL.md`` are skills the anton harness adds to anton's
  read-only skill roots, after its own. SkillService lists them as built-in,
  and their names are reserved: every change to one is refused with
  MANAGED_SKILL_REFUSAL.
- ``data-vault/`` holds the operator's own saved connections, in the vault's
  own file format. ``cowork.services.connectors.persist.vault_for_scope``
  layers them over the reader's own connections (LayeredDataVault), and every
  change to one is refused with
  cowork.services.connectors.layered_vault.SHARED_CONNECTION_REFUSAL.
- ``shared/connections.json`` lists the connections people share
  (SharedConnections): each entry names one person's own saved connection,
  which everyone else reads live from that person's vault and can't change.
  ``python -m cowork.ops.connections share|unshare`` writes it; the server
  only reads it. A missing file, or a missing ``shared/`` folder, shares
  nothing.

create_app refuses to start when the setting names a folder that is missing,
or one of memory/, skills/ and data-vault/ in it, so a missing mount can never
free the reserved names, and when ``shared/connections.json`` is malformed.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from uuid import UUID

from anton.core.tools.skill_format import SKILL_FILE
from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError, field_validator

from cowork.common.settings.app_settings import get_app_settings

MANAGED_SKILL_REFUSAL = "This skill is provided by your operator and can't be changed here."


class SharedConnection(BaseModel):
    """One person's saved connection, shared with everyone on the deployment:
    one entry of ``shared/connections.json``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The person who created it: their sub, and their partition
    # (cowork.principal.personal_partition_id), whose vault holds it.
    user_id: str
    partition_id: str
    engine: str = Field(min_length=1)
    name: str = Field(min_length=1)

    @field_validator("user_id", "partition_id")
    @classmethod
    def _canonical_uuid(cls, value: str) -> str:
        return str(UUID(value))


class SharedConnectionsError(ValueError):
    """``shared/connections.json`` exists but can't be read as SharedConnections."""


class SharedConnections(RootModel[tuple[SharedConnection, ...]]):
    """The shared-connection registry, in file order. For an engine and name
    two people share, everyone else reads the first entry."""

    model_config = ConfigDict(frozen=True)

    @classmethod
    def read(cls, path: Path) -> SharedConnections:
        """The registry at ``path``, or none when the file is missing.

        Raises SharedConnectionsError for a file that can't be read or isn't a
        JSON list of entries, naming the file and what is wrong with it.
        """
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return cls(())
        except OSError as error:
            raise SharedConnectionsError(f"{path} can't be read: {error}") from error
        try:
            return cls.model_validate_json(raw)
        except ValidationError as error:
            raise SharedConnectionsError(
                f"{path} is not a list of shared connections "
                f"(user_id, partition_id, engine, name): {error}"
            ) from error


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
        who shares no connection needs none."""
        return [
            path
            for path in (self.root, self.memory_dir, self.skills_dir, self.vault_dir)
            if not path.is_dir()
        ]

    @property
    def shared_connections_file(self) -> Path:
        return self.shared_dir / "connections.json"

    def shared_connections(self) -> SharedConnections:
        """The shared-connection registry; raises SharedConnectionsError when
        the file is malformed (SharedConnections.read)."""
        return SharedConnections.read(self.shared_connections_file)

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
