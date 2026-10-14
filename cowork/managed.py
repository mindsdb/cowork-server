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
- ``data-vault/`` holds saved connections in the vault's own file format.
  ``cowork.services.connectors.persist.vault_for_scope`` layers them over the
  reader's own connections (LayeredDataVault), and every change to one is
  refused with MANAGED_CONNECTION_REFUSAL.

create_app refuses to start when the setting names a folder that is missing,
or one of the three folders in it, so a missing mount can never free the
reserved names.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from anton.core.tools.skill_format import SKILL_FILE
from pydantic import BaseModel, ConfigDict

from cowork.common.settings.app_settings import get_app_settings

MANAGED_SKILL_REFUSAL = "This skill is provided by your operator and can't be changed here."
MANAGED_CONNECTION_REFUSAL = (
    "This connection is provided by your operator and can't be changed here."
)


class ManagedConnectionError(PermissionError):
    """A save or a delete reached a connection the managed folder provides.

    The server answers it with 403 and MANAGED_CONNECTION_REFUSAL
    (cowork.server.create_app's exception handler).
    """

    def __init__(self, *, engine: str, name: str) -> None:
        super().__init__(MANAGED_CONNECTION_REFUSAL)
        self.engine = engine
        self.name = name


class ManagedFolder(BaseModel):
    """The managed folder and the three folders in it."""

    model_config = ConfigDict(frozen=True)

    root: Path
    memory_dir: Path
    skills_dir: Path
    vault_dir: Path

    @classmethod
    def at(cls, root: Path) -> ManagedFolder:
        return cls(
            root=root,
            memory_dir=root / "memory",
            skills_dir=root / "skills",
            vault_dir=root / "data-vault",
        )

    def missing_dirs(self) -> list[Path]:
        """The root and each folder in it that is not a directory, in that order."""
        return [
            path
            for path in (self.root, self.memory_dir, self.skills_dir, self.vault_dir)
            if not path.is_dir()
        ]

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
