"""
This module defines the memory stores for canonical slot files on disk.
"""

from pathlib import Path

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import TenantScope, scoped_user_storage_root
from cowork.harnesses.memory.registry import MemorySlot, SLOT_REGISTRY

PROJECT_SLOTS = (MemorySlot.RULES, MemorySlot.LESSONS)


class MemoryStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        """Resolved slot directory — the same layout anton's Hippocampus reads."""
        return self._root

    def _validate_slot(self, slot_id: MemorySlot) -> None:
        return

    def _path_for(self, slot_id: MemorySlot | str) -> Path:
        slot_id = MemorySlot(slot_id) if isinstance(slot_id, str) else slot_id
        self._validate_slot(slot_id)
        return self._root / SLOT_REGISTRY[slot_id].filename

    def read(self, slot_id: MemorySlot | str) -> str:
        path = self._path_for(slot_id)
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8")

    def write(self, slot_id: MemorySlot | str, content: str) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        self._path_for(slot_id).write_text(content.rstrip() + "\n", encoding="utf-8")

    def delete(self, slot_id: MemorySlot | str) -> None:
        path = self._path_for(slot_id)
        if path.is_file():
            path.unlink()


class GlobalMemoryStore(MemoryStore):
    """The ``global`` memory scope: identity, personal rules, cross-project lessons.

    Per-(org, user) in org mode, not per-org: anton overwrites identity by key, so
    a shared tier would let one member's turn replace another's (ADR-0002).
    """

    def __init__(self, root: Path | None = None, scope: TenantScope | None = None) -> None:
        base = (
            root.expanduser()
            if root is not None
            else Path(get_app_settings().memory.root_dir).expanduser()
        )
        super().__init__(scoped_user_storage_root(base, scope))

    def list_slots(self) -> list[MemorySlot]:
        return list(SLOT_REGISTRY.keys())


class ProjectMemoryStore(MemoryStore):
    """The ``project`` memory scope: rules and lessons shared by everyone in the
    org who works in this project."""

    def __init__(self, project_path: Path) -> None:
        super().__init__(Path(project_path) / ".anton" / "memory")

    def _validate_slot(self, slot_id: MemorySlot) -> None:
        if slot_id not in PROJECT_SLOTS:
            raise ValueError(f"{slot_id.value} is not supported for project-scoped memory.")
