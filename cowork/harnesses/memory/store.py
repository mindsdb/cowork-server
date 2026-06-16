"""
This module defines the shared memory store for performing operations on the memory files.
"""

from pathlib import Path

from cowork.common.settings.app_settings import get_app_settings
from cowork.harnesses.memory.registry import MemorySlot, SLOT_REGISTRY


class SharedMemoryStore:
    def __init__(self, root: Path | None = None) -> None:
        self._root = root or Path(get_app_settings().memory.root_dir).expanduser()

    def _path_for(self, slot_id: MemorySlot | str) -> Path:
        slot_id = MemorySlot(slot_id) if isinstance(slot_id, str) else slot_id
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

    def list_slots(self) -> list[MemorySlot]:
        return list(SLOT_REGISTRY.keys())
