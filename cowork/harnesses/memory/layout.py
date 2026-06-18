"""
This module is responsible for ensuring that the canonical memory files and 
harness runtime symlinks exist.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from cowork.common.settings.app_settings import get_app_settings
from cowork.harnesses.memory.adapter import BaseMemoryAdapter
from cowork.harnesses.memory.registry import SLOT_REGISTRY

logger = logging.getLogger(__name__)


class MemoryLayout:
    def __init__(self, memory_root: Path | None = None) -> None:
        self._memory_root = (
            memory_root.expanduser()
            if memory_root is not None
            else Path(get_app_settings().memory.root_dir).expanduser()
        )

    @property
    def memory_root(self) -> Path:
        return self._memory_root

    def ensure_canonical_files(self) -> Path:
        """Create the canonical memory directory and empty slot files if missing."""
        self._memory_root.mkdir(parents=True, exist_ok=True)

        for meta in SLOT_REGISTRY.values():
            path = self._memory_root / meta.filename
            if not path.exists():
                path.write_text("", encoding="utf-8")

        return self._memory_root

    def _ensure_symlink(self, link: Path, target: Path) -> None:
        """Link a harness runtime path → the canonical slot file.

        Prefers a relative symlink. On Windows, creating a symlink needs
        Administrator or Developer Mode and otherwise raises OSError
        (WinError 1314), which would crash layout setup. When symlinks
        aren't permitted we fall back to a one-time copy so the app still
        starts — but the copy is independent of the canonical file, so that
        harness's memory is NOT shared on such a platform. We log a warning
        to make the degraded state visible (enable Developer Mode for
        sharing). Never clobbers a pre-existing real file.
        """
        link = link.expanduser()
        target = target.expanduser().resolve()

        if link.exists() and not link.is_symlink():
            return

        if link.is_symlink() and link.resolve() == target:
            return

        if link.is_symlink() or link.exists():
            link.unlink()

        link.parent.mkdir(parents=True, exist_ok=True)
        rel = os.path.relpath(target, link.parent)

        try:
            link.symlink_to(rel)
        except OSError as exc:
            logger.warning(
                "symlink %s -> %s failed (%s); copied instead — this harness's "
                "memory will NOT be shared on this platform (enable Developer "
                "Mode on Windows for sharing)", link, target, exc,
            )
            shutil.copyfile(target, link)

    def ensure_layout(self, adapters: list[BaseMemoryAdapter]) -> None:
        """Ensure canonical files exist and create adapter-declared runtime symlinks."""
        memory_dir = self.ensure_canonical_files()

        for adapter in adapters:
            for link_path, slot in adapter.RUNTIME_SYMLINKS.items():
                self._ensure_symlink(link_path, memory_dir / SLOT_REGISTRY[slot].filename)

