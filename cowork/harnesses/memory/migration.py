"""
This module performs a one-time migration of harness-local memory files into 
the shared canonical store.
At the moment this was written, only Anton and Hermes were supported as harnesses.
As a result, only the memory files for these two harnesses are migrated.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlmodel import Session, select

from cowork.harnesses.memory.registry import MemorySlot
from cowork.harnesses.memory.store import SharedMemoryStore
from cowork.models.setting import Setting


logger = logging.getLogger(__name__)

_MEMORY_MIGRATION_SENTINEL = "_memory_migrated"

_MIGRATION_SOURCES: list[tuple[Path, MemorySlot]] = [
    (Path.home() / ".cowork/anton/memory/rules.md", MemorySlot.RULES),
    (Path.home() / ".cowork/anton/memory/lessons.md", MemorySlot.LESSONS),
    (Path.home() / ".cowork/anton/memory/profile.md", MemorySlot.PROFILE),
    (Path.home() / ".cowork/hermes/memories/USER.md", MemorySlot.PROFILE),
    (Path.home() / ".cowork/hermes/memories/MEMORY.md", MemorySlot.LESSONS),
]


def _combine_slot_memory(chunks: list[str]) -> str:
    """Merge legacy sources for the same slot, preserving source order."""
    parts: list[str] = []
    for chunk in chunks:
        text = chunk.strip()
        if text and text not in parts:
            parts.append(text)
    return "\n\n".join(parts)


def migrate_harness_memory_to_shared(session: Session) -> bool:
    """Copy legacy harness memory into the canonical store if not already migrated.

    Returns True if migration ran, False if the sentinel indicates it already ran.
    """
    if session.exec(
        select(Setting).where(Setting.key == _MEMORY_MIGRATION_SENTINEL)
    ).first() is not None:
        return False

    store = SharedMemoryStore()
    store._root.mkdir(parents=True, exist_ok=True)

    pre_existing = {slot: store.read(slot).strip() for slot in MemorySlot}
    incoming_by_slot: dict[MemorySlot, list[tuple[Path, str]]] = {}

    for source, slot in _MIGRATION_SOURCES:
        if not source.is_file():
            continue
        incoming = source.read_text(encoding="utf-8")
        if not incoming.strip():
            continue
        incoming_by_slot.setdefault(slot, []).append((source, incoming))

    for slot, entries in incoming_by_slot.items():
        if pre_existing.get(slot):
            continue
        combined = _combine_slot_memory([text for _, text in entries])
        if not combined:
            continue
        store.write(slot, combined)
        for source, _ in entries:
            logger.info("Migrated %s → %s", source, slot.value)

    # Hermes memory files block symlink creation while they exist as real files.
    from cowork.harnesses.memory.adapter import get_memory_adapter

    adapter = get_memory_adapter("hermes")
    if adapter is not None:
        for link_path in adapter.RUNTIME_SYMLINKS:
            if link_path.is_file() and not link_path.is_symlink():
                link_path.unlink()
                logger.info(
                    "Removed legacy Hermes memory file %s (content in canonical store)",
                    link_path,
                )

    session.add(Setting(key=_MEMORY_MIGRATION_SENTINEL, value="1"))
    session.commit()
    return True
