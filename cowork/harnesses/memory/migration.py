"""
One-time migrations of legacy harness-local memory files into the shared
canonical store. The Hermes source paths stay listed so an install upgrading
straight from a pre-migration build still gets that content merged.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlmodel import Session, select

from cowork.common.paths import cowork_home
from cowork.harnesses.memory.registry import MemorySlot
from cowork.harnesses.memory.store import GlobalMemoryStore
from cowork.models.setting import Setting


logger = logging.getLogger(__name__)

_MEMORY_MIGRATION_SENTINEL = "_memory_migrated"
_HERMES_RETIRE_SENTINEL = "_hermes_memory_retired"

_HERMES_MEMORY_FILES: list[tuple[Path, MemorySlot]] = [
    (cowork_home() / "hermes/memories/USER.md", MemorySlot.PROFILE),
    (cowork_home() / "hermes/memories/MEMORY.md", MemorySlot.LESSONS),
]

_MIGRATION_SOURCES: list[tuple[Path, MemorySlot]] = [
    (cowork_home() / "anton/memory/rules.md", MemorySlot.RULES),
    (cowork_home() / "anton/memory/lessons.md", MemorySlot.LESSONS),
    (cowork_home() / "anton/memory/profile.md", MemorySlot.PROFILE),
    (cowork_home() / "hermes/memories/USER.md", MemorySlot.PROFILE),
    (cowork_home() / "hermes/memories/MEMORY.md", MemorySlot.LESSONS),
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

    store = GlobalMemoryStore()
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

    session.add(Setting(key=_MEMORY_MIGRATION_SENTINEL, value="1"))
    session.commit()
    return True


def retire_hermes_memory(session: Session) -> bool:
    """Fold divergent Hermes memory copies into the canonical store.

    On Windows without symlink permission the layout step copied the canonical
    file into the Hermes paths, and Hermes then wrote to the copy. The Hermes
    harness is gone, so nothing reads those copies again; append any content
    not already in the canonical slot. Symlinks already point at the canonical
    file and are skipped. Source files are never modified or deleted.
    Returns True if it ran, False if the sentinel says it already did.
    """
    if session.exec(
        select(Setting).where(Setting.key == _HERMES_RETIRE_SENTINEL)
    ).first() is not None:
        return False

    store = GlobalMemoryStore()
    for source, slot in _HERMES_MEMORY_FILES:
        if source.is_symlink() or not source.is_file():
            continue
        incoming = source.read_text(encoding="utf-8")
        if not incoming.strip():
            continue
        existing = store.read(slot).strip()
        have = {p.strip() for p in existing.split("\n\n") if p.strip()}
        new_parts = [p.strip() for p in incoming.split("\n\n") if p.strip() and p.strip() not in have]
        if new_parts:
            store.write(slot, "\n\n".join(([existing] if existing else []) + new_parts))
            logger.info("Merged Hermes memory %s → %s", source, slot.value)

    session.add(Setting(key=_HERMES_RETIRE_SENTINEL, value="1"))
    session.commit()
    return True
