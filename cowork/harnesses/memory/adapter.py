"""
This module defines the base class for memory adapters.

The core concept for enabling shared memory across harnesses is as follows:
- The Anton Hippocampus will be treated as the source of truth. This means that there
are three canonical slots (files) involved (defined in the registry module).
- Each harness will follow two integration rules:
  1. Direct mapping when possible: 1:1 file ↔ canonical file via symlink.
  2. Prompt injection for extras: when a harness reads more than it writes (Anton's
  memory stores more context), inject read-only canonical content via system prompt.

Each harness adapter should define:
- RUNTIME_SYMLINKS: the direct mapping between the harness's memory files and the canonical
slots. This is used to create the symlinks in the harness's memory directory.
- PROMPT_INJECT_SLOTS: the slots that will be injected into the system prompt. This is used to
inject the read-only canonical content into the system prompt.

The Symlinks will be created automatically based on the given definitions.
The prompt, however, will need to be injected within each harness by calling the 
build_prompt_context method.
"""
from pathlib import Path

from cowork.harnesses.memory.registry import MemorySlot
from cowork.harnesses.memory.store import ProjectMemoryStore, PROJECT_SLOTS, SharedMemoryStore


class BaseMemoryAdapter:
    """Harnesses subclass this and override the class attributes."""

    harness_id: str
    RUNTIME_SYMLINKS: dict[Path, MemorySlot] = {}
    PROMPT_INJECT_SLOTS: list[MemorySlot] = []

    def build_prompt_context(self, project_path: Path) -> str:
        return self._format_global_prompt_context() + "\n\n" + self._format_project_prompt_context(project_path)

    def _format_project_prompt_context(self, project_path: Path) -> str:
        store = ProjectMemoryStore(project_path)
        parts = []
        for slot in PROJECT_SLOTS:
            content = store.read(slot).strip()
            if content:
                parts.append(self._format_slot_for_prompt(slot, content))
        # Header and footer are added to avoid trailing lines being absorbed into the project context.
        if not parts:
            return ""
        return f"{"# Project Memory"}\n\n" + "\n\n".join(parts) + "\n\nEnd of Project Memory."

    def _format_global_prompt_context(self) -> str:
        store = SharedMemoryStore()
        parts = []
        for slot in self.PROMPT_INJECT_SLOTS:
            content = store.read(slot).strip()
            if content:
                parts.append(self._format_slot_for_prompt(slot, content))
        if not parts:
            return ""
        return "\n\n".join(parts)

    def _format_slot_for_prompt(self, slot: MemorySlot, content: str) -> str:
        headings = {
            MemorySlot.RULES: "## Behavioral Rules",
            MemorySlot.PROFILE: "## User Profile",
            MemorySlot.LESSONS: "## Agent-learned Knowledge",
        }
        return f"{headings.get(slot, f'## {slot.value}')}\n{content}"


_registry: dict[str, type[BaseMemoryAdapter]] = {}


def register(cls: type[BaseMemoryAdapter]) -> type[BaseMemoryAdapter]:
    _registry[cls.harness_id] = cls
    return cls


def get_memory_adapter(harness_id: str) -> BaseMemoryAdapter | None:
    cls = _registry.get(harness_id)
    return cls() if cls else None


def all_memory_adapters() -> list[BaseMemoryAdapter]:
    return [cls() for cls in _registry.values()]