"""
This module defines the base class for memory adapters.

The core concept for enabling shared memory across harnesses is as follows:
- The Anton Hippocampus will be treated as the source of truth. This means that there
are three canonical slots (files) involved (defined in the registry module).
- Each harness will follow two integration rules:
  1. Direct mapping when possible: 1:1 file ↔ canonical file via symlink.
  2. Prompt injection for extras: when a harness reads more than it writes (Anton’s 
  memory stores more context), inject read-only canonical content via system prompt.

Each harness adapter should define:
- RUNTIME_SYMLINKS: the direct mapping between the harness's memory files and the canonical 
slots. This is used to create the symlinks in the harness's memory directory.
- PROMPT_INJECT_SLOTS: the slots that will be injected into the system prompt. This is used to
inject the read-only canonical content into the system prompt.
"""
from typing import Protocol

from cowork.harnesses.memory.registry import MemorySlot


class BaseMemoryAdapter(Protocol):
    """Helper base — harnesses only override the dicts."""

    harness_id: str

    RUNTIME_SYMLINKS: dict[str, MemorySlot] = {}
    PROMPT_INJECT_SLOTS: list[MemorySlot] = []
