"""
This module ensures the canonical memory layout (and any registered harness
adapter symlinks) exists at startup.
"""

from cowork.harnesses.memory.adapter import all_memory_adapters
from cowork.harnesses.memory.layout import MemoryLayout


def ensure_all_layouts() -> None:
    # Always runs: ensure_layout creates the canonical slot files even when no
    # adapter is registered.
    MemoryLayout().ensure_layout(all_memory_adapters())
