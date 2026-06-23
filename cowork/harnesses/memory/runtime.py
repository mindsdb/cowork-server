"""
This module orchestrates memory layout for registered harness adapters at startup.
"""

from cowork.harnesses.memory.adapter import all_memory_adapters
from cowork.harnesses.memory.layout import MemoryLayout


def ensure_all_layouts() -> None:
    adapters = all_memory_adapters()
    if adapters:
        MemoryLayout().ensure_layout(adapters)
