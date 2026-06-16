from pathlib import Path

from cowork.harnesses.hermes_harness.settings import HermesHarnessSettings
from cowork.harnesses.memory.adapter import BaseMemoryAdapter
from cowork.harnesses.memory.registry import MemorySlot


settings = HermesHarnessSettings()

memory_dir = Path(settings.root_dir) / "memories"


class HermesMemoryAdapter(BaseMemoryAdapter):
    harness_id = "hermes"
    RUNTIME_SYMLINKS = {
        memory_dir / "USER.md": MemorySlot.PROFILE,
        memory_dir / "MEMORY.md": MemorySlot.LESSONS,
    }
    PROMPT_INJECT_SLOTS = [MemorySlot.RULES]
