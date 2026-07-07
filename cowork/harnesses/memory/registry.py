"""
This module defines the canonical slots for shared harness memory.

These slots have been based on the Anton's memory system.
When a new harness is onboarded, it's memory mechanim should be mapped to these canonical slots.
If a particular aspect of a new harness's memory does not adhere to these slots, a new slot can be created.
"""
from dataclasses import dataclass
from enum import StrEnum


class MemorySlot(StrEnum):
    PROFILE = "profile"
    RULES = "rules"
    LESSONS = "lessons"


@dataclass(frozen=True)
class SlotMeta:
    filename: str
    description: str


SLOT_REGISTRY: dict[MemorySlot, SlotMeta] = {
    MemorySlot.PROFILE: SlotMeta("profile.md", "User identity, preferences"),
    MemorySlot.RULES: SlotMeta("rules.md", "Behavioral gates"),
    MemorySlot.LESSONS: SlotMeta("lessons.md", "Agent-learned knowledge"),
}
