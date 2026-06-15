"""
This module defines the canonical slots for shared harness memory.
These slots have been based on the Anton's memory system.
When a new harness is onboarded, it's memory mechanim should be mapped to these canonical slots.
If a particular aspect of a new harness's memory does not adhere to these slots, a new slot can be created.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CanonicalSlot:
    id: str
    filename: str
    description: str


# The registry — add entries as new harnesses need new shared concepts.
CANONICAL_SLOTS: dict[str, CanonicalSlot] = {
    "profile": CanonicalSlot(
        id="profile",
        filename="profile.md",
        description="User identity, preferences (Anton profile, Hermes user)",
    ),
    "rules": CanonicalSlot(
        id="rules",
        filename="rules.md",
        description="Behavioral gates — Anton-only for now",
    ),
    "lessons": CanonicalSlot(
        id="lessons",
        filename="lessons.md",
        description="Agent-learned knowledge (Anton lessons, Hermes memory)",
    ),
}


def get_filename(slot_id: str) -> str:
    slot = CANONICAL_SLOTS.get(slot_id)
    if slot is None:
        raise ValueError(f"Unknown canonical memory slot: {slot_id!r}")
    return slot.filename
