"""The skills block of a cloud turn's request: every skill the pod's agent may
recall in the turn.

On the wire, ``{"skills": {<slug>: {"files": {<relpath>: <text>}, "always_on":
true}}}``, with ``always_on`` left out for a skill recalled on demand. The
pod's anton reads the same shape and the same field names (its cloud turn
contract's ``skills``); this server keeps its own model of an entry, since
the anton release it pins predates that contract's.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TurnSkillEntry(BaseModel):
    """One skill of the block: its folder's files by path relative to it,
    SKILL.md among them, and whether the pod puts it in every turn's
    prompt."""

    model_config = ConfigDict(frozen=True)

    files: dict[str, str]
    always_on: bool = False


# The block: each skill by its slug, in the order the pod reads them.
TurnSkills = dict[str, TurnSkillEntry]


def turn_skills_wire(skills: TurnSkills) -> dict[str, dict[str, object]]:
    """The block as the request carries it: each entry's files, and
    ``always_on`` only where it is true."""
    return {slug: entry.model_dump(exclude_defaults=True) for slug, entry in skills.items()}
