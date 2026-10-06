"""A ``Permission`` defined OUTSIDE cowork/api/v1/permissions.py, under
deferred annotations, whose ``check`` names a type that module does not
import.

That combination is the one ``require()``'s ``eval_str=True`` exists for, and
it cannot be written inside permissions.py by construction: the point is a
name only the subclass's own module can resolve. Lives here rather than in
the test file because ``from __future__ import annotations`` is per-module.
"""
from __future__ import annotations

from pydantic import BaseModel


class ForeignBody(BaseModel):
    who: str


class ForeignBodyPermission:
    """Allows everything; the body model is what is being exercised."""

    async def check(self, payload: ForeignBody) -> str:
        return payload.who
