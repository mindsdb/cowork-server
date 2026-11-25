"""Preview checks: anton asks whether an edited artifact runs in the open preview.

anton's artifact editor, after it writes an edit, can ask "does this artifact
still run where the user is looking at it?". In process that question goes:

1. anton awaits the turn's ``preview_checker(artifact_slug, timeout_s)``
   (``cowork/harnesses/anton_harness/preview_checker.py``).
2. That opens a check here and writes one ``response.artifact_check`` frame to
   the turn's stream.
3. The cowork web app, if a viewer for that artifact is mounted, reloads its
   preview, collects what the in-frame shim reports (preview_shim.js), and
   answers with ``POST /api/v1/artifacts/checks/{check_id}``. With no viewer
   mounted it answers ``unavailable`` at once, so anton does not wait out the
   timeout.
4. The answer resolves the future the checker awaits.

Same shape and same reasons as ``answers.AnswerBroker``: one cowork-server
process holds the turn, so the POST reaches the process that holds the
future, and an in-memory map is enough. Unlike a question, a check is
addressed by its id alone (the web app's POST carries no conversation id),
so the id is random and unguessable and the endpoint still checks that the
caller may touch the conversation the check belongs to.

A quick edit from the artifact viewer (ENG-4227) asks the same question
outside any turn: its check is opened with an ``owner``, the person whose
request runs the edit, and the endpoint authorizes against that owner instead
of a live turn. Its key is the quick edit's own (``quick-edit:<artifact>``),
never a conversation id, so a turn's end never closes it.
"""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cowork.streaming.turn_index import TurnOwner

__all__ = [
    "CHECK_ID_PREFIX",
    "CheckSubmitResult",
    "PreviewCheckBroker",
    "preview_checks",
]

#: Every check id starts with this; the rest is 128 random bits in hex.
CHECK_ID_PREFIX = "acheck:"


class CheckSubmitResult(Enum):
    """What happened to a submitted answer. The values match the answer
    endpoint's statuses, so the web app reads both the same way."""

    ACCEPTED = "accepted"
    # Never opened, timed out, or the turn ended: all read as 404.
    NOT_FOUND = "not_found"
    ALREADY_ANSWERED = "already_answered"


@dataclass
class _PendingCheck:
    conversation_id: str
    future: asyncio.Future
    # Set for a check no turn holds (a quick edit): who may answer it.
    owner: "TurnOwner | None" = None


class PreviewCheckBroker:
    """Process-wide map of preview checks awaiting the web app's answer."""

    def __init__(self) -> None:
        self._pending: dict[str, _PendingCheck] = {}

    def open(self, conversation_id: str, *, owner: "TurnOwner | None" = None) -> tuple[str, asyncio.Future]:
        """Register a new check for *conversation_id*: its id and the future
        its answer resolves. ``owner`` marks a check no turn holds; the
        endpoint then authorizes the answer against it."""
        check_id = f"{CHECK_ID_PREFIX}{secrets.token_hex(16)}"
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[check_id] = _PendingCheck(conversation_id=conversation_id, future=future, owner=owner)
        return check_id, future

    def owner_of(self, check_id: str) -> "TurnOwner | None":
        """The owner a check was opened with, or None for a turn's check (and
        for no check at all: ``conversation_of`` tells those apart)."""
        pending = self._pending.get(check_id)
        return pending.owner if pending is not None else None

    def conversation_of(self, check_id: str) -> str | None:
        """The conversation an open check belongs to, or None if there is no
        such check. The endpoint authorizes against it before submitting."""
        pending = self._pending.get(check_id)
        return pending.conversation_id if pending is not None else None

    def submit(self, check_id: str, answer: dict) -> CheckSubmitResult:
        pending = self._pending.get(check_id)
        if pending is None:
            return CheckSubmitResult.NOT_FOUND
        if pending.future.done():
            return CheckSubmitResult.ALREADY_ANSWERED
        pending.future.set_result(answer)
        return CheckSubmitResult.ACCEPTED

    def close(self, check_id: str) -> None:
        """Drop the check. Idempotent: the checker calls it in a ``finally``,
        after an answer, a timeout or a cancel. An unanswered future resolves
        None, the same as a check that never got an answer."""
        pending = self._pending.pop(check_id, None)
        if pending is not None and not pending.future.done():
            pending.future.set_result(None)

    def close_conversation(self, conversation_id: str) -> None:
        """Drop every check of *conversation_id*, resolving each still open to
        None. The harness calls it when the turn ends, so an answer that
        arrives afterwards is a 404 and nothing waits on a finished turn."""
        for check_id in [cid for cid, p in self._pending.items() if p.conversation_id == conversation_id]:
            self.close(check_id)

    def reset(self) -> None:
        """Drop every check. For tests: this object is a process global."""
        for check_id in list(self._pending):
            self.close(check_id)


# Single global instance per server process, like streaming.answers.broker.
preview_checks: PreviewCheckBroker = PreviewCheckBroker()
