"""Messages the person added to a running turn (ENG-4194).

When someone sends a message while Anton works, POST /responses/addon may hand
it to the running turn instead of queueing it as the next one. The turn's
anton ``ChatSession`` keeps the unread messages and shows them to the model at
its next step boundary; this module is the server's side of that: which turn
can take one, the messages added to it so far, and the ``response.addon``
events that tell the app where each one went.

Keyed by ``conversation_id`` and process-local, like ``AnswerBroker``: the
POST reaches the process that runs the turn by construction on the in-process
backend. The remote (pod) backend has no channel here: the pod's turn keeps
the messages (anton's ``cloud_turn/addons.py``), cowork-server hands them over
through Redis (``cowork.turnqueue.addons``), and the pod's ``addon`` steps
become the same ``TurnAddon`` events (``producer.step_stream_events``).

Events do not go to the stream buffer directly. The harness yields them
between anton's own events (``AntonHarness.stream_response``), so they pass
through the formatter, get a sequence number and are saved with the answer's
events like every other frame, which is what replays them on reload. An
"added" event therefore reaches the stream with anton's next event; the person
who sent the message already knows from the POST's own response.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

__all__ = ["AddonChannel", "AddonRegistry", "TurnAddon", "addons", "supports_addons"]

AddonState = Literal["added", "read", "returned"]
DecidedBy = Literal["jev", "user"]
WithdrawResult = Literal["withdrawn", "already_read", "not_found"]

# What the installed anton must have. Older builds lack them; their turns get
# no channel, and every message sent during them is queued.
_SESSION_METHODS = ("add_user_input", "withdraw_user_input", "take_unread_user_inputs")


def supports_addons(session: Any) -> bool:
    """Whether this anton session can take a message mid-turn."""
    return all(callable(getattr(session, name, None)) for name in _SESSION_METHODS)


@dataclass(frozen=True)
class TurnAddon:
    """Synthetic stream event: what happened to one added message. Mapped to
    a ``response.addon`` SSE event by the formatter, which the answer's saved
    events keep.

    - ``added``: the running turn accepted it.
    - ``read``: anton showed it to the model.
    - ``returned``: the turn ended before the model saw it; the app runs it
      as the next task, so nothing is lost.
    """

    addon_id: str
    state: AddonState
    # None only on a pod's event that did not carry it (a "read" or
    # "returned" for a message the pod no longer knows); left out of the
    # payload then.
    text: str | None
    decided_by: DecidedBy | None

    def payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "response.addon",
            "addon_id": self.addon_id,
            "state": self.state,
        }
        if self.text is not None:
            payload["text"] = self.text
        if self.decided_by is not None:
            payload["decided_by"] = self.decided_by
        return payload


@dataclass
class _Addon:
    text: str
    decided_by: DecidedBy
    read: bool = False


class AddonChannel:
    """One running turn's side of added messages."""

    def __init__(
        self,
        conversation_id: str,
        session: Any,
        *,
        task: str,
        progress: Callable[[], list[str]],
    ) -> None:
        self.conversation_id = conversation_id
        self.task = task
        self._session = session
        self._progress = progress
        self._addons: dict[str, _Addon] = {}
        self._events: list[TurnAddon] = []
        # Messages are decided one at a time, in the order they arrive, so a
        # follow-up is judged with the one before it already added.
        self.deciding = asyncio.Lock()
        session.on_user_input_read = self._mark_read

    def progress(self) -> list[str]:
        """The turn's recent steps for Jev; empty when they can't be read."""
        try:
            return list(self._progress())
        except Exception:
            logger.warning("Could not read the running turn's progress", exc_info=True)
            return []

    def added_texts(self) -> list[str]:
        return [addon.text for addon in self._addons.values()]

    def add(self, text: str, *, decided_by: DecidedBy) -> str | None:
        """Hand *text* to the running turn. Its addon id, or None when the
        turn no longer takes messages (it is ending)."""
        addon_id = str(uuid.uuid4())
        try:
            accepted = self._session.add_user_input(addon_id, text)
        except Exception:
            logger.warning("anton refused an added message", exc_info=True)
            return None
        if not accepted:
            return None
        self._addons[addon_id] = _Addon(text=text, decided_by=decided_by)
        self._events.append(TurnAddon(addon_id, "added", text, decided_by))
        return addon_id

    def withdraw(self, addon_id: str) -> WithdrawResult:
        """Take an added message back before the model sees it ("Move to
        queue")."""
        addon = self._addons.get(addon_id)
        if addon is None:
            return "not_found"
        if addon.read:
            return "already_read"
        try:
            withdrawn = self._session.withdraw_user_input(addon_id)
        except Exception:
            logger.warning("anton could not withdraw an added message", exc_info=True)
            withdrawn = False
        if not withdrawn:
            # anton took it at a step boundary and has not reported it read
            # yet: the model has it either way.
            return "already_read"
        del self._addons[addon_id]
        return "withdrawn"

    def _mark_read(self, ids: list[str]) -> None:
        """anton's ``on_user_input_read``: runs on the event loop, must not
        block."""
        for addon_id in ids:
            addon = self._addons.get(addon_id)
            if addon is None or addon.read:
                continue
            addon.read = True
            self._events.append(TurnAddon(addon_id, "read", addon.text, addon.decided_by))

    def drain(self) -> list[TurnAddon]:
        """The events since the last drain, oldest first."""
        events, self._events = self._events, []
        return events

    def take_returned(self) -> list[TurnAddon]:
        """At turn end: every message the model never saw, as ``returned``
        events, after any events still undrained."""
        try:
            unread = self._session.take_unread_user_inputs() or []
        except Exception:
            logger.warning("Could not read the turn's unread added messages", exc_info=True)
            unread = []
        events = self.drain()
        for entry in unread:
            addon_id = str(entry.get("id") or "")
            addon = self._addons.pop(addon_id, None)
            decided_by: DecidedBy = addon.decided_by if addon is not None else "jev"
            events.append(TurnAddon(addon_id, "returned", str(entry.get("text") or ""), decided_by))
        return events

    def detach(self) -> None:
        if getattr(self._session, "on_user_input_read", None) == self._mark_read:
            self._session.on_user_input_read = None


class AddonRegistry:
    """Process-wide map of running turns that take added messages."""

    def __init__(self) -> None:
        self._by_cid: dict[str, AddonChannel] = {}

    def open(
        self,
        conversation_id: str,
        session: Any,
        *,
        task: str,
        progress: Callable[[], list[str]],
    ) -> AddonChannel | None:
        """A channel for the turn *session* runs, or None when its anton is
        too old to take messages mid-turn."""
        if not supports_addons(session):
            return None
        channel = AddonChannel(conversation_id, session, task=task, progress=progress)
        self._by_cid[conversation_id] = channel
        return channel

    def get(self, conversation_id: str) -> AddonChannel | None:
        return self._by_cid.get(conversation_id)

    def close(self, conversation_id: str, channel: AddonChannel) -> None:
        """Drop *channel*. Leaves a later turn's channel alone."""
        if self._by_cid.get(conversation_id) is channel:
            del self._by_cid[conversation_id]
        channel.detach()

    def reset(self) -> None:
        """For tests: this object is a process global."""
        for cid, channel in list(self._by_cid.items()):
            self.close(cid, channel)


# Single global instance per server process, mirroring streaming.answers.broker.
addons: AddonRegistry = AddonRegistry()
