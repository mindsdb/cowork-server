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

__all__ = [
    "AddonChannel", "AddonRegistry", "TurnAddon", "addons", "supports_addons", "turn_limit_reason",
    "MAX_ADDON_CHARS", "MAX_ADDONS_PER_TURN", "MAX_ADDED_BYTES_PER_TURN",
]

AddonState = Literal["added", "read", "returned", "withdrawn"]
DecidedBy = Literal["jev", "user"]
# "internal": anton raised; the message stays where it is (503 at the route).
WithdrawResult = Literal["withdrawn", "already_read", "not_found", "internal"]

# Limits on what one turn takes mid-task, in both backends: each message is
# typed into the composer, and everything added is read by Jev and by the
# model on every later step, so a runaway client can't grow either without
# bound. Past them the message is queued as the next task, as without this
# feature.
#: Characters in one added message (POST /addon queues it as too_large).
MAX_ADDON_CHARS = 4000
#: Messages one turn takes (too_many), withdrawn ones included.
MAX_ADDONS_PER_TURN = 20
#: UTF-8 bytes of text one turn takes, withdrawn ones included (too_large).
MAX_ADDED_BYTES_PER_TURN = 64 * 1024


def turn_limit_reason(*, count: int, total_bytes: int, text: str) -> Literal["too_many", "too_large"] | None:
    """Why a turn that already took *count* messages totalling *total_bytes*
    can't take *text* too, or None when it can."""
    if count >= MAX_ADDONS_PER_TURN:
        return "too_many"
    if total_bytes + len(text.encode("utf-8")) > MAX_ADDED_BYTES_PER_TURN:
        return "too_large"
    return None

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
    - ``withdrawn``: the person moved it back to the queue ("Move to queue")
      before the model saw it; saved so a reload shows it moved.

    ``local_key`` is the app's own id for the message (POST /addon's
    ``local_key``), on every event of an addon that sent one.
    """

    addon_id: str
    state: AddonState
    # None only on a pod's event that did not carry it (a "read" or
    # "returned" for a message the pod no longer knows); left out of the
    # payload then.
    text: str | None
    decided_by: DecidedBy | None
    local_key: str | None = None

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
        if self.local_key is not None:
            payload["local_key"] = self.local_key
        return payload


@dataclass
class _Addon:
    text: str
    decided_by: DecidedBy
    local_key: str | None = None
    read: bool = False

    def event(self, addon_id: str, state: AddonState) -> TurnAddon:
        return TurnAddon(addon_id, state, self.text, self.decided_by, self.local_key)


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
        # Every message the turn took, withdrawn ones included, for the
        # per-turn limits (turn_limit_reason).
        self.accepted_count = 0
        self.accepted_bytes = 0
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

    def limit_reason(self, text: str) -> Literal["too_many", "too_large"] | None:
        return turn_limit_reason(count=self.accepted_count, total_bytes=self.accepted_bytes, text=text)

    def add(self, text: str, *, decided_by: DecidedBy, local_key: str | None = None) -> str | None:
        """Hand *text* to the running turn. Its addon id, or None when the
        turn no longer takes messages (it is ending). An exception from
        anton propagates: it is not a refusal, and the route reports it as
        ``internal``."""
        addon_id = str(uuid.uuid4())
        if not self._session.add_user_input(addon_id, text):
            return None
        addon = _Addon(text=text, decided_by=decided_by, local_key=local_key)
        self._addons[addon_id] = addon
        self.accepted_count += 1
        self.accepted_bytes += len(text.encode("utf-8"))
        self._events.append(addon.event(addon_id, "added"))
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
            # Not a verdict: whether the model has it is unknown, so the
            # message stays where it is rather than reading as already read.
            logger.exception("anton failed to withdraw an added message",
                             extra={"conversation_id": self.conversation_id})
            return "internal"
        if not withdrawn:
            # anton took it at a step boundary and has not reported it read
            # yet: the model has it either way.
            return "already_read"
        del self._addons[addon_id]
        # Saved with the turn's events, so a reload shows the message moved
        # back to the queue. An "added" the stream has not carried yet is
        # dropped instead: then the stream never shows it at all.
        undrained = [e for e in self._events if e.addon_id == addon_id and e.state == "added"]
        if undrained:
            self._events = [e for e in self._events if e.addon_id != addon_id]
        else:
            self._events.append(addon.event(addon_id, "withdrawn"))
        return "withdrawn"

    def _mark_read(self, ids: list[str]) -> None:
        """anton's ``on_user_input_read``: runs on the event loop, must not
        block."""
        for addon_id in ids:
            addon = self._addons.get(addon_id)
            if addon is None or addon.read:
                continue
            addon.read = True
            self._events.append(addon.event(addon_id, "read"))

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
            if addon is None:
                # Only ids this channel accepted reach the stream.
                logger.warning("anton returned an added message this turn never took",
                               extra={"conversation_id": self.conversation_id})
                continue
            events.append(addon.event(addon_id, "returned"))
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
