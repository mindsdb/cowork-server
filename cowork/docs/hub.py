"""In-process fan-out of doc changes to the editors and chats watching them.

Two kinds of topic:

- ``doc:<doc folder>``: every write to that doc, with the doc itself, for
  each editor that has it open (so the person watches the agent's sections
  land, and two windows on one doc stay in step). Keyed by the folder the
  server resolved, never by an id read from the doc's files, which the
  agent's code can write: a doc can't be made to share another one's topic;
- ``conv:<conversation id>``: "the agent created / changed a doc in this
  conversation", so the chat can show the doc's card and open it the moment
  the agent makes it, instead of at the end of the turn.

Publishing is safe from any thread (tool handlers and request handlers run on
worker threads as well as on the event loop). A subscriber that falls behind
loses events rather than blocking the writer; every event carries the doc's
``rev``, and a client that sees a gap refetches.

Where several replicas serve one shared tree, a relay (``cowork.docs.relay``)
carries each published event to the other replicas. ``publish`` hands the relay
a small message and never waits on it; the relay gives the other replicas'
messages back through ``deliver_remote``. A ``doc:`` message carries no doc:
it wakes this replica's streams of that doc (a ``NUDGE`` event), and each
stream reads the doc from the file. A ``conv:`` message is the event itself.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_QUEUE_MAX = 64
# A conversation's doc events are kept this long, so a chat that subscribes a
# moment late (a new conversation learns its id from the turn's first event)
# still hears about the doc the agent just made.
_REPLAY_S = 120.0
_REPLAY_MAX = 32

#: The event a doc stream gets when another replica wrote that doc. Never sent
#: to a client: the stream answers it with the doc it reads from the file.
NUDGE = "doc.nudge"
#: A doc was published (its page is live). Carries no doc, so it never wakes a doc stream
#: to reread the file; the stream passes it on as it is.
DOC_PUBLISHED = "doc.published"

_DOC_PREFIX = "doc:"
_CONV_PREFIX = "conv:"
# Keys a relay message adds around the event; no event uses them.
_ENVELOPE = ("origin", "topic")
# Doc event types that mean the file changed: this replica's streams reread it.
_NUDGED = frozenset({"doc.created", "doc.updated"})
# What another replica's message may carry. The type becomes an SSE event name,
# and a doc stream does arithmetic on the rev.
_REMOTE_TYPES = _NUDGED | {DOC_PUBLISHED}
_CLIENT_ID_MAX = 100


class DocHub:
    def __init__(self) -> None:
        self._subs: dict[str, set[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = {}
        self._recent: dict[str, deque] = {}
        self._guard = threading.Lock()
        # Tells this process's messages from the other replicas' on the relay.
        self.origin = uuid.uuid4().hex
        self._relay: Callable[[dict[str, Any]], None] | None = None

    def attach(self, relay: Callable[[dict[str, Any]], None]) -> None:
        """Hand every published event's message to ``relay`` as well."""
        with self._guard:
            self._relay = relay

    def detach(self, relay: Callable[[dict[str, Any]], None]) -> None:
        with self._guard:
            if self._relay == relay:
                self._relay = None

    def subscribe(self, topic: str, *, replay: bool = False) -> asyncio.Queue:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAX)
        with self._guard:
            self._subs.setdefault(topic, set()).add((loop, queue))
            if replay:
                cutoff = time.monotonic() - _REPLAY_S
                for at, event in self._recent.get(topic, ()):
                    if at >= cutoff:
                        queue.put_nowait({**event, "replayed": True})
        return queue

    def unsubscribe(self, topic: str, queue: asyncio.Queue) -> None:
        with self._guard:
            subs = self._subs.get(topic)
            if not subs:
                return
            for entry in list(subs):
                if entry[1] is queue:
                    subs.discard(entry)
            if not subs:
                self._subs.pop(topic, None)

    def publish(self, topic: str, event: dict[str, Any], *, keep: bool = False) -> None:
        self._fan_out(topic, event, keep=keep)
        relay = self._relay
        if relay is None:
            return
        try:
            relay(self._message(topic, event))
        except Exception:  # the other replicas catch up by polling the file
            logger.debug("doc event not handed to the relay", exc_info=True)

    def deliver_remote(self, message: dict[str, Any]) -> None:
        """A message another replica published: wake the doc's streams, or pass the conversation's event on."""
        if message.get("origin") == self.origin:
            return
        topic = message.get("topic")
        if not isinstance(topic, str) or not _well_formed(message):
            return
        if topic.startswith(_DOC_PREFIX):
            kind = NUDGE if message.get("type") in _NUDGED else message.get("type")
            self._fan_out(topic, {"type": kind, "rev": message.get("rev"), "clientId": message.get("clientId")})
        elif topic.startswith(_CONV_PREFIX):
            self._fan_out(topic, {k: v for k, v in message.items() if k not in _ENVELOPE}, keep=True)

    def _message(self, topic: str, event: dict[str, Any]) -> dict[str, Any]:
        # A doc event's body stays here: the other replicas read the doc from the shared file.
        if topic.startswith(_DOC_PREFIX):
            body = {"type": event.get("type"), "rev": event.get("rev"), "clientId": event.get("clientId")}
        else:
            body = dict(event)
        return {**body, "origin": self.origin, "topic": topic}

    def _fan_out(self, topic: str, event: dict[str, Any], *, keep: bool = False) -> None:
        with self._guard:
            targets = list(self._subs.get(topic, ()))
            if keep:
                now = time.monotonic()
                # Conversations come and go; a topic with nothing left to replay is dropped.
                for old in [t for t, kept in self._recent.items() if kept and kept[-1][0] < now - _REPLAY_S]:
                    del self._recent[old]
                self._recent.setdefault(topic, deque(maxlen=_REPLAY_MAX)).append((now, event))
        for loop, queue in targets:
            try:
                loop.call_soon_threadsafe(_offer, queue, event)
            except RuntimeError:
                # The subscriber's loop has closed; its stream is gone.
                self.unsubscribe(topic, queue)


def _well_formed(message: dict[str, Any]) -> bool:
    rev, client_id = message.get("rev"), message.get("clientId")
    return (
        message.get("type") in _REMOTE_TYPES
        and isinstance(rev, int) and not isinstance(rev, bool)
        and (client_id is None or (isinstance(client_id, str) and len(client_id) <= _CLIENT_ID_MAX))
    )


def _offer(queue: asyncio.Queue, event: dict[str, Any]) -> None:
    try:
        queue.put_nowait(event)
    except asyncio.QueueFull:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            queue.put_nowait({**event, "gap": True})
        except asyncio.QueueFull:
            pass


hub = DocHub()


def doc_topic(folder: Path | str) -> str:
    # Normalized without touching the disk, so it is safe on the event loop.
    return _DOC_PREFIX + os.path.normpath(os.path.abspath(folder))


def conv_topic(conversation_id: str) -> str:
    return _CONV_PREFIX + str(conversation_id).lower()
