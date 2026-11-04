"""In-process fan-out of doc changes to the editors and chats watching them.

Two kinds of topic:

- ``doc:<artifact id>``: every write to that doc, with the doc itself, for
  each editor that has it open (so the person watches the agent's sections
  land, and two windows on one doc stay in step);
- ``conv:<conversation id>``: "the agent created / changed a doc in this
  conversation", so the chat can show the doc's card and open it the moment
  the agent makes it, instead of at the end of the turn.

Publishing is safe from any thread (tool handlers and request handlers run on
worker threads as well as on the event loop). A subscriber that falls behind
loses events rather than blocking the writer; every event carries the doc's
``rev``, and a client that sees a gap refetches.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Any

_QUEUE_MAX = 64
# A conversation's doc events are kept this long, so a chat that subscribes a
# moment late (a new conversation learns its id from the turn's first event)
# still hears about the doc the agent just made.
_REPLAY_S = 120.0
_REPLAY_MAX = 32


class DocHub:
    def __init__(self) -> None:
        self._subs: dict[str, set[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = {}
        self._recent: dict[str, deque] = {}
        self._guard = threading.Lock()

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


def doc_topic(artifact_id: str) -> str:
    return "doc:" + str(artifact_id).replace("-", "").lower()


def conv_topic(conversation_id: str) -> str:
    return "conv:" + str(conversation_id).lower()
