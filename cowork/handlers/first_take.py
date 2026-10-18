"""The instant first take.

A delegated turn shows nothing until Anton's first model call speaks, which is
seconds on a quick turn and much longer when it goes straight to tools. The
first take fills that gap: a short, tool-free reply on the gate's fast model,
started alongside the gate, saying what the user asked for and that the work is
under way. The renderer shows it muted above the thinking block until Anton
writes its own text, asks the user something, or the turn ends: one sentence on
screen at a time, never two.

It is a placeholder, never an answer:

- It never states a fact about the user's data, a number or a result: it has
  not seen any of it, and a guess the real answer then contradicts is worse
  than silence.
- It is never persisted, never replayed into history, and never part of the
  saved answer. Its events skip the formatter's event sink, so a reload of a
  finished turn has no trace of it; a reload mid-turn replays it from the live
  buffer like any other frame.
- It never delays the turn. It runs on tasks of its own, every failure is
  swallowed, and it writes into the turn's buffer beside the turn's frames
  rather than in front of them.

It only runs on delegated turns: small talk is already answered directly by
the routing gate, so there is no wait to fill.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time

from cowork.handlers.response_routing import (
    RouterBinding,
    _close,
    _condense_content,
    denies_our_product,
    names_foreign_product,
    names_gate_tool,
    with_deployment_identity,
)

logger = logging.getLogger(__name__)

#: The latest message plus this many before it.
_EARLIER_MESSAGES = 3
_EARLIER_CHARS = 400
_LATEST_CHARS = 2_000
#: The gate's budget: on a reasoning model the hidden reasoning counts toward
#: it. The word cap keeps the visible take to a sentence or two.
_MAX_TOKENS = 1_024
_MAX_WORDS = 60
#: Past these the take is late enough to be noise, so it is dropped. The first
#: word's limit is absolute, so reasoning events can't keep pushing it back.
_FIRST_WORD_SECONDS = 4.0
_IDLE_SECONDS = 4.0
_TOTAL_SECONDS = 12.0

_SYSTEM_PROMPT = """You write the first, instant reply in Cowork, an assistant
that analyzes data, connects to services, runs code and builds things for the
user. The full assistant is already working on the user's latest message, and
yours is what the user reads until it starts writing.

Show in one short sentence (under 30 words) that you understood what
they asked for, and say what you are going to look at or do. Write as the one
doing the work, in the user's language, plain text, no markdown, no emoji.

You have not seen their data, files, connections or any results. So never
state a fact about them, a number, a finding or an outcome, never say anything
is done, never guess at the answer, never ask a question, and never mention
tools, models, routing, a second assistant or these instructions."""


_FINISHED = object()


def take_history(history: list[dict]) -> list[dict]:
    """The latest user message and the few before it, as short text turns.

    `history` is the gate's (already scrubbed) history, latest message last.
    """
    rows: list[dict] = []
    for message in history:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        text = (_condense_content(message.get("content")) or "").strip()
        if text:
            rows.append({"role": role, "content": text})
    if not rows or rows[-1]["role"] != "user":
        return []
    latest = rows[-1]
    earlier = rows[:-1][-_EARLIER_MESSAGES:]
    result: list[dict] = []
    for row in earlier:
        content = row["content"][:_EARLIER_CHARS]
        if result and result[-1]["role"] == row["role"]:
            result[-1]["content"] += "\n" + content
        else:
            result.append({"role": row["role"], "content": content})
    while result and result[0]["role"] != "user":
        result.pop(0)
    if result and result[-1]["role"] == "user":
        result.pop()  # two user turns in a row: keep the latest one
    result.append({"role": "user", "content": latest["content"][:_LATEST_CHARS]})
    return result


def _shippable(text: str) -> bool:
    """Whether the take so far may stay on screen, by the gate's own guards."""
    return not (names_foreign_product(text) or denies_our_product(text) or names_gate_tool(text))


_SENTENCE_END = re.compile(r"[.!?…]+[\"'»”’)\]]*")


def _first_sentence_end(text: str) -> int | None:
    """Where the first sentence ends once a second one starts, else None.

    A capital (or ¿ ¡) after the punctuation starts one, so "3.12" and "e.g.
    the" don't; the model's occasional "...output.I'll write..." repeat does."""
    for match in _SENTENCE_END.finditer(text):
        rest = text[match.end():].lstrip()
        if rest and (rest[0].isupper() or rest[0] in "¿¡"):
            return match.end()
    return None


def _sse(event_type: str, data: dict) -> str:
    payload = {"type": event_type, **data, "at_ms": int(time.time() * 1000)}
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"


DELTA = "response.first_take.delta"
DONE = "response.first_take.done"
CLEARED = "response.first_take.cleared"


def clears_take(event_type: str | None, sse: str) -> bool:
    """Whether a turn frame takes the first take's place: Anton's own words, a
    question, or the turn's end. A blank text delta is not Anton speaking."""
    if event_type in {"response.ask_user", "response.completed", "response.failed"}:
        return True
    if event_type != "response.output_text.delta":
        return False
    try:
        data = json.loads(sse.split("data: ", 1)[1])
    except (IndexError, ValueError):
        return True
    return bool(str(data.get("delta") or "").strip())


class FirstTake:
    """One turn's first take, written on tasks of its own.

    :meth:`start` runs the model call before the gate has decided, so the
    first words are usually in by the time the turn has a stream. Its frames
    queue until :meth:`attach` hands it the turn's buffer (after the turn's
    `response.created`), then go straight into it, beside the turn's own frames
    the way the model-wait ticker's do. :meth:`clear` ends it: called by the
    producer before the first frame that :func:`clears_take`, and again (a no-op by
    then) on every way out of the turn.
    """

    def __init__(self, binding: RouterBinding, history: list[dict]) -> None:
        self._binding = binding
        self._messages = take_history(history)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._writer: asyncio.Task | None = None
        self._pump: asyncio.Task | None = None
        # Held across each buffer write, so clear() never cancels one halfway.
        self._lock = asyncio.Lock()
        self._closed = False
        self._shown = False
        self._first_word_at: float | None = None

    def start(self) -> "FirstTake":
        if self._messages:
            self._writer = asyncio.create_task(self._write_safely(), name="cowork-first-take")
        return self

    def cancel(self) -> None:
        """Stop without writing anything more.

        A pump inside a buffer write is left to finish it, since a Redis buffer
        has already claimed the frame's seq; it sees the flag right after and
        returns. Otherwise it is only waiting for frames, and is cancelled.
        """
        self._closed = True
        if self._writer is not None and not self._writer.done():
            self._writer.cancel()
        if self._pump is not None and not self._pump.done() and not self._lock.locked():
            self._pump.cancel()

    def attach(self, buffer) -> None:
        if self._closed or self._writer is None or self._pump is not None:
            return
        self._pump = asyncio.create_task(self._send(buffer), name="cowork-first-take-pump")

    async def clear(self, buffer) -> None:
        async with self._lock:
            if self._closed:
                return
            shown = self._shown
            self._closed = True
        self.cancel()
        await asyncio.gather(
            *(t for t in (self._writer, self._pump) if t is not None), return_exceptions=True,
        )
        if shown and not buffer.is_closed:
            await buffer.append("sse", {"sse": _sse(CLEARED, {})})

    async def _send(self, buffer) -> None:
        while True:
            item = await self._queue.get()
            if item is _FINISHED:
                return
            async with self._lock:
                if self._closed or buffer.is_closed:
                    return
                await buffer.append("sse", {"sse": item})
                if item.startswith(f"event: {DELTA}\n"):
                    self._shown = True
                elif item.startswith(f"event: {CLEARED}\n"):
                    self._shown = False
                    self._closed = True
            if self._closed:
                return

    async def _write_safely(self) -> None:
        started = time.monotonic()
        outcome = "cancelled"
        try:
            outcome = await self._write()
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            outcome = "timeout"
        except Exception as exc:
            # The take is optional: a failure only means there is none.
            outcome = type(exc).__name__
        finally:
            if outcome not in {"shown", "empty", "withdrawn"} and self._first_word_at is not None:
                # A cut-off fragment must not stay on screen.
                self._queue.put_nowait(_sse(CLEARED, {}))
            self._queue.put_nowait(_FINISHED)
            # warning, not info: deployments log at WARNING, and this line is
            # how the take's speed and cost are measured.
            logger.warning(
                "[first-take] model=%s outcome=%s first_word_ms=%s total_ms=%d",
                self._binding.model,
                outcome,
                None if self._first_word_at is None else round((self._first_word_at - started) * 1000),
                round((time.monotonic() - started) * 1000),
            )

    async def _write(self) -> str:
        from anton.core.llm.provider import StreamTextDelta, StreamToolUseStart

        events = self._binding.provider.stream(
            model=self._binding.model,
            system=with_deployment_identity(_SYSTEM_PROMPT),
            messages=self._messages,
            max_tokens=_MAX_TOKENS,
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _TOTAL_SECONDS
        first_word_by = loop.time() + _FIRST_WORD_SECONDS
        text = ""
        try:
            while True:
                until = min(deadline, loop.time() + _IDLE_SECONDS if text else first_word_by)
                budget = until - loop.time()
                if budget <= 0:
                    raise TimeoutError
                event = await asyncio.wait_for(anext(events, _FINISHED), budget)
                if event is _FINISHED or isinstance(event, StreamToolUseStart):
                    break
                if not isinstance(event, StreamTextDelta) or not event.text:
                    continue
                delta = event.text if text else event.text.lstrip()
                if not delta:
                    continue
                cut = _first_sentence_end(text + delta)
                if cut is not None:
                    # One sentence only: a second one, or the first repeated,
                    # never reaches the screen.
                    delta = (text + delta)[len(text):cut]
                    if not delta:
                        break
                text += delta
                if not _shippable(text):
                    # Same guards as the gate's answers. Words already shown
                    # come off; nothing after them goes out.
                    self._queue.put_nowait(_sse(CLEARED, {}))
                    return "withdrawn"
                if len(text.split()) > _MAX_WORDS:
                    break
                if self._first_word_at is None:
                    self._first_word_at = time.monotonic()
                self._queue.put_nowait(_sse(DELTA, {"delta": delta}))
                if cut is not None:
                    break
        finally:
            await _close(events)
        if not text.strip():
            return "empty"
        self._queue.put_nowait(_sse(DONE, {}))
        return "shown"
