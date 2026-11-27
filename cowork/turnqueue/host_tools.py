"""Tools this server runs for a remote (pod) turn ("host tools").

The pod's anton offers each tool of the request's ``host_tools`` block to its
model and forwards each call as a ``host_tool_call`` step on its stdout, which
scratchpad-controller relays as a ``turn_step``. The turn's relay loop hands
the step to ``HostToolRunner.start``: the call runs as a task of its own, so
the loop goes on reading the pod's replies (Stop, liveness) meanwhile. Steps
and the answer go back on the queue ask_user answers take
(``answer_queue_key``), which the controller writes to the pod's stdin:
``host_tool_progress`` entries while the call runs, then one or more
``host_tool_result`` parts, each a stdin line that fits the pod's limit.

Each call runs at most once: the pod mints the call id, so the once-only key
also holds the turn's correlation id. Nothing is sent once the turn has ended
(``close``), and the turn end waits a bounded time for calls still running
(``finish``) without cancelling them: a call's work may be in a thread that
runs on regardless.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from cowork.turnqueue.answers import ANSWER_TTL_S, MAX_ANSWER_BYTES, fits_stdin_line
from cowork.turnqueue.redis_client import answer_queue_key, get_redis

logger = logging.getLogger(__name__)

__all__ = [
    "CALL_ID", "HOST_TOOL_CALL_STEP", "MAX_RESULT_BYTES", "TURN_END_WAIT_S", "EmitStep", "ExecuteHostTool",
    "HostCall", "HostToolOutcome", "HostToolRunner", "call_key", "capped", "progress_entry", "result_entries",
]

HOST_TOOL_CALL_STEP = "host_tool_call"
CALL_ID = re.compile(r"^htool:[0-9a-f]{32}$")
DEFAULT_TIMEOUT_S = 60
MAX_TIMEOUT_S = 120
#: Past the pod's own timeout, so a late re-delivery still finds the key.
CALL_GRACE_S = 60
#: The whole answer, in UTF-8 bytes of text.
MAX_RESULT_BYTES = 1024 * 1024
MAX_PROGRESS_CHARS = 500
#: The turn end's wait for calls still running: a doc lock's wait (5 s) plus
#: parsing a large doc under it (3-8 s), with room to spare.
TURN_END_WAIT_S = 20.0
_TOO_BIG_TEXT = "The answer is over 1 MiB, more than the host sends for one call."
_HOST_ERROR_TEXT = "Refused (host_error): the host could not send this answer."
_BUSY_TEXT = "Refused (busy): nothing was changed."


@dataclass(frozen=True)
class HostCall:
    call_id: str
    tool: str
    input: dict[str, Any]


@dataclass(frozen=True)
class HostToolOutcome:
    text: str
    ok: bool = True


EmitStep = Callable[[str], Awaitable[None]]
ExecuteHostTool = Callable[[HostCall, EmitStep], Awaitable[HostToolOutcome]]


def call_key(correlation_id: str, call_id: str) -> str:
    return f"cowork:htool:{correlation_id}:{call_id}"


def _line(payload: dict[str, Any]) -> str:
    # Compact and ensure_ascii=False, so the size tracks the raw UTF-8 size the controller measures.
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _encodable(text: str) -> str:
    # Doc text can hold a lone surrogate (agent code in the pod may write one);
    # unreplaced, it fails the size check and the push in Redis.
    return text.encode("utf-8", "replace").decode("utf-8")


def progress_entry(call_id: str, n: int, text: str) -> str:
    text = _encodable(text)
    if len(text) > MAX_PROGRESS_CHARS:
        text = text[: MAX_PROGRESS_CHARS - 1] + "…"
    return _line({"kind": "host_tool_progress", "call_id": call_id, "entry_id": f"{call_id}:{n}", "text": text})


def _result_line(call_id: str, seq: int, *, last: bool, ok: bool, text: str) -> str:
    return _line({"kind": "host_tool_result", "call_id": call_id, "entry_id": f"{call_id}:{seq}", "seq": seq,
                  "last": last, "ok": ok, "text": text})


def result_entries(call_id: str, *, ok: bool, text: str) -> list[str]:
    """The answer as stdin lines that each fit the pod's limit once the
    controller re-wraps them, cut on character boundaries by encoded size:
    escaping can grow a character to six bytes."""
    entries: list[str] = []
    start, seq = 0, 0
    while True:
        rest = text[start:]
        whole = _result_line(call_id, seq, last=True, ok=ok, text=rest)
        if fits_stdin_line(whole):
            entries.append(whole)
            return entries
        lo, hi = 1, min(len(rest), MAX_ANSWER_BYTES)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if fits_stdin_line(_result_line(call_id, seq, last=False, ok=ok, text=rest[:mid])):
                lo = mid
            else:
                hi = mid - 1
        entries.append(_result_line(call_id, seq, last=False, ok=ok, text=rest[:lo]))
        start += lo
        seq += 1


def capped(outcome: HostToolOutcome, too_big_text: str = _TOO_BIG_TEXT) -> HostToolOutcome:
    if len(outcome.text.encode("utf-8")) > MAX_RESULT_BYTES:
        return HostToolOutcome(too_big_text, ok=False)
    return outcome


def _bounded(timeout: object) -> int:
    return min(max(timeout, 1), MAX_TIMEOUT_S) if isinstance(timeout, int) else DEFAULT_TIMEOUT_S


class HostToolRunner:
    """One remote turn's host tool calls. ``timeouts`` names the tools this
    turn offers (its ``host_tools`` block) with each one's ``timeout_s``."""

    def __init__(
        self,
        *,
        correlation_id: str,
        timeouts: Mapping[str, int],
        execute: ExecuteHostTool,
        too_big_text: str = _TOO_BIG_TEXT,
        r: aioredis.Redis | None = None,
    ) -> None:
        self._corr = correlation_id
        self._timeouts = {name: _bounded(timeout) for name, timeout in timeouts.items()}
        self._execute = execute
        self._too_big_text = too_big_text
        self._r = r
        self._closed = False
        self._waited = False
        self._calls: dict[str, tuple[HostCall, asyncio.Task]] = {}

    def start(self, data: Mapping[str, Any], *, refusal: str | None = None) -> None:
        """Run the call a ``host_tool_call`` step names, as a task; with
        ``refusal``, answer it with that text (``ok: false``) instead of
        running it. Never raises."""
        call_id, tool, raw = data.get("call_id"), data.get("tool"), data.get("input")
        if not isinstance(call_id, str) or not CALL_ID.fullmatch(call_id) or not isinstance(tool, str):
            logger.info("host tool call dropped: malformed correlation_id=%s", self._corr)
            return
        if self._closed or call_id in self._calls:
            return
        call = HostCall(call_id=call_id, tool=tool, input=raw if isinstance(raw, dict) else {})
        self._calls[call_id] = (call, asyncio.create_task(self._run(call, refusal), name=f"host-tool:{call_id}"))

    def pending_tasks(self) -> list[asyncio.Task]:
        return [task for _call, task in self._calls.values() if not task.done()]

    def close(self) -> None:
        """The turn has ended: nothing more is sent to the pod."""
        self._closed = True

    async def finish(self, *, timeout: float | None = None) -> list[HostCall]:
        """Close, then wait for the calls still running, at most ``timeout``
        (TURN_END_WAIT_S by default); the calls still running after it.

        Waits once per turn: the turn end may run more than once (a clean
        finish, then a Stop's save), and a hung call must not delay each of
        them. A later call returns [] at once."""
        self.close()
        if self._waited:
            return []
        self._waited = True
        pending = self.pending_tasks()
        if pending:
            await asyncio.wait(pending, timeout=TURN_END_WAIT_S if timeout is None else timeout)
        return [call for call, task in self._calls.values() if not task.done()]

    async def _run(self, call: HostCall, refusal: str | None = None) -> None:
        r = self._r or get_redis()
        timeout = self._timeouts.get(call.tool)
        try:
            first = await r.set(
                call_key(self._corr, call.call_id), "1", nx=True, ex=(timeout or DEFAULT_TIMEOUT_S) + CALL_GRACE_S,
            )
        except RedisError:
            logger.warning("host tool call not run: Redis failed correlation_id=%s call_id=%s",
                           self._corr, call.call_id, exc_info=True)
            # Best effort: the push may go through where SET timed out, and spares
            # the pod a wait that ends in "may have been applied".
            if not self._closed:
                await self._push(r, call, [_result_line(call.call_id, 0, last=True, ok=False, text=_BUSY_TEXT)])
            return
        if not first:
            logger.info("host tool call already run correlation_id=%s call_id=%s", self._corr, call.call_id)
            return
        if refusal is not None:
            outcome = HostToolOutcome(refusal, ok=False)
        elif timeout is None:
            outcome = HostToolOutcome(
                f"Unknown host tool {call.tool!r}. This turn offers: {', '.join(sorted(self._timeouts))}.", ok=False,
            )
        else:
            outcome = await self._outcome(call, r)
        if self._closed:
            return
        # Capping and cutting up to 1 MiB serialises it many times over: off the event loop.
        try:
            entries = await asyncio.to_thread(self._answer, call.call_id, outcome)
        except Exception:
            logger.exception("host tool answer not built correlation_id=%s call_id=%s", self._corr, call.call_id)
            entries = [_result_line(call.call_id, 0, last=True, ok=False, text=_HOST_ERROR_TEXT)]
        if self._closed:
            return
        await self._push(r, call, entries)

    def _answer(self, call_id: str, outcome: HostToolOutcome) -> list[str]:
        outcome = capped(HostToolOutcome(_encodable(outcome.text), ok=outcome.ok), self._too_big_text)
        return result_entries(call_id, ok=outcome.ok, text=outcome.text)

    async def _outcome(self, call: HostCall, r: aioredis.Redis) -> HostToolOutcome:
        steps = itertools.count()

        async def emit(text: str) -> None:
            if self._closed or not text:
                return
            await self._push(r, call, [progress_entry(call.call_id, next(steps), text)])

        try:
            return await self._execute(call, emit)
        except Exception:
            logger.exception("host tool %s failed correlation_id=%s call_id=%s", call.tool, self._corr, call.call_id)
            return HostToolOutcome(f"{call.tool} failed on the host; check its effect before retrying.", ok=False)

    async def _push(self, r: aioredis.Redis, call: HostCall, entries: list[str]) -> None:
        key = answer_queue_key(self._corr)
        try:
            async with r.pipeline(transaction=True) as pipe:
                for entry in entries:
                    pipe.rpush(key, entry)
                pipe.expire(key, ANSWER_TTL_S)
                await pipe.execute()
        except (RedisError, UnicodeEncodeError, ValueError):
            logger.warning("host tool answer not queued correlation_id=%s call_id=%s",
                           self._corr, call.call_id, exc_info=True)
