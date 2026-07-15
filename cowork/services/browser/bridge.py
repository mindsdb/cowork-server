"""`BridgeCommandService` — the server-side command broker.

The server is the command authority; the Electron **main** process owns the
CDP socket and executes. This broker:

1. `enqueue(...)` a `BridgeCommand` — placed on an in-memory queue keyed by
   `session_id` and registered against an `asyncio.Future` keyed by
   `command_id`.
2. The Electron poller long-polls `next(session_id)` to pull the command,
   executes it over CDP, and calls `resolve(command_id, result)` (via
   `POST /commands/{id}/result`).
3. `execute(...)` = enqueue + await the future under a bounded timeout. A
   command that is never pulled (no poller) or never resolved times out and
   returns a `timeout` result — it NEVER hangs forever and NEVER returns
   `ok`.

This is process-global (single-instance desktop sidecar), mirroring
`cowork.streaming.registry`.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict, deque

from cowork.schemas.browser import (
    BridgeCommand,
    BridgeCommandResult,
    BrowserActionType,
    ResultCode,
)

logger = logging.getLogger(__name__)

# Default bounded timeout for a brokered command. Kept modest so a missing
# poller surfaces as a fast `bridge_disconnected` rather than a long hang.
DEFAULT_COMMAND_TIMEOUT_S = 30.0


class BridgeCommandService:
    """In-memory broker: one queue per session, one future per command."""

    def __init__(self, default_timeout_s: float = DEFAULT_COMMAND_TIMEOUT_S) -> None:
        self._default_timeout_s = default_timeout_s
        self._lock = asyncio.Lock()
        # Pending commands not yet pulled by a poller, FIFO per session.
        self._queues: dict[str, deque[BridgeCommand]] = defaultdict(deque)
        # Futures awaiting a result, keyed by command_id.
        self._futures: dict[str, asyncio.Future[BridgeCommandResult]] = {}
        # Signals a poller waiting on next() that a command is available.
        self._arrivals: dict[str, asyncio.Event] = defaultdict(asyncio.Event)
        # command_id -> session_id, so a resolve/cancel can find the queue.
        self._command_session: dict[str, str] = {}

    def new_command_id(self) -> str:
        return uuid.uuid4().hex

    async def enqueue(self, command: BridgeCommand) -> asyncio.Future[BridgeCommandResult]:
        """Register a command + its result future, wake any waiting poller."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[BridgeCommandResult] = loop.create_future()
        async with self._lock:
            self._futures[command.command_id] = future
            self._command_session[command.command_id] = command.session_id
            self._queues[command.session_id].append(command)
            self._arrivals[command.session_id].set()
        return future

    async def next(
        self, session_id: str, *, wait_s: float = 25.0
    ) -> BridgeCommand | None:
        """Long-poll for the next queued command for a session.

        Returns the command if one is (or becomes) available within
        `wait_s`, else `None` so the poller can re-poll.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_s
        while True:
            async with self._lock:
                queue = self._queues.get(session_id)
                if queue:
                    cmd = queue.popleft()
                    if not queue:
                        self._arrivals[session_id].clear()
                    return cmd
                event = self._arrivals[session_id]
                event.clear()
            remaining = deadline - loop.time()
            if remaining <= 0:
                return None
            try:
                await asyncio.wait_for(event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return None

    async def resolve(self, command_id: str, result: BridgeCommandResult) -> bool:
        """Resolve a command's future with the poster's result.

        Returns False if the command is unknown/already resolved.
        """
        async with self._lock:
            future = self._futures.pop(command_id, None)
            self._command_session.pop(command_id, None)
        if future is None or future.done():
            return False
        future.set_result(result)
        return True

    async def fail(self, command_id: str, result_code: ResultCode, detail: str | None = None) -> bool:
        """Resolve a command with a terminal failure result_code.

        Used when the server itself detects a dead target (tab/Chrome death)
        for an in-flight command — resolves with `target_lost`, never `ok`.
        """
        return await self.resolve(
            command_id,
            BridgeCommandResult(
                command_id=command_id, result_code=result_code, detail=detail
            ),
        )

    async def execute(
        self,
        *,
        session_id: str,
        action_type: BrowserActionType,
        conversation_id: str | None = None,
        domain: str | None = None,
        href: str | None = None,
        direction: str | None = None,
        command_id: str | None = None,
        timeout_s: float | None = None,
    ) -> BridgeCommandResult:
        """Enqueue a command and await its result under a bounded timeout.

        A hung command (no poller pulled it, or the poller never posted a
        result) resolves to `ResultCode.timeout` — NEVER `ok`, NEVER an
        infinite wait.
        """
        cid = command_id or self.new_command_id()
        command = BridgeCommand(
            command_id=cid,
            action_type=action_type,
            session_id=session_id,
            conversation_id=conversation_id,
            domain=domain,
            href=href,
            direction=direction,
        )
        future = await self.enqueue(command)
        timeout = timeout_s if timeout_s is not None else self._default_timeout_s
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            # Clean up so a late poster can't resolve a dead future, and
            # drop the command from the queue if still unclaimed.
            await self._discard(cid, session_id)
            return BridgeCommandResult(
                command_id=cid,
                result_code=ResultCode.timeout,
                detail="command not completed within timeout",
            )
        except asyncio.CancelledError:
            # The awaiting producer was cancelled (e.g. `/responses/cancel`
            # tore down the turn). Discard the command so it can't be pulled
            # by a poller after we've walked away, and so a late poster can't
            # resolve an abandoned future — then re-raise so cancellation
            # propagates. Without this, a cancelled command would leak (a
            # dead future + a still-queued command / stuck `in_flight` row).
            await self._discard(cid, session_id)
            raise

    async def drain_session(
        self, session_id: str, result_code: ResultCode, detail: str | None = None
    ) -> int:
        """Resolve + drop every outstanding command for a session.

        Used when the control gate is set (stopped / taken_over): any queued
        command is removed and every awaiting future for the session is
        resolved with a terminal `result_code` so its producer stops waiting
        instead of hanging until timeout. Returns the number of futures
        resolved.

        A command that is not yet resolved is completed with `result_code`
        (never `ok`); the awaiting `execute()` therefore returns a terminal,
        non-ok result and the action row is persisted `failed`.
        """
        async with self._lock:
            command_ids = [
                cid for cid, sid in self._command_session.items() if sid == session_id
            ]
            # Drop any still-queued commands so a poller can't pull them.
            self._queues.pop(session_id, None)
            self._arrivals[session_id].clear()
        resolved = 0
        for cid in command_ids:
            if await self.resolve(
                cid,
                BridgeCommandResult(
                    command_id=cid, result_code=result_code, detail=detail
                ),
            ):
                resolved += 1
        return resolved

    async def _discard(self, command_id: str, session_id: str) -> None:
        async with self._lock:
            self._futures.pop(command_id, None)
            self._command_session.pop(command_id, None)
            queue = self._queues.get(session_id)
            if queue:
                self._queues[session_id] = deque(
                    c for c in queue if c.command_id != command_id
                )

    def pending_count(self, session_id: str) -> int:
        return len(self._queues.get(session_id, ()))


# Process-global broker (single-instance desktop sidecar).
bridge_command_service = BridgeCommandService()
