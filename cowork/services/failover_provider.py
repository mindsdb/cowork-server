"""A provider that tries an ordered list of (real provider, model) candidates,
rotating to the next one when the current candidate is out of free quota,
unreachable, or can't fit the request in its context window.

This is the mechanism behind "free forever": a specific model pick from
the composer is tried first, then every other enabled registry entry is
appended as a safety net, in priority order. A single free-tier 429
degrades to the next free source instead of failing the turn.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from anton.core.llm.provider import (
    ContextOverflowError,
    LLMProvider,
    LLMResponse,
    StreamEvent,
    TokenLimitExceeded,
)

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    provider: LLMProvider
    model: str
    label: str  # "{slug}/{model}" — for logs and for reporting which candidate served the turn.


class AllCandidatesFailedError(Exception):
    """Raised when every candidate in the failover chain failed."""

    def __init__(self, attempts: list[tuple[str, Exception]]):
        self.attempts = attempts
        detail = "; ".join(f"{label}: {exc}" for label, exc in attempts)
        super().__init__(f"All {len(attempts)} provider candidate(s) failed — {detail}")


# Errors worth rotating to the next candidate for. Anything else (e.g. a
# genuine bug in request construction) is allowed to propagate immediately
# rather than being masked by silently trying every candidate.
_FAILOVER_EXCEPTIONS = (TokenLimitExceeded, ContextOverflowError, ConnectionError)


class FailoverLLMProvider(LLMProvider):
    """Wraps N (provider, model) candidates behind the single LLMProvider interface.

    `model=` passed to complete()/stream() is ignored in favor of each
    candidate's own model id — callers get `LLMClient` to report a fixed
    `planning_model`/`coding_model`, but this class picks per-candidate.
    """

    name = "failover"

    def __init__(self, candidates: list[Candidate]) -> None:
        if not candidates:
            raise ValueError("FailoverLLMProvider requires at least one candidate")
        self._candidates = candidates
        self.last_served_by: str | None = None

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        max_tokens: int = 4096,
        native_web_tools: set[str] | None = None,
    ) -> LLMResponse:
        attempts: list[tuple[str, Exception]] = []
        for candidate in self._candidates:
            try:
                response = await candidate.provider.complete(
                    model=candidate.model,
                    system=system,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    max_tokens=max_tokens,
                    native_web_tools=native_web_tools if candidate.provider.native_web_tools() else None,
                )
                self.last_served_by = candidate.label
                return response
            except _FAILOVER_EXCEPTIONS as exc:
                logger.warning("Provider candidate %s failed, rotating: %s", candidate.label, exc)
                attempts.append((candidate.label, exc))
                continue
        raise AllCandidatesFailedError(attempts)

    async def stream(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        native_web_tools: set[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        attempts: list[tuple[str, Exception]] = []
        for candidate in self._candidates:
            yielded_any = False
            try:
                async for event in candidate.provider.stream(
                    model=candidate.model,
                    system=system,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    native_web_tools=native_web_tools if candidate.provider.native_web_tools() else None,
                ):
                    yielded_any = True
                    yield event
                self.last_served_by = candidate.label
                return
            except _FAILOVER_EXCEPTIONS as exc:
                if yielded_any:
                    # Content already reached the caller (and, downstream, the
                    # user) — restarting on another candidate would duplicate
                    # or contradict what's already on screen. Surface the
                    # error instead of pretending we can cleanly resume.
                    logger.warning(
                        "Provider candidate %s failed mid-stream after partial output; "
                        "not failing over (would duplicate visible content): %s",
                        candidate.label, exc,
                    )
                    raise
                logger.warning("Provider candidate %s failed before any output, rotating: %s", candidate.label, exc)
                attempts.append((candidate.label, exc))
                continue
        raise AllCandidatesFailedError(attempts)

    def native_web_tools(self) -> set[str]:
        # Conservative: only claim native web tools if every candidate
        # supports them, since a mid-chain failover would otherwise silently
        # drop web search/fetch results the caller was told to expect.
        common: set[str] | None = None
        for candidate in self._candidates:
            supported = candidate.provider.native_web_tools()
            common = supported if common is None else (common & supported)
        return common or set()
