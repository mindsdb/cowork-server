"""The turn's ``preview_checker``: ask the open preview whether an artifact runs.

anton's artifact editor calls ``await preview_checker(artifact_slug,
timeout_s)`` after an edit and reads the answer as its "preview" check:

* a dict ``{"status": "passed" | "failed", "errors": [...], "ms": int}`` when
  the web app answered, or
* ``None`` when nothing could answer (no viewer open, no client on the
  stream, timeout, or the turn ended): anton records the check as unverified.

The question goes out as one ``response.artifact_check`` frame written
straight to the turn's buffer, like the liveness ticker's frames
(streaming/liveness.py), not through the formatter's ``event_sink``. So it is
never saved to the conversation's events: a check means something only while
this turn waits on it, and replaying it on a reload would make the app reload
a preview for an id that is already gone. A client that reconnects mid-turn
tails the buffer and still sees it.

Never raises into anton: any failure here reads as "unverified".
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable

from cowork.streaming.preview_checks import PreviewCheckBroker, preview_checks
from cowork.streaming.sse import sse_frame

logger = logging.getLogger(__name__)

__all__ = [
    "ARTIFACT_CHECK_EVENT",
    "MAX_TIMEOUT_S",
    "MIN_TIMEOUT_S",
    "PreviewChecker",
    "artifact_check_sse",
    "build_preview_checker",
    "preview_checker_kwargs",
]

ARTIFACT_CHECK_EVENT = "response.artifact_check"

# The app reloads the preview and waits for load plus a 3 s settle, so less
# than this cannot be answered; more than this holds a tool call open for
# longer than any preview needs to boot.
MIN_TIMEOUT_S = 5
MAX_TIMEOUT_S = 120
# Bounds the slug that goes on the wire; anton's slugs are short folder names.
_MAX_SLUG = 200

PreviewChecker = Callable[[str, float], Awaitable["dict | None"]]


def artifact_check_sse(*, check_id: str, artifact_slug: str, timeout_s: int) -> str:
    """One ``response.artifact_check`` frame as an SSE string."""
    return sse_frame(ARTIFACT_CHECK_EVENT, {
        "type": ARTIFACT_CHECK_EVENT,
        "check_id": check_id,
        "artifact_slug": artifact_slug,
        "timeout_s": timeout_s,
    })


def _clamp_timeout(timeout_s: object) -> int:
    try:
        value = float(timeout_s)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return MIN_TIMEOUT_S
    if math.isnan(value):
        return MIN_TIMEOUT_S
    if math.isinf(value):
        return MAX_TIMEOUT_S if value > 0 else MIN_TIMEOUT_S
    return min(max(math.ceil(value), MIN_TIMEOUT_S), MAX_TIMEOUT_S)


def _default_buffer_for(conversation_id: str):
    """The live turn's stream buffer, or None when no turn can write to it.

    Imported late: the registry pulls in the stream backends, which this
    module does not need to import at load time.
    """
    from cowork.streaming.registry import registry

    handle = registry.get(conversation_id)
    if handle is None or not handle.is_answering:
        return None
    return handle.buffer


def build_preview_checker(
    conversation_id: str,
    *,
    broker: PreviewCheckBroker = preview_checks,
    buffer_for: Callable[[str], object] = _default_buffer_for,
) -> PreviewChecker:
    """The ``preview_checker`` for one turn of *conversation_id*."""

    async def preview_checker(artifact_slug: str, timeout_s: float) -> dict | None:
        slug = str(artifact_slug or "").strip()
        if not slug or len(slug) > _MAX_SLUG:
            return None
        buffer = buffer_for(conversation_id)
        if buffer is None:
            # Nothing tails this turn here (a remote turn, or the turn is
            # already finishing), so no app could answer.
            return None
        timeout = _clamp_timeout(timeout_s)
        check_id, future = broker.open(conversation_id)
        try:
            try:
                await buffer.append("sse", {"sse": artifact_check_sse(
                    check_id=check_id, artifact_slug=slug, timeout_s=timeout,
                )})
            except Exception:
                logger.warning("Could not publish preview check %s", check_id, exc_info=True)
                return None
            try:
                answer = await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
            except asyncio.TimeoutError:
                logger.info("Preview check %s for %s got no answer in %ss", check_id, slug, timeout)
                return None
        finally:
            broker.close(check_id)
        if not isinstance(answer, dict) or answer.get("status") not in ("passed", "failed"):
            # None (turn ended) or "unavailable" (no viewer for this artifact).
            return None
        return answer

    return preview_checker


def preview_checker_kwargs(config_cls, checker: PreviewChecker | None) -> dict:
    """``{"preview_checker": checker}`` when the installed anton's
    ``ChatSessionConfig`` declares the field, else ``{}``.

    The field lands in anton separately; until then passing it would raise on
    every turn. Same rule as ``build_info.supported_kwargs``, and a None
    checker passes nothing so anton keeps its own default.
    """
    if checker is None:
        return {}
    from cowork.build_info import supported_kwargs

    return supported_kwargs(config_cls, preview_checker=checker)
