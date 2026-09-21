"""Shadow-mode timing probe for Jev (TypeSafe, served via MindsHub's
``/v1/decisions``) against Cowork's own respond-vs-delegate gate.

Asks Jev the same question the LLM gate answers, on the same turn, purely for
comparison. Never used to route: `probe`'s result is not read by
`decide_route` and must never affect it. Any failure here, a bad response,
a timeout, a network error, is swallowed and logged; a broken shadow probe
must never break or slow down a real turn beyond its own timeout.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from cowork.common.settings.app_settings import TurnQueueSettings

logger = logging.getLogger(__name__)

# Mirrors every forced-delegate category in response_routing._SYSTEM_PROMPT,
# not just "needs tools" — otherwise Jev answers a narrower question than the
# gate does, and a measured disagreement reflects that gap, not Jev's own
# accuracy.
_ROUTE_QUESTION: dict[str, Any] = {
    "type": "choice",
    "instructions": (
        "Given this conversation, decide whether the assistant's next reply "
        "can be written directly, or needs the full agent. Match these rules "
        "exactly, even where a plain answer seems possible."
    ),
    "criteria": {
        "answer_directly": (
            "The next reply can be written from the conversation and stable "
            "general knowledge alone: no tools, no browsing, no file or data "
            "access, no calculation, no verifying a fact that may be stale, "
            "no multi-step plan, and it is not a question about the "
            "assistant itself (which model or provider is running, what it "
            "can access) or about the Cowork product (what it is, install, "
            "update, cost, platforms, where a setting lives)."
        ),
        "needs_agent": (
            "The next reply needs code execution, file or data access, "
            "browsing, a calculation, verifying a fact that may be stale, a "
            "multi-step plan, or it is a question about the assistant "
            "itself or about the Cowork product."
        ),
    },
}


def _elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


async def probe(
    *,
    messages: list[dict],
    llm_block: dict | None,
    settings: TurnQueueSettings,
) -> dict[str, Any] | None:
    """One `/v1/decisions` call on the turn's minted minds-cloud credential.

    Returns `jev_*` fields for logging, or None when shadow mode is off or no
    minted credential is available (desktop/BYOK turns mint no such block).

    Wrapped in `asyncio.timeout`, not just httpx's own timeout kwarg: httpx's
    applies per phase (connect/write/read/pool) and measures inactivity, not
    total elapsed time, so a trickling response could run past the budget
    without tripping it — and since `_route_request` awaits this via
    `asyncio.gather` alongside the gate, an unbounded probe would hold up the
    real turn, not just log a slow number.
    """
    if not settings.jev_shadow_enabled or not llm_block:
        return None
    base_url = str(llm_block.get("base_url") or "").rstrip("/")
    api_key = llm_block.get("api_key")
    if not base_url or not api_key:
        return None

    payload = {
        "model": settings.jev_shadow_model,
        "state": messages,
        "questions": {"route": _ROUTE_QUESTION},
    }
    started = time.monotonic()
    try:
        async with asyncio.timeout(settings.jev_shadow_timeout_seconds):
            async with httpx.AsyncClient(timeout=settings.jev_shadow_timeout_seconds) as client:
                response = await client.post(
                    f"{base_url}/decisions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                )
    except TimeoutError:
        return {"jev_ms": _elapsed_ms(started), "jev_error": "timeout"}
    except httpx.HTTPError:
        return {"jev_ms": _elapsed_ms(started), "jev_error": "transport_error"}
    except Exception:
        logger.info("[jev-shadow] probe failed", exc_info=True)
        return {"jev_ms": _elapsed_ms(started), "jev_error": "exception"}

    elapsed_ms = _elapsed_ms(started)
    if response.status_code != 200:
        return {"jev_ms": elapsed_ms, "jev_error": f"http_{response.status_code}"}
    try:
        body = response.json()
        answer = body["answers"]["route"]
        return {
            "jev_ms": elapsed_ms,
            "jev_choice": answer["choice"],
            "jev_confidence": answer["confidence"],
            "jev_model": body.get("model"),
        }
    except Exception:
        logger.info("[jev-shadow] malformed response", exc_info=True)
        return {"jev_ms": elapsed_ms, "jev_error": "malformed_response"}
