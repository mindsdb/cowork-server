"""Shadow-mode timing probe for Jev (TypeSafe, served via MindsHub's
``/v1/decisions``) against Cowork's own respond-vs-delegate gate.

Asks Jev the same question the LLM gate answers, on the same turn, purely for
comparison. Never used to route: `probe`'s result is not read by
`decide_route` and must never affect it. Any failure here, a bad response,
a timeout, a network error, is swallowed and logged; a broken shadow probe
must never break or slow down a real turn beyond its own timeout.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from cowork.common.settings.app_settings import TurnQueueSettings

logger = logging.getLogger(__name__)

# Mirrors response_routing._SYSTEM_PROMPT's respond-vs-delegate split, phrased
# as Jev criteria instead of a system prompt.
_ROUTE_QUESTION: dict[str, Any] = {
    "type": "choice",
    "instructions": (
        "Given this conversation, can the assistant's next reply be written "
        "directly from the conversation and stable general knowledge, or does "
        "it need to run code, browse, access files, retrieve data, or "
        "otherwise use tools?"
    ),
    "criteria": {
        "answer_directly": (
            "The next reply can be written from the conversation and stable "
            "general knowledge alone, with no tools."
        ),
        "needs_agent": (
            "The next reply needs code execution, file or data access, "
            "browsing, or other tool use."
        ),
    },
}


async def probe(
    *,
    messages: list[dict],
    llm_block: dict | None,
    settings: TurnQueueSettings,
) -> dict[str, Any] | None:
    """One `/v1/decisions` call on the turn's minted minds-cloud credential.

    Returns `jev_*` fields for logging, or None when shadow mode is off or no
    minted credential is available (desktop/BYOK turns mint no such block, so
    they're out of scope for this probe).
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
        async with httpx.AsyncClient(timeout=settings.jev_shadow_timeout_seconds) as client:
            response = await client.post(
                f"{base_url}/decisions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
        elapsed_ms = round((time.monotonic() - started) * 1000)
        if response.status_code != 200:
            return {"jev_ms": elapsed_ms, "jev_error": f"http_{response.status_code}"}
        answer = response.json()["answers"]["route"]
        return {
            "jev_ms": elapsed_ms,
            "jev_choice": answer["choice"],
            "jev_confidence": answer["confidence"],
        }
    except Exception:
        # Never propagates: a shadow probe crashing a real turn would be a
        # worse outcome than the measurement it exists to take.
        logger.info("[jev-shadow] probe failed", exc_info=True)
        return {"jev_ms": round((time.monotonic() - started) * 1000), "jev_error": "exception"}
