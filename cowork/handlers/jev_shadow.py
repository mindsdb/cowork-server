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
import json
import logging
import math
import time
from typing import Any

import httpx

from cowork.common.settings.app_settings import TurnQueueSettings
from cowork.services.providers import MINDS_REQUEST_KIND_HEADER, MINDS_REQUEST_KIND_PROBE

logger = logging.getLogger(__name__)

_VALID_CHOICES = frozenset({"answer_directly", "needs_agent"})

# Mirrors every forced-delegate category in response_routing._SYSTEM_PROMPT,
# not just "needs tools" — otherwise Jev answers a narrower question than the
# gate does, and a measured disagreement reflects that gap, not Jev's own
# accuracy.
_ROUTE_QUESTION: dict[str, Any] = {
    "type": "choice",
    "instructions": (
        "Classify what is required to produce the assistant's next reply. "
        "Return exactly one of: answer_directly or needs_agent. Classify as "
        "needs_agent if any condition listed for it applies; otherwise "
        "classify as answer_directly. When uncertain, choose needs_agent."
    ),
    "criteria": {
        "answer_directly": (
            "The reply can be produced solely from text already visible in "
            "the conversation and stable general knowledge. It requires no "
            "tool use, external retrieval, access to an attachment, file, "
            "account, or dataset, calculation, fact verification, or "
            "multi-step planning. It is not about the assistant's runtime "
            "identity, provider, model, permissions, or available "
            "resources, and it is not about using, installing, updating, "
            "pricing, supporting platforms, or configuring Cowork."
        ),
        "needs_agent": (
            "The reply requires any tool use; browsing or external "
            "retrieval; access to an attachment, file, account, or "
            "dataset; executing or testing code; verifying potentially "
            "time-sensitive information; a calculation; or a multi-step "
            "plan. Also choose this for questions about the assistant's "
            "runtime identity, provider, model, permissions, or available "
            "resources, and for questions about using, installing, "
            "updating, pricing, supporting platforms, or configuring "
            "Cowork. Choose this option when uncertain."
        ),
    },
}


# Separates the probe's Jev traces from the gate's own ("cowork-gate") and from
# user turns in Langfuse.
JEV_SHADOW_TAG = "jev-shadow"


def _elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


def _trace_headers() -> dict[str, str]:
    """Langfuse headers attributing the probe to the turn whose gate it shadows.

    Without them the gateway stamps the call ``origin:direct-api``: from its
    side a probe on the user's minted key is indistinguishable from that user
    calling Jev directly. The probe runs under the gate's anton
    TraceContext (``_spawn_jev_shadow_probe``), so this reads it rather than
    re-deriving the turn's identity.

    Deliberately narrower than anton's own header builder, which serves LLM
    calls: the gate's ``cowork-gate`` tag is not forwarded (the probe is not the
    gate call), and ``turn_id`` is never sent — the gateway renames any trace
    carrying harness + turn_id to ``{harness}:turn-N``, which would count every
    probe as a user turn.

    No ``Langfuse-Session-Id`` either: the gateway stores it as the row's
    ``session_id``, and the customer's Traces *Sessions* view counts every row
    in a session regardless of ``request_kind`` — so the probe would appear in
    the user's own conversation, and a failed probe would mark it failed. The
    conversation rides in the metadata instead, and ``correlation_id`` joins
    the probe to the gate trace, which does carry the session.
    """
    from anton.core.llm.tracing import get_trace_context, surface_tag

    ctx = get_trace_context()
    if ctx is None:
        return {}
    headers: dict[str, str] = {}
    tags = [ctx.harness, surface_tag(ctx.surface) if ctx.surface else None, JEV_SHADOW_TAG]
    headers["Langfuse-Tags"] = ",".join(t for t in tags if t)
    metadata: dict[str, object] = dict(ctx.metadata or {})
    metadata.pop("turn_id", None)
    if ctx.harness:
        # What makes the gateway stamp origin:harness rather than direct-api.
        metadata["harness"] = ctx.harness
    if ctx.surface:
        metadata["surface"] = ctx.surface
    if ctx.session_id:
        metadata["conversation_id"] = ctx.session_id
    headers["Langfuse-Metadata"] = json.dumps(metadata)
    return headers


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
    without tripping it. The probe runs detached from the turn it shadows
    (`_spawn_jev_shadow_probe`), so a slow probe never delays the turn, but an
    unbounded one would still hold its task and connection open indefinitely.
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
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        # Our experiment, not the user's request: hides the row
                        # from their Traces list by default, like the key-test
                        # pings. It only reaches Postgres, never Langfuse, so
                        # our own attribution below is unaffected.
                        MINDS_REQUEST_KIND_HEADER: MINDS_REQUEST_KIND_PROBE,
                        **_trace_headers(),
                    },
                    json=payload,
                )
    except TimeoutError:
        return {"jev_ms": _elapsed_ms(started), "jev_error": "timeout"}
    except httpx.HTTPError:
        return {"jev_ms": _elapsed_ms(started), "jev_error": "transport_error"}
    except Exception:
        logger.warning("[jev-shadow] probe failed", exc_info=True)
        return {"jev_ms": _elapsed_ms(started), "jev_error": "exception"}

    elapsed_ms = _elapsed_ms(started)
    if response.status_code != 200:
        return {"jev_ms": elapsed_ms, "jev_error": f"http_{response.status_code}"}
    try:
        body = response.json()
        answer = body["answers"]["route"]
        choice = answer["choice"]
        confidence = answer["confidence"]
        if choice not in _VALID_CHOICES:
            raise ValueError(f"unexpected choice {choice!r}")
        # bool is an int subclass; excluded explicitly so a JSON `true`/`false`
        # can't pass as a numeric 1/0 confidence.
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError(f"confidence is not numeric: {confidence!r}")
        if not math.isfinite(confidence) or not (0 <= confidence <= 1):
            raise ValueError(f"confidence out of range: {confidence!r}")
        return {
            "jev_ms": elapsed_ms,
            "jev_choice": choice,
            "jev_confidence": confidence,
            "jev_model": body.get("model"),
        }
    except Exception:
        logger.warning("[jev-shadow] malformed response", exc_info=True)
        return {"jev_ms": elapsed_ms, "jev_error": "malformed_response"}
