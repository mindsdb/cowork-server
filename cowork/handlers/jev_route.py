"""The route gate on MindsHub: Jev (TypeSafe, served via MindsHub's
``/v1/decisions``) decides whether a turn is answered by the fast direct path
or delegated to the agent.

Jev only decides. The direct answer is written by the user's own model
(`direct_answer`), which can still hand the turn to the agent, so the question
leans direct. Its wording and the default threshold were chosen with
``scripts/eval_jev_route`` (results in that folder): rerun it before changing
either.

Every failure, a bad response, a timeout, a network error, comes back as an
`error` on the decision, never raised, and the caller delegates: the agent is
always a correct answer, only a slower one.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx

_ANSWER_DIRECTLY = "answer_directly"
_VALID_CHOICES = frozenset({_ANSWER_DIRECTLY, "needs_agent"})

# Describes the direct path as built: one call to the user's model that sees
# the conversation's visible text, knows its deployment's identity, can search
# the web, and can delegate. Tool markers are the format `_condense_content`
# writes; Jev is told that omitted output is invisible to the fast responder,
# which is what keeps a follow-up about tool-only numbers on the agent.
ROUTE_QUESTION: dict[str, Any] = {
    "type": "choice",
    "instructions": (
        "Decide who should write the assistant's next reply to the latest user "
        "message. The fast responder is one chat model call that sees the visible "
        "text of this conversation, knows which product, model and provider it is, "
        "and can run a web search; it cannot use the user's data, files, accounts "
        "or code. A line '[ran tool: X]' means the agent did work with tool X; "
        "'[tool output omitted]' means that output is NOT visible to the fast "
        "responder. Choose answer_directly when the fast responder can write a good "
        "reply; otherwise choose needs_agent."
    ),
    "criteria": {
        "answer_directly": (
            "Greetings, thanks and small talk; general knowledge and explanations; "
            "restating, summarizing, rephrasing or translating what the assistant "
            "already wrote in its visible replies, including numbers it stated; "
            "simple reasoning over values stated in visible text; current events, "
            "prices, weather or other public facts one web search can answer; "
            "questions about the assistant's own model, provider or maker, or about "
            "the product's features, installation, updates or pricing."
        ),
        "needs_agent": (
            "The reply needs a value that is not in the visible text, for example "
            "one only in omitted tool output, a file, a dashboard or the user's "
            "data; new or changed analysis, charts, dashboards, files or code; "
            "querying, refreshing or exporting the user's data; questions about "
            "this user's own connections, files or account; research that needs "
            "many searches or a multi-step plan."
        ),
    },
}

# Separates the gate's Jev traces from user turns in Langfuse; the same tag
# the LLM gate's calls carry, since this is that gate.
JEV_ROUTE_TAG = "cowork-gate"


@dataclass(frozen=True)
class JevTarget:
    """Where to ask: the decision endpoint's full URL and its key.

    Full, not a base URL: hosts serve the same body at different paths
    (MindsHub ``/v1/decisions``, TypeSafe ``/v1/systemone``)."""

    url: str
    api_key: str

    @classmethod
    def mindshub(cls, base_url: str, api_key: str) -> "JevTarget":
        """MindsHub's endpoint under its inference base URL (incl. /v1)."""
        return cls(url=f"{base_url.rstrip('/')}/decisions", api_key=api_key)


@dataclass(frozen=True)
class JevRoute:
    """One decision. Content-free: a verdict, numbers and an error class."""

    direct: bool = False
    p_direct: float | None = None
    ms: int = 0
    model: str = ""
    # Empty on a usable answer; otherwise why the turn delegates by default.
    error: str = ""


GateMode = Literal["own", "jev", "off", "zdr", "llm"]

# MindsHub's reason for refusing a model to an organization with zero data
# retention on, whose agreements do not cover the model's provider (Jev's,
# TypeSafe, today). Read from the model listing's `disabled_reason`.
ZDR_RESTRICTED = "zdr_restricted"


def zdr_restricted(decision_disabled_reasons, model: str) -> bool:
    """Whether MindsHub refuses the decision model `model` to this
    organization for zero data retention (`MindsModelListing.decision_disabled_reasons`)."""
    return (decision_disabled_reasons or {}).get(model) == ZDR_RESTRICTED


def own_decision_target(settings) -> "JevTarget | None":
    """The organization's own decision model, when both its URL and key are set."""
    key = settings.decision_model_api_key
    key = key.get_secret_value() if key is not None else ""
    url = (settings.decision_model_url or "").strip()
    return JevTarget(url=url, api_key=key) if key and url else None


def gate_mode(settings, *, zdr: bool = False) -> GateMode:
    """Which gate decides a turn, from settings alone. One source for the turn
    handler and for the settings response, so the UI shows what turns do.

    - "own": the organization's own decision model, for any provider.
      Configuring it is the opt-in, so the Jev opt-out does not apply.
    - "jev": Jev on MindsHub, for turns that answer on MindsHub.
    - "off": the organization turned Jev on MindsHub off and has no decision
      model of its own: MindsHub turns run the agent.
    - "zdr": as "off", but because MindsHub refuses Jev to the organization
      under zero data retention (`zdr`, from `zdr_restricted`), whatever the
      switch says.
    - "llm": the LLM gate, for turns that answer on other providers.
    """
    from cowork.common.settings.user_settings import Provider

    if own_decision_target(settings) is not None:
        return "own"
    if settings.resolved_planning_provider is not Provider.MINDS_CLOUD:
        return "llm"
    if zdr:
        return "zdr"
    return "jev" if settings.jev_decisions_enabled else "off"


def _elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


def route_state(messages: list[dict]) -> dict:
    """The gate's text history as Jev state, the latest message apart."""
    return {"latest_user_message": messages[-1]["content"], "earlier_conversation": messages[:-1]}


def _trace_headers() -> dict[str, str]:
    """Langfuse headers attributing the decision to the turn it routes.

    Without them the gateway stamps the call ``origin:direct-api``: from its
    side a call on the user's key is indistinguishable from that user calling
    Jev directly. The call runs under the gate's anton TraceContext
    (`ResponsesHandler._route_request`), so this reads it rather than
    re-deriving the turn's identity.

    Deliberately narrower than anton's own header builder, which serves LLM
    calls: ``turn_id`` is never sent — the gateway renames any trace carrying
    harness + turn_id to ``{harness}:turn-N``, which would count every decision
    as a user turn. No ``Langfuse-Session-Id`` either: the customer's Traces
    *Sessions* view marks a conversation failed when any row in it fails, and a
    refused decision is one the turn recovers from by delegating. The
    conversation rides in the metadata instead, and ``correlation_id`` joins
    the decision to the rest of the turn.
    """
    from anton.core.llm.tracing import get_trace_context, surface_tag

    ctx = get_trace_context()
    if ctx is None:
        return {}
    headers: dict[str, str] = {}
    tags = [ctx.harness, surface_tag(ctx.surface) if ctx.surface else None, JEV_ROUTE_TAG]
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


def _p_direct(answer: dict) -> float:
    """The probability of a direct answer, validated."""
    value = (answer.get("probabilities") or {}).get(_ANSWER_DIRECTLY)
    if value is None:
        raise ValueError("no probability for answer_directly")
    # bool is an int subclass; a JSON true must not pass as probability 1.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"probability is not a finite number: {value!r}")
    if not 0 <= value <= 1:
        raise ValueError(f"probability out of range: {value!r}")
    return float(value)


async def decide(
    *,
    target: JevTarget,
    messages: list[dict],
    model: str,
    timeout_seconds: float,
    min_p_direct: float,
) -> JevRoute:
    """One `/v1/decisions` call: direct when Jev gives a direct answer at
    least `min_p_direct`. Never raises.

    Wrapped in `asyncio.timeout`, not just httpx's own timeout kwarg: httpx's
    applies per phase (connect/write/read/pool) and measures inactivity, not
    total elapsed time, so a trickling response could run past the budget
    without tripping it. This call sits ahead of every turn on MindsHub.
    """
    payload = {"model": model, "state": route_state(messages), "questions": {"route": ROUTE_QUESTION}}
    started = time.monotonic()
    try:
        async with asyncio.timeout(timeout_seconds):
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(
                    target.url,
                    headers={**_trace_headers(), "Authorization": f"Bearer {target.api_key}"},
                    json=payload,
                )
    except TimeoutError:
        return JevRoute(ms=_elapsed_ms(started), error="timeout")
    except httpx.HTTPError:
        return JevRoute(ms=_elapsed_ms(started), error="transport_error")
    except Exception as exc:  # the gate must never raise into the turn
        return JevRoute(ms=_elapsed_ms(started), error=type(exc).__name__)

    ms = _elapsed_ms(started)
    if response.status_code != 200:
        return JevRoute(ms=ms, error=f"http_{response.status_code}")
    try:
        body = response.json()
        answer = body["answers"]["route"]
        if answer["choice"] not in _VALID_CHOICES:
            raise ValueError(f"unexpected choice {answer['choice']!r}")
        p_direct = _p_direct(answer)
    except Exception:  # any shape problem is one error class
        return JevRoute(ms=ms, error="malformed_response")
    return JevRoute(
        direct=p_direct >= min_p_direct, p_direct=p_direct, ms=ms, model=str(body.get("model") or ""),
    )
