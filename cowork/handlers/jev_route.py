"""The route gate on MindsHub: Jev (TypeSafe, served via MindsHub's
``/v1/decisions``) decides whether a turn is answered by the fast direct path
or delegated to the agent.

Jev only decides. The direct answer is written by the user's own model
(`direct_answer`), which can still hand the turn to the agent, so the question
leans direct. Its wording and the default threshold were chosen with
``scripts/eval_jev_route`` (results in that folder): rerun it before changing
either.

Every failure, a refusal, a bad response, a timeout, a network error, comes
back as an `error` on the decision, never raised. Jev bills like any MindsHub
model, so a refusal (a spent allowance, a restriction on the organization) is
an ordinary failure: the caller falls back to the LLM gate, or to the agent
when Jev ran out the gate's time (`falls_back_to_llm_gate`).
"""
from __future__ import annotations

import asyncio
import json
import math
import socket
import time
from collections.abc import Callable
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
            "many searches or a multi-step plan; any request to change, send, "
            "confirm or approve something, and any reply to a question the "
            "assistant asked earlier in the conversation."
        ),
    },
}

# Asked in the same call: whether the reply needs a web search. The direct
# answer waits for its first text far longer when it does (the search runs on
# the gateway before any text), so a slow wrong route costs less when it does
# not. Never decides the route.
_SEARCH = "search_web"
_SEARCH_CHOICES = frozenset({_SEARCH, "no_search"})
SEARCH_QUESTION: dict[str, Any] = {
    "type": "choice",
    "instructions": (
        "Decide whether a good reply to the latest user message needs a web "
        "search for current or public facts that are not in the conversation."
    ),
    "criteria": {
        _SEARCH: (
            "Current events, news, prices, weather, schedules, recent releases, or "
            "any public fact that may have changed or that a careful answer would check."
        ),
        "no_search": (
            "Greetings and small talk; restating or reasoning over what the "
            "conversation already says; stable general knowledge; questions about "
            "the product or the assistant itself; anything the agent must do."
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
    # MindsHub's own endpoint: only it reads the Langfuse trace headers, and
    # they carry user, organization and conversation ids, so an organization's
    # own decision endpoint never receives them.
    mindshub_gateway: bool = False

    @classmethod
    def mindshub(cls, base_url: str, api_key: str) -> "JevTarget":
        """MindsHub's endpoint under its inference base URL (incl. /v1)."""
        return cls(url=f"{base_url.rstrip('/')}/decisions", api_key=api_key, mindshub_gateway=True)


@dataclass(frozen=True)
class JevRoute:
    """One decision. Content-free: a verdict, numbers and an error class."""

    direct: bool = False
    p_direct: float | None = None
    # Whether Jev expects the reply to need a web search. True when it did not
    # say, so the direct answer keeps its longer wait.
    needs_search: bool = True
    ms: int = 0
    model: str = ""
    # Empty on a usable answer; otherwise why Jev could not decide.
    error: str = ""

    @property
    def falls_back_to_llm_gate(self) -> bool:
        """Whether a failed decision should ask the LLM gate instead. Every
        failure does, except a timeout: that one already spent the gate's
        time budget, so the turn goes straight to the agent."""
        return bool(self.error) and self.error != "timeout"


GateMode = Literal["own", "jev", "off", "llm"]


def own_decision_target(settings) -> "JevTarget | None":
    """The organization's own decision model, when both its URL and key are set."""
    key = settings.decision_model_api_key
    key = key.get_secret_value() if key is not None else ""
    url = (settings.decision_model_url or "").strip()
    return JevTarget(url=url, api_key=key) if key and url else None


def gate_mode(settings) -> GateMode:
    """Which gate decides a turn, from settings alone. One source for the turn
    handler and for the settings response, so the UI shows what turns do.

    - "own": the organization's own decision model, for any provider.
      Configuring it is the opt-in, so the Jev opt-out does not apply.
    - "jev": Jev on MindsHub, for turns that answer on MindsHub.
    - "off": the organization turned Jev on MindsHub off and has no decision
      model of its own: MindsHub turns run the agent.
    - "llm": the LLM gate, for turns that answer on other providers.

    A decision MindsHub refuses (an organization it does not serve Jev to)
    needs no mode of its own: the turn falls back to the LLM gate.
    """
    from cowork.common.settings.user_settings import Provider

    if own_decision_target(settings) is not None:
        return "own"
    if settings.resolved_planning_provider is not Provider.MINDS_CLOUD:
        return "llm"
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


# One client per event loop keeps the connection to the decision endpoint open
# across turns: a client per call paid a TCP and TLS handshake inside the
# gate's budget on every turn. An httpx client is bound to the loop it first
# ran on, so a loop change (tests run one per test) gets a fresh client.
# Timeouts are per request, from the caller's budget.
_client: tuple[asyncio.AbstractEventLoop, httpx.AsyncClient] | None = None


def _http_client() -> httpx.AsyncClient:
    global _client
    loop = asyncio.get_running_loop()
    if _client is None or _client[0] is not loop or _client[1].is_closed:
        _client = (loop, httpx.AsyncClient())
    return _client[1]


# Replaced in tests.
_RESOLVER: Callable[..., list[tuple]] = socket.getaddrinfo
_PINNED_TRANSPORT: httpx.AsyncBaseTransport | None = None


class _EndpointRefused(Exception):
    """An organization's own decision endpoint this server will not dial."""


def _hosted() -> bool:
    from cowork.common.settings.app_settings import get_app_settings

    return get_app_settings().tenancy_mode == "org"


async def _post_pinned(target: "JevTarget", *, headers: dict, payload: dict, timeout_seconds: float) -> httpx.Response:
    """Post to an organization's own endpoint from the hosted server, at an
    address vetted as public (the connectors' pattern, `connectors.egress`).

    An organization admin sets the URL, so without this the server would post
    to any internal address it can reach, every turn. https only, resolved
    once, and the vetted address dialed as a literal, with the hostname kept on
    ``Host`` and SNI so the certificate is still checked against it. A client
    of its own per call: a shared pool keys connections by address, and two
    hostnames behind one address would share a connection whose certificate
    was checked for only one of them. trust_env stays off, since a proxy
    would tunnel to the pinned address and defeat the pin.
    """
    from cowork.services.connectors.egress import (
        EgressHostNotPublic,
        EgressHostUnresolved,
        connection_attempts,
        vetted_public_addresses,
    )

    url = httpx.URL(target.url)
    if url.scheme != "https" or not url.host:
        raise _EndpointRefused("not https")
    try:
        addresses = await asyncio.to_thread(vetted_public_addresses, url.host, _RESOLVER)
    except (EgressHostUnresolved, EgressHostNotPublic) as exc:
        raise _EndpointRefused("not public") from exc
    headers = {**headers, "Host": url.netloc.decode("ascii")}
    extensions = {"sni_hostname": url.host}
    attempts = connection_attempts(addresses, total_seconds=timeout_seconds, fallback_seconds=1.0)
    async with httpx.AsyncClient(follow_redirects=False, trust_env=False, transport=_PINNED_TRANSPORT) as client:
        for index, (address, connect_seconds) in enumerate(attempts):
            try:
                return await client.post(
                    url.copy_with(host=str(address)),
                    headers=headers,
                    json=payload,
                    extensions=extensions,
                    timeout=httpx.Timeout(timeout_seconds, connect=connect_seconds),
                )
            except (httpx.ConnectError, httpx.ConnectTimeout):
                # Only a failed connect moves on: the key is sent after it.
                if index == len(attempts) - 1:
                    raise
    raise AssertionError("unreachable")  # connection_attempts never returns []


async def close_client() -> None:
    """Close the shared client during shutdown. Idempotent."""
    global _client
    if _client is None:
        return
    loop, client = _client
    _client = None
    if loop is asyncio.get_running_loop():
        await client.aclose()


def _probability(answer: dict, choice: str) -> float:
    """The probability Jev gives `choice`, validated."""
    value = (answer.get("probabilities") or {}).get(choice)
    if value is None:
        raise ValueError(f"no probability for {choice}")
    # bool is an int subclass; a JSON true must not pass as probability 1.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"probability is not a finite number: {value!r}")
    if not 0 <= value <= 1:
        raise ValueError(f"probability out of range: {value!r}")
    return float(value)


def _needs_search(body: dict) -> bool:
    """Jev's answer to the search question; True when it is missing or
    malformed, which only costs the direct answer a longer wait."""
    try:
        answer = body["answers"]["search"]
        if answer["choice"] not in _SEARCH_CHOICES:
            return True
        return _probability(answer, _SEARCH) >= 0.5
    except Exception:
        return True


async def decide(
    *,
    target: JevTarget,
    messages: list[dict],
    model: str,
    timeout_seconds: float,
    min_p_direct: float,
) -> JevRoute:
    """One `/v1/decisions` call: direct when Jev gives a direct answer at
    least `min_p_direct`, and whether the reply needs a web search. Never
    raises.

    Wrapped in `asyncio.timeout`, not just httpx's own timeout kwarg: httpx's
    applies per phase (connect/write/read/pool) and measures inactivity, not
    total elapsed time, so a trickling response could run past the budget
    without tripping it. This call sits ahead of every turn on MindsHub.
    """
    payload = {
        "model": model,
        "state": route_state(messages),
        "questions": {"route": ROUTE_QUESTION, "search": SEARCH_QUESTION},
    }
    started = time.monotonic()
    try:
        async with asyncio.timeout(timeout_seconds):
            headers = {
                **(_trace_headers() if target.mindshub_gateway else {}),
                "Authorization": f"Bearer {target.api_key}",
            }
            if not target.mindshub_gateway and _hosted():
                response = await _post_pinned(
                    target, headers=headers, payload=payload, timeout_seconds=timeout_seconds,
                )
            else:
                response = await _http_client().post(
                    target.url, headers=headers, json=payload, timeout=timeout_seconds,
                )
    except _EndpointRefused:
        # Falls back to the LLM gate, as any failure but a timeout does.
        return JevRoute(ms=_elapsed_ms(started), error="endpoint_refused")
    except (TimeoutError, httpx.TimeoutException):
        # httpx's own timeouts fire at the same budget: the gate's time is
        # spent either way, so both read as a timeout (no LLM gate after it).
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
        p_direct = _probability(answer, _ANSWER_DIRECTLY)
    except Exception:  # any shape problem is one error class
        return JevRoute(ms=ms, error="malformed_response")
    return JevRoute(
        direct=p_direct >= min_p_direct, p_direct=p_direct, needs_search=_needs_search(body),
        ms=ms, model=str(body.get("model") or ""),
    )
