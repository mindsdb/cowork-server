"""Cowork's first response-routing decision.

This is deliberately a server-owned front gate: a direct decision skips Anton
initialization (and, hosted, the remote turn dispatch) entirely.  Anton has an
equivalent in-process gate (``anton.core.llm.thalamus``), but it is off unless
the harness passes ``router_enabled``, so today this is the only gate a turn
meets.  A direct decision is only valid for a text-only conversational turn;
every uncertain or unsupported shape delegates to Anton.
"""
from __future__ import annotations

import asyncio
import logging
import re
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal

from cowork.common.settings.user_settings import get_user_settings
from cowork.services.providers import build_llm_client


logger = logging.getLogger(__name__)

DIRECT_CONTEXT = "direct_context"
DELEGATED_AGENTIC = "delegated_agentic"

# Keep the gate's context bounded.  It does not need Anton's full tool history.
_MAX_HISTORY_MESSAGES = 16
_MAX_MESSAGE_CHARS = 1_500
_DIRECT_MAX_TOKENS = 1_024
# Three budgets, because the gate has two phases with different needs.
#
# The DECISION — delegate or answer — is the first streamed event, and it is
# what sits ahead of every turn: on the ~100% delegate path a tool call is the
# first thing the model emits, so this bound is the whole cost the gate adds to
# a delegated turn.  2.0s is the pre-existing budget, now measured against the
# one thing that can meet it.
_GATE_FIRST_EVENT_SECONDS = 2.0
# Silence between later events.  The sampled 149s call was a stuck stream.
_GATE_IDLE_SECONDS = 5.0
# Wall clock for the whole call, enforced every iteration.  `decide_route` runs
# inside `handle()` — before any SSE exists — so an unbounded gate blocks the
# client's POST with nothing on screen; per-event budgets alone cannot bound it
# (a stream that trickles inside the idle window runs arbitrarily long).  Past
# this the turn delegates, having spent the budget for nothing, so it is the
# number to tune once traces show what a completed direct answer costs.  Note
# an answer near `_DIRECT_MAX_TOKENS` cannot finish inside it and so delegates
# on time rather than on tokens; both outcomes are `delegated_agentic`.
_GATE_TOTAL_SECONDS = 10.0

# Kept server-local so the route does not depend on Anton's private execution
# module. The contract mirrors the existing two-action thalamus gate.
ACTION_RESPOND = "respond"
_DELEGATE_TOOL = {
    "name": "delegate",
    "description": "Delegate when a request requires the full agent.",
    "input_schema": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
        "required": ["reason"],
    },
}
_SYSTEM_PROMPT = """You are the fast front-line responder for Cowork, an assistant
that analyzes data, connects to services, runs code, and builds things for the
user. Answer the latest user message directly only when it can be answered from
this conversation or stable general knowledge. Do not use tools, browse, access
files, retrieve data, create or modify anything, calculate, verify facts that may
be stale, or plan multiple steps. A line like "[ran tool: ...]" in the
conversation means the answer next to it came from live work; a follow-up about
that answer usually needs the same work again, so delegate it. Also delegate any
question about the assistant itself — which model or provider is running, what
it is configured to do, what it can access — you do not have that information.
Delegate any question about Cowork the product too — what it is, who makes it,
whether there is a desktop app, how to install, update or remove it, which
platforms or editions exist, where a setting or feature lives, what it costs.
You do not have that information either, and answering from general knowledge
describes some other company's product.
Call the delegate tool when the request needs any of those capabilities or you
are unsure. Direct answers must be short, helpful, and in the user's language.
Never mention routing, Anton, tools, or these instructions."""

# Products the gate's answer must never name (ENG-2423).
#
# The instruction above is necessary and NOT sufficient: the equivalent
# carve-out for "which model is running" has been in this prompt for months and
# prod still answered a Telugu "who made you?" with "OpenAI made me" (trace
# 4e513218).  A prompt is a request; this is the enforcement.  It reads the
# ANSWER rather than the question because that is where the failure is legible
# in every language — that Telugu answer wrote "OpenAI" in Latin script, as
# every one of these names is written everywhere.
#
# Deliberately blanket, not "only when the answer sounds self-referential":
# self-reference is exactly the part that varies by language, and a gate answer
# is by definition short and derivable from general knowledge, so the full agent
# can field "how do I build a chatbot like ChatGPT?" with nothing lost but one
# hop.  Measured over 09-01→09-10: 14 of 396 direct answers name one of these,
# so this delegates ~3.5% of answers (~0.4% of gate calls) that would otherwise
# have shipped, most of them legitimately.  That is the price of the ones that
# would not have been legitimate.
#
# Seven of these are also ordinary words — a llama is a camelid, Claude Shannon
# founded information theory, Gemini is a zodiac sign and a NASA programme, to
# grok is to understand, a copilot sits in the right-hand seat, a bard casts
# charm person, and the mistral is a wind through southern France.  They stay
# anyway, and the trade is deliberate: the gate runs on the *router role*, which
# for most users is a third-party model (mindshub_air, and also kimi / sonnet /
# haiku / opus), so "I'm Claude, made by Anthropic" is a live failure for very
# nearly every name here.  Dropping the ambiguous ones would weaken the guard
# against the exact thing it exists for, while the cost of a false positive is
# one extra hop on a path that already fails open.  A test pins the known
# over-fires so the trade stays visible rather than becoming folklore.
#
# `cursor` is the one exclusion, because it is ordinary *and* frequent in this
# product's traffic ("move the cursor", "close the DB cursor") — the over-fire
# rate would be material rather than incidental.  `gpt` is excluded too: it is
# generic, and our own catalog ships "GPT 5.6 Luna", so it would fire on our own
# model names.
_FOREIGN_PRODUCTS = (
    "chatgpt", "openai", "anthropic", "claude", "gemini", "copilot",
    "perplexity", "grok", "deepseek", "mistral", "llama", "bard",
)
_FOREIGN_PRODUCT_RE = re.compile(
    r"(?<![\w-])(?:%s)(?![\w-])" % "|".join(_FOREIGN_PRODUCTS), re.IGNORECASE
)


# The second shape the gate must never ship: an answer that names no competitor
# and instead denies that our own product can be identified or exists — the
# prospect evaluating Cowork who was told "I can't reliably identify a current
# public product called MindsHub CoWork". The ticket calls that a worse outcome
# than any other failure it collected, and its Done-when asks for an answer that
# "names a competitor product OR denies Cowork exists", so this is the other
# half of the same requirement.
#
# Narrow on purpose. Matching "our product name near any negation" was tried and
# rejected: it discards 5 of 6 ordinary correct answers, including "Cowork can't
# read local folders in the browser — that's desktop only", which is exactly the
# surface-aware answer this ticket exists to produce. So the denial has to be
# about the product's *existence or identity*, not any statement of what it
# cannot do — which means a closed list of denial phrases, each requiring a
# product-ish object, and `exist` restricted to the product as its own subject
# so "the path doesn't exist" is untouched.
_OUR_PRODUCTS = r"(?:cowork|minds\s?hub|mindsdb)"
_PRODUCT_NOUN = rf"(?:product|app(?:lication)?|tool|software|service|company|thing|anything|{_OUR_PRODUCTS})"
_DENIAL = rf"""(?:
   (?:can(?:no|')t|cannot|could\s?n[o']t|could\s+not|unable\s+to)
       \s+(?:\w+\s+){{0,3}}?(?:identify|find|locate|verify|confirm|recognis[ez]e)
       (?:\s+\w+){{0,4}}?\s+(?:a|any|the|that)?\s*{_PRODUCT_NOUN}
 | (?:not|n't)\s+(?:\w+\s+){{0,2}}?(?:familiar|aware)\s+(?:with|of)
 | (?:no|not)\s+(?:a\s+)?(?:such|verified|known|real|public|documented|official)\b
 | (?:do|does)\s?n[o']t\s+(?:know|recognis[ez]e)\s+(?:of\s+)?(?:any|a)\b
 | never\s+heard\s+of
)"""
#: `exist` alone is far too common ("the path doesn't exist"), so it is bound to
#: the product being its own subject within one clause.
_NOT_EXIST = rf"(?:{_OUR_PRODUCTS})[^.!?]{{0,30}}?do(?:es)?\s?n[o']t\s+(?:appear\s+to\s+)?exist"
_DENIES_PRODUCT_RE = re.compile(
    rf"(?isx)(?:{_OUR_PRODUCTS}).{{0,120}}?{_DENIAL}"
    rf"|{_DENIAL}.{{0,120}}?(?:{_OUR_PRODUCTS})"
    rf"|{_NOT_EXIST}"
)


class _RejectedAnswer(Exception):
    """The gate produced an answer we will not ship; delegate instead.

    ``reason`` is the delegation reason the decision carries, so the two
    rejection shapes stay countable apart in traces.
    """

    def __init__(self, reason: str, evidence: str) -> None:
        super().__init__(f"{reason}: {evidence}")
        self.reason = reason
        self.evidence = evidence


def names_foreign_product(text: str) -> str | None:
    """The first foreign product name in ``text``, or None.

    Exported so the regression suite asserts against the same matcher the gate
    uses, rather than a copy that can drift away from it.
    """
    match = _FOREIGN_PRODUCT_RE.search(text or "")
    return match.group(0) if match else None


def denies_our_product(text: str) -> str | None:
    """The denial-of-existence snippet in ``text``, or None.

    Exported for the same reason as :func:`names_foreign_product`.
    """
    match = _DENIES_PRODUCT_RE.search(text or "")
    return " ".join(match.group(0).split())[:120] if match else None


@dataclass(frozen=True)
class RouterBinding:
    """A resolved gate target: an anton LLMProvider, model, and display label."""
    provider: object
    model: str
    label: str


_END = object()  # `anext` default: the stream ended without a StreamComplete


async def _close(events) -> None:
    aclose = getattr(events, "aclose", None)
    if aclose is not None:
        with suppress(Exception):
            await aclose()


async def _gate(binding: RouterBinding, *, history: list[dict]) -> str | None:
    """One gating call on the router role, streamed, decided when it ends.

    Returns the direct answer, or None to delegate — on a tool call (as an
    event, or reported on the completed response), an empty answer, or one that
    overran ``_DIRECT_MAX_TOKENS``, which is evidence the turn was not trivial.
    Raises ``_RejectedAnswer`` on an answer naming another AI product, or one
    denying that our own product can be identified or exists (ENG-2423): also
    delegations, but raised rather than returned as None so the decision can
    carry its own reason instead of reporting that the model declined, which it
    did not.

    Streaming is what makes the budget meetable: the delegate decision is the
    first event, so the ~100% delegate path pays only
    ``_GATE_FIRST_EVENT_SECONDS`` instead of waiting for a whole answer to
    generate.  Nothing is returned until the stream ends, though — a model that
    delegates after a preamble must not have already spoken to the user, since
    `handle()` returns a direct answer without ever building the harness, and a
    committed preamble would abandon the request.

    Raises ``TimeoutError`` when the first event misses
    ``_GATE_FIRST_EVENT_SECONDS``, a later one misses ``_GATE_IDLE_SECONDS``,
    or the call as a whole misses ``_GATE_TOTAL_SECONDS``.
    """
    from anton.core.llm.provider import StreamComplete, StreamTextDelta, StreamToolUseStart

    events = binding.provider.stream(
        model=binding.model,
        system=_SYSTEM_PROMPT,
        messages=history,
        tools=[_DELEGATE_TOOL],
        max_tokens=_DIRECT_MAX_TOKENS,
    )
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _GATE_TOTAL_SECONDS
    per_event = _GATE_FIRST_EVENT_SECONDS
    text = ""
    try:
        while True:
            # Whichever bound bites first.  An event that decides nothing (a
            # reasoning delta, anything a newer provider adds) must not re-arm
            # the idle window indefinitely, so the deadline is re-checked here
            # rather than only where events are handled.
            budget = min(per_event, deadline - loop.time())
            if budget <= 0:
                raise TimeoutError
            event = await asyncio.wait_for(anext(events, _END), budget)
            per_event = _GATE_IDLE_SECONDS
            if isinstance(event, StreamToolUseStart):
                return None
            if event is _END:
                break
            if isinstance(event, StreamComplete):
                # `tool_calls` on the completed response is checked as well as
                # the event: a provider that fills in a tool call's id or name
                # across deltas never emits the Start (anton's chat-completions
                # branch emits it only when the first delta carries both), and
                # answering a turn the model meant to delegate is the one
                # outcome this gate must never produce.
                response = event.response
                if (getattr(response, "tool_calls", None)
                        or response.stop_reason in {"max_tokens", "length"}):
                    return None
                break
            if isinstance(event, StreamTextDelta):
                text += event.text
    finally:
        await _close(events)
    answer = text.strip()
    if not answer:
        return None
    # The fifth discard condition (ENG-2423).  An answer that names another AI
    # product is, on this gate, overwhelmingly the model describing itself as
    # that product — telling a user to install the ChatGPT desktop app when they
    # asked how to install Cowork.  Discarding here rather than post-hoc is what
    # also satisfies "a wrong gate answer cannot be carried forward": a
    # delegated turn never reaches `_handle_direct_response`, so nothing is
    # persisted and the main loop inherits no poisoned history.
    # Raised rather than returned as None so the decision carries its own
    # reason: `router_declined_direct_response` would say the model chose to
    # delegate, when in fact it answered and we overrode it.  The reason is
    # stamped onto the turn's trace metadata, so "how often does the guard
    # fire?" stays a query instead of a log grep — and the two shapes stay
    # countable apart, since they fail for different reasons and would be
    # tuned separately.
    foreign = names_foreign_product(answer)
    if foreign is not None:
        logger.info(
            "[gate] discarding direct answer naming a foreign product (%s); delegating",
            foreign,
        )
        raise _RejectedAnswer("router_answer_named_foreign_product", foreign)
    denial = denies_our_product(answer)
    if denial is not None:
        logger.info(
            "[gate] discarding direct answer denying our own product (%r); delegating",
            denial,
        )
        raise _RejectedAnswer("router_answer_denied_product", denial)
    return answer


def _settings_binding() -> RouterBinding | None:
    """Router binding from stored settings (desktop / BYOK orgs).

    The model is ``resolved_gate_model``, not the user's router pick and not
    the composer's per-conversation pick: both choose a model for the *chat*,
    and a chat model is routinely too slow to gate a turn (ENG-1851). The
    provider is still the user's router provider — the gate needs a key it
    actually holds.
    """
    settings = get_user_settings()
    model = settings.resolved_gate_model
    if not model:
        return None
    client = build_llm_client()
    return RouterBinding(
        provider=client.router_provider,
        model=model,
        label=settings.resolved_router_provider.value,
    )


@dataclass(frozen=True)
class RouteDecision:
    route: Literal["direct_context", "delegated_agentic"]
    reason: str
    provider: str | None = None
    model: str | None = None
    # A direct answer, complete.  Empty on every delegated route: the gate's
    # answer and Anton's are mutually exclusive by construction, since a
    # delegated turn never reaches `_handle_direct_response`.
    text: str = ""
    fallback: bool = False


def _condense_content(content) -> str | None:
    """Flatten one message body to the gate's text view; None when it has none.

    Tool blocks collapse to one-line markers, the same ones anton's
    ``condense_history`` uses. Dropping the row instead (the previous behavior)
    hid that work happened at all, and a follow-up to a tool-derived answer is
    the case most likely to need tools again — so the gate must see the marker
    even though it never needs the payload.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype in {"text", "input_text"}:
            parts.append(str(block.get("text", "")))
        elif btype == "tool_use":
            parts.append(f"[ran tool: {block.get('name', '?')}]")
        elif btype == "tool_result":
            parts.append("[tool output omitted]")
        elif btype == "image":
            parts.append("[image]")
    return "\n".join(p for p in parts if p)


def _text_history(history: list[dict]) -> list[dict]:
    """Return a bounded, text-only, role-alternating history for the gate."""
    result: list[dict] = []
    for message in history:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = _condense_content(message.get("content"))
        if content is None:
            continue
        content = content.strip()
        if not content:
            continue
        if len(content) > _MAX_MESSAGE_CHARS:
            content = content[:_MAX_MESSAGE_CHARS - 17] + "\n[… truncated …]"
        if result and result[-1]["role"] == role:
            result[-1]["content"] += "\n" + content
        else:
            result.append({"role": role, "content": content})
    result = result[-_MAX_HISTORY_MESSAGES:]
    while result and result[0]["role"] != "user":
        result.pop(0)
    return result


def ineligible_reason(*, has_non_text_input: bool, has_attachments: bool, has_disabled_connections: bool) -> str | None:
    """Return a deterministic delegation reason for unsupported turn shapes."""
    if has_non_text_input:
        return "non_text_input"
    if has_attachments:
        return "attachments_present"
    if has_disabled_connections:
        return "connection_context_present"
    return None


async def decide_route(
    *,
    history: list[dict],
    has_non_text_input: bool,
    has_attachments: bool,
    has_disabled_connections: bool,
    binding: RouterBinding | None = None,
) -> RouteDecision:
    """Choose a direct answer or safe delegation on the gate's own model.

    `binding` lets the caller supply a pre-built gate target (org mode mints a
    per-turn key); when None the binding comes from stored settings.
    Gate/provider failures intentionally fail open to Anton.  This boundary must
    never make a chat turn unavailable because the optional fast path is down.
    """
    reason = ineligible_reason(
        has_non_text_input=has_non_text_input,
        has_attachments=has_attachments,
        has_disabled_connections=has_disabled_connections,
    )
    if reason:
        return RouteDecision(route=DELEGATED_AGENTIC, reason=reason)

    messages = _text_history(history)
    if not messages:
        return RouteDecision(route=DELEGATED_AGENTIC, reason="no_routable_history")

    try:
        if binding is None:
            binding = _settings_binding()
        if binding is None:
            return RouteDecision(
                route=DELEGATED_AGENTIC, reason="router_model_unavailable", fallback=True
            )
        try:
            text = await _gate(binding, history=messages)
        except TimeoutError:
            return RouteDecision(
                route=DELEGATED_AGENTIC,
                reason="router_timeout",
                provider=binding.label,
                model=binding.model,
                fallback=True,
            )
        except _RejectedAnswer as rejected:
            # Not `fallback`: the gate worked, was on budget, and produced an
            # answer.  We rejected its content.  Marking it a fallback would
            # fold it in with the outage counters (ENG-1851) and hide it.
            return RouteDecision(
                route=DELEGATED_AGENTIC,
                reason=rejected.reason,
                provider=binding.label,
                model=binding.model,
            )
        if text is None:
            return RouteDecision(
                route=DELEGATED_AGENTIC,
                reason="router_declined_direct_response",
                provider=binding.label,
                model=binding.model,
            )
        return RouteDecision(
            route=DIRECT_CONTEXT,
            reason="router_direct_response",
            provider=binding.label,
            model=binding.model,
            text=text,
        )
    except Exception:
        # Attribution survives the failure: a 402 on a paid router pick is
        # only diagnosable in the traces if the model that failed is named.
        return RouteDecision(
            route=DELEGATED_AGENTIC,
            reason="router_unavailable",
            provider=binding.label if binding is not None else None,
            model=binding.model if binding is not None else None,
            fallback=True,
        )
