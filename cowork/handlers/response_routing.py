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
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal

from cowork.common.settings.user_settings import get_user_settings
from cowork.services.providers import build_llm_client


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
Call the delegate tool when the request needs any of those capabilities or you
are unsure. Direct answers must be short, helpful, and in the user's language.
Never mention routing, Anton, tools, or these instructions."""


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
    return text.strip() or None


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
