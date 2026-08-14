"""Cowork's first response-routing decision.

This is deliberately a server-owned front gate.  Anton keeps its own gate for
turns delegated below, so the rollout is safe while the two implementations
coexist.  A direct decision is only valid for a text-only conversational turn;
every uncertain or unsupported shape delegates to Anton.
"""
from __future__ import annotations

import asyncio
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
# The gate sits ahead of every turn; a slow router must not delay Anton.
_GATE_TIMEOUT_SECONDS = 2.0

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
_SYSTEM_PROMPT = """You are Cowork's fast front-line responder. Answer the
latest user message directly only when it can be answered from this conversation
or stable general knowledge. Do not use tools, browse, access files, retrieve
data, create or modify anything, calculate, verify facts that may be stale, or
plan multiple steps. Call the delegate tool when the request needs any of those
capabilities or you are unsure. Direct answers must be short, helpful, and in
the user's language. Never mention routing, Anton, tools, or these instructions."""


async def _gate(llm, *, history: list[dict]):
    """Use the resolved router role without coupling Cowork to Anton's gate."""
    response = await llm.gate(
        system=_SYSTEM_PROMPT,
        messages=history,
        tools=[_DELEGATE_TOOL],
        max_tokens=_DIRECT_MAX_TOKENS,
    )
    if response.tool_calls or response.stop_reason in {"max_tokens", "length"}:
        return None
    text = (response.content or "").strip()
    return text or None


@dataclass(frozen=True)
class RouteDecision:
    route: Literal["direct_context", "delegated_agentic"]
    reason: str
    provider: str | None = None
    model: str | None = None
    text: str = ""
    fallback: bool = False


def _text_history(history: list[dict]) -> list[dict]:
    """Return a bounded, text-only, role-alternating history for the gate."""
    result: list[dict] = []
    for message in history:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
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
) -> RouteDecision:
    """Choose a direct answer or safe delegation using the configured router role.

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
        settings = get_user_settings()
        provider = settings.resolved_router_provider
        model = settings.resolved_router_model
        if not model:
            return RouteDecision(
                route=DELEGATED_AGENTIC, reason="router_model_unavailable", fallback=True
            )
        text = await asyncio.wait_for(
            _gate(build_llm_client(), history=messages), _GATE_TIMEOUT_SECONDS
        )
        if text is None:
            return RouteDecision(
                route=DELEGATED_AGENTIC,
                reason="router_declined_direct_response",
                provider=provider.value,
                model=model,
            )
        return RouteDecision(
            route=DIRECT_CONTEXT,
            reason="router_direct_response",
            provider=provider.value,
            model=model,
            text=text,
        )
    except TimeoutError:
        return RouteDecision(
            route=DELEGATED_AGENTIC, reason="router_timeout", fallback=True
        )
    except Exception:
        return RouteDecision(
            route=DELEGATED_AGENTIC, reason="router_unavailable", fallback=True
        )
