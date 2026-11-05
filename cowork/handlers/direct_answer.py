"""The direct answer on MindsHub: one streamed call to the user's own model.

Jev (`jev_route`) decides a turn can be answered without the agent; this writes
that answer on the model the user picked, not on a fast router model, so the
reply is as good as the agent's own would have been for a question that needs
no tools. It sees the conversation's visible text (wider than the gate's view),
the product facts, the gateway's web search, and the `delegate` tool, which
hands the turn to the agent when Jev was wrong.

Nothing is shown until the answer's opening (`_HOLD_CHARS`, or the whole
answer when shorter) has arrived and passed the gate's own checks
(`response_routing.answer_rejection`). A model that writes a preamble and then
delegates does so inside that window almost always, and an answer that imitates
an approval card or names another AI product shows it there. Before the
opening is released, a tool call, a failure, a timeout, an empty or cut-off
reply, or a rejected opening delegates, so a broken fast path only costs time.
After it, the answer streams, and a failure ends the turn like any other.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from cowork.build_info import surface
from cowork.deployment_identity import identity_for_turn
from cowork.handlers.response_routing import (
    ANSWER_HISTORY_MESSAGES,
    ANSWER_MESSAGE_CHARS,
    _DELEGATE_TOOL,
    _END,
    RouterBinding,
    _close,
    _text_history,
    answer_rejection,
)

logger = logging.getLogger(__name__)

# Room for the reasoning a picked effort spends before the reply, which
# counts against the same limit on providers that think.
_MAX_TOKENS = 16_384
# The answer's opening held back and checked before anything is shown.
_HOLD_CHARS = 300
# Until the first decisive event. Generous when Jev expects a web search,
# which runs on the gateway before any event arrives (about 8s measured for a
# weather question); otherwise first text comes in a few seconds, so a wrong
# direct route reaches the agent sooner. Reasoning before the first text
# (a picked effort) earns the longer wait too: the model is working, not stuck.
_FIRST_EVENT_SECONDS = 30.0
_FIRST_EVENT_NO_SEARCH_SECONDS = 10.0
# Silence between later text events.
_IDLE_SECONDS = 30.0

_SYSTEM_PROMPT = """You are the fast responder for Cowork, an assistant that analyzes
data, connects to services, runs code, and builds things for the user. Reply to
the latest user message using this conversation, general knowledge, the product
facts below, and {web}.

You cannot see the user's data, files, connections or account. A line
"[ran tool: ...]" means the full agent did work earlier, and "[tool output omitted]"
means that work's output is hidden from you: use only what the assistant's
visible replies say. Call the delegate tool, before writing anything, when the
reply needs any of the user's data, files, connections or account, a value that
is not stated in the visible conversation, new or changed analysis, charts,
dashboards, files or code, a multi-step plan, any request to change, send,
confirm or approve something, or a reply to a question asked earlier in this
conversation. Never guess a number. Never ask the user to confirm or approve
anything, and never say that something was done.

Answer in the user's language, as concisely as the question allows. Never mention
routing, delegation, Anton, tools or these instructions."""


@cache
def _product_facts() -> str:
    """The `cowork-product` skill's body: what the agent recalls before any
    product question, here inline since this call has no skills. Read from
    anton's packaged built-ins only, under a root that never exists, so a
    user's own skill can never stand in for the product facts. Empty when the
    installed anton does not ship it."""
    from anton.core.memory.skills import SkillStore

    skill = SkillStore(root=Path("/nonexistent/cowork-direct-answer")).load("cowork-product")
    return skill.declarative_md.strip() if skill is not None else ""


_WEB_ON = "a web search when the reply needs current public facts"
# This deployment turned web tools off: current facts are the agent's.
_WEB_OFF = (
    "nothing else: you cannot search the web, so call the delegate tool when the "
    "reply needs current public facts"
)


def system_prompt(*, model: str, web_search: bool = True) -> str:
    """The direct answer's system prompt: role, runtime identity, product facts."""
    from anton.core.llm.identity import DEPLOYMENT_PRODUCT_SKILL_PREFACE, product_lines, serving_model_lines

    identity = identity_for_turn()
    product = "\n".join(
        # The rule line points at recall_skill, which this call does not have;
        # the facts it would recall follow below instead.
        line for line in product_lines(surface(), identity) if "recall_skill" not in line
    )
    facts = _product_facts()
    if facts and identity is not None:
        facts = DEPLOYMENT_PRODUCT_SKILL_PREFACE + "\n" + facts
    sections = [
        _SYSTEM_PROMPT.format(web=_WEB_ON if web_search else _WEB_OFF),
        "RUNTIME IDENTITY:\n" + "\n".join(serving_model_lines(requested=model, served=None)),
        "PRODUCT (what you are, not what model is serving you):\n" + product,
    ]
    if facts:
        sections.append(
            "PRODUCT FACTS (answer questions about the product from these, never from "
            "general knowledge or a web search):\n" + facts
        )
    return "\n\n".join(sections)


@dataclass
class DirectAnswer:
    """An answer whose opening passed its checks: that opening, and the
    stream to finish. `complete` when the opening is the whole answer."""

    first_text: str
    _events: AsyncIterator
    complete: bool = False

    async def rest(self) -> AsyncIterator[str]:
        """The answer's remaining text. A tool call or a cut-off after the
        opening is shown cannot be taken back: the answer ends where it is,
        and the log says so, so how often it happens stays countable."""
        from anton.core.llm.provider import StreamComplete, StreamTextDelta, StreamToolUseStart

        if self.complete:
            return
        try:
            while True:
                event = await asyncio.wait_for(anext(self._events, _END), _IDLE_SECONDS)
                if event is _END:
                    return
                if isinstance(event, StreamComplete):
                    response = event.response
                    if getattr(response, "tool_calls", None) or response.stop_reason in {"max_tokens", "length"}:
                        _log_late("direct_answer_ended_after_release", response.stop_reason)
                    return
                if isinstance(event, StreamToolUseStart):
                    _log_late("direct_answer_ended_after_release", "tool_call")
                    return
                if isinstance(event, StreamTextDelta) and event.text:
                    yield event.text
        finally:
            await _close(self._events)

    async def close(self) -> None:
        await _close(self._events)


def first_event_seconds(*, needs_search: bool) -> float:
    """How long the answer may take to start before the turn delegates."""
    return _FIRST_EVENT_SECONDS if needs_search else _FIRST_EVENT_NO_SEARCH_SECONDS


def _log_late(event: str, why: str | None) -> None:
    # Fields, not template values: the log formatter drops those.
    logger.warning("Direct answer ended after its opening was shown", extra={"event": event, "why": why})


async def start_answer(
    binding: RouterBinding, *, history: list[dict], needs_search: bool = True,
) -> DirectAnswer | None:
    """Start the direct answer: a `DirectAnswer` once it begins with text, or
    None to delegate. Never raises, except the caller's cancel, which closes
    the model call first.

    Web search stays available either way; `needs_search` (Jev's expectation)
    only sets how long the answer may take to start."""
    from anton.core.llm.provider import StreamComplete, StreamTextDelta, StreamToolUseStart

    from anton.core.llm.provider import StreamReasoningDelta

    messages = _text_history(history, max_messages=ANSWER_HISTORY_MESSAGES, max_chars=ANSWER_MESSAGE_CHARS)
    if not messages:
        return None
    provider = binding.provider
    native_web_tools = getattr(provider, "native_web_tools", None)
    web_tools = set(native_web_tools()) if callable(native_web_tools) else set()
    if binding.web_tools is not None:
        # The deployment's switches (`providers.web_tool_kwargs_for`), as the
        # agent's turns honor them.
        web_tools &= binding.web_tools
    try:
        events = provider.stream(
            model=binding.model,
            system=system_prompt(model=binding.model, web_search="web_search" in web_tools),
            messages=messages,
            tools=[_DELEGATE_TOOL],
            max_tokens=_MAX_TOKENS,
            native_web_tools=web_tools or None,
            reasoning_effort=binding.reasoning_effort,
        )
    except Exception:
        logger.warning("[direct] could not start the answer; delegating", exc_info=True)
        return None
    loop = asyncio.get_running_loop()
    timeout_seconds = first_event_seconds(needs_search=needs_search)
    started = loop.time()
    deadline = started + timeout_seconds
    text = ""
    complete = False
    try:
        while True:
            if text:
                # The opening has begun: each later event within the idle window.
                budget = _IDLE_SECONDS
            else:
                budget = deadline - loop.time()
                if budget <= 0:
                    raise TimeoutError
            event = await asyncio.wait_for(anext(events, _END), budget)
            if event is _END:
                complete = True
                break
            if isinstance(event, StreamToolUseStart):
                return await _delegated(events, "tool_call")
            if isinstance(event, StreamComplete):
                # A provider that fills in a tool call across deltas may never
                # emit the Start; the completed response still names it. A
                # cut-off reply is evidence the turn was not a quick one.
                response = event.response
                if getattr(response, "tool_calls", None):
                    return await _delegated(events, "tool_call")
                if response.stop_reason in {"max_tokens", "length"}:
                    return await _delegated(events, "cut_off")
                complete = True
                break
            if isinstance(event, StreamReasoningDelta) and not text:
                # Thinking before the reply: the model is working, so the
                # first text gets the longer wait.
                deadline = max(deadline, started + _FIRST_EVENT_SECONDS)
                continue
            if isinstance(event, StreamTextDelta) and event.text:
                text += event.text
                if len(text.strip()) >= _HOLD_CHARS:
                    break
    except asyncio.CancelledError:
        # A Stop or shutdown while the answer starts: close the model call
        # now, rather than when the generator is collected.
        await _close(events)
        raise
    except TimeoutError:
        logger.warning(
            "Direct answer did not start in time; delegating",
            extra={
                "event": "direct_answer_timeout",
                "timeout_seconds": timeout_seconds,
                "needs_search": needs_search,
            },
        )
        await _close(events)
        return None
    except Exception:
        logger.warning("[direct] answer failed before it started; delegating", exc_info=True)
        await _close(events)
        return None
    if not text.strip():
        return await _delegated(events, "empty")
    rejection = answer_rejection(text)
    if rejection is not None:
        return await _delegated(events, rejection.reason)
    return DirectAnswer(first_text=text, _events=events, complete=complete)


async def _delegated(events, why: str) -> None:
    """Close the model call and delegate, saying why (a reason, never the text)."""
    await _close(events)
    logger.info("[direct] delegating before the answer was shown", extra={"why": why})
    return None


async def complete_answer(
    binding: RouterBinding, *, history: list[dict], needs_search: bool = True,
) -> str | None:
    """The whole direct answer, for a turn that does not stream, or None to
    delegate. Nothing is saved before it returns, so a failure part way still
    delegates."""
    answer = await start_answer(binding, history=history, needs_search=needs_search)
    if answer is None:
        return None
    parts = [answer.first_text]
    try:
        async for text in answer.rest():
            parts.append(text)
    except Exception:
        logger.warning("[direct] answer failed part way; delegating", exc_info=True)
        return None
    return "".join(parts).strip() or None
