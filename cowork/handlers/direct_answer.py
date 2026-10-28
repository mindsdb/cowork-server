"""The direct answer on MindsHub: one streamed call to the user's own model.

Jev (`jev_route`) decides a turn can be answered without the agent; this writes
that answer on the model the user picked, not on a fast router model, so the
reply is as good as the agent's own would have been for a question that needs
no tools. It sees the conversation's visible text (wider than the gate's view),
the product facts, the gateway's web search, and the `delegate` tool, which
hands the turn to the agent when Jev was wrong.

Nothing is shown or saved until the first decisive event: text means the
model is answering, a tool call means it delegated. Before that, any failure,
timeout or empty reply delegates too, so a broken fast path only costs time.
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
    _DELEGATE_TOOL,
    _END,
    RouterBinding,
    _close,
    _text_history,
)

logger = logging.getLogger(__name__)

# Wider than the gate's view (16 messages, 1,500 chars each): recalling what an
# earlier reply said is the job this path is routed for, and a long reply cut
# at 1,500 chars can lose exactly the numbers the user asks about again.
MAX_HISTORY_MESSAGES = 40
_MAX_MESSAGE_CHARS = 12_000
_MAX_TOKENS = 4_096
# Until the first decisive event. Generous because a web search runs on the
# gateway before any event arrives: about 8s measured for a weather question.
_FIRST_EVENT_SECONDS = 30.0
# Silence between later text events.
_IDLE_SECONDS = 30.0

_SYSTEM_PROMPT = """You are the fast responder for Cowork, an assistant that analyzes
data, connects to services, runs code, and builds things for the user. Reply to
the latest user message using this conversation, general knowledge, the product
facts below, and a web search when the reply needs current public facts.

You cannot see the user's data, files, connections or account. A line
"[ran tool: ...]" means the full agent did work earlier, and "[tool output omitted]"
means that work's output is hidden from you: use only what the assistant's
visible replies say. Call the delegate tool, before writing anything, when the
reply needs any of the user's data, files, connections or account, a value that
is not stated in the visible conversation, new or changed analysis, charts,
dashboards, files or code, or a multi-step plan. Never guess a number.

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


def system_prompt(*, model: str) -> str:
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
        _SYSTEM_PROMPT,
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
    """An answer that has started: its first text, and the stream to finish."""

    first_text: str
    _events: AsyncIterator

    async def rest(self) -> AsyncIterator[str]:
        """The answer's remaining text. A tool call after text has started is
        ignored: the answer is already on screen, so it ends where it is."""
        from anton.core.llm.provider import StreamComplete, StreamTextDelta

        try:
            while True:
                event = await asyncio.wait_for(anext(self._events, _END), _IDLE_SECONDS)
                if event is _END or isinstance(event, StreamComplete):
                    return
                if isinstance(event, StreamTextDelta) and event.text:
                    yield event.text
        finally:
            await _close(self._events)

    async def close(self) -> None:
        await _close(self._events)


async def start_answer(binding: RouterBinding, *, history: list[dict]) -> DirectAnswer | None:
    """Start the direct answer: a `DirectAnswer` once it begins with text, or
    None to delegate. Never raises (a database pool timeout aside, which
    nothing here can raise)."""
    from anton.core.llm.provider import StreamComplete, StreamTextDelta, StreamToolUseStart

    messages = _text_history(history, max_messages=MAX_HISTORY_MESSAGES, max_chars=_MAX_MESSAGE_CHARS)
    if not messages:
        return None
    provider = binding.provider
    native_web_tools = getattr(provider, "native_web_tools", None)
    try:
        events = provider.stream(
            model=binding.model,
            system=system_prompt(model=binding.model),
            messages=messages,
            tools=[_DELEGATE_TOOL],
            max_tokens=_MAX_TOKENS,
            native_web_tools=native_web_tools() if callable(native_web_tools) else None,
        )
    except Exception:
        logger.warning("[direct] could not start the answer; delegating", exc_info=True)
        return None
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _FIRST_EVENT_SECONDS
    try:
        while True:
            budget = deadline - loop.time()
            if budget <= 0:
                raise TimeoutError
            event = await asyncio.wait_for(anext(events, _END), budget)
            if event is _END:
                break
            if isinstance(event, StreamToolUseStart):
                break
            if isinstance(event, StreamComplete):
                # A provider that fills in a tool call across deltas may never
                # emit the Start; the completed response still names it.
                break
            if isinstance(event, StreamTextDelta) and event.text.strip():
                return DirectAnswer(first_text=event.text, _events=events)
    except TimeoutError:
        logger.warning(
            "Direct answer did not start in time; delegating",
            extra={"event": "direct_answer_timeout", "timeout_seconds": _FIRST_EVENT_SECONDS},
        )
    except Exception:
        logger.warning("[direct] answer failed before it started; delegating", exc_info=True)
    await _close(events)
    return None


async def complete_answer(binding: RouterBinding, *, history: list[dict]) -> str | None:
    """The whole direct answer, for a turn that does not stream, or None to
    delegate. Nothing is saved before it returns, so a failure part way still
    delegates."""
    answer = await start_answer(binding, history=history)
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
