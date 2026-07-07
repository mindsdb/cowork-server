"""Context-consumption telemetry (ENG-642).

Three structured, greppable log streams — one JSON payload per line:

    [llm_usage]      one line per LLM call (tokens, context pressure)
    [turn_summary]   one line per turn (calls, totals, TTFT, duration)
    [prompt_anatomy] one line per turn (char size of every prompt component)

The per-call usage numbers originate from anton's ``StreamComplete`` event,
which cowork previously discarded. ``scripts/context_report.py`` aggregates
all three streams into the baseline report.

Telemetry must never affect a turn: callers wrap these in try/except, and
nothing here makes an LLM call or touches the DB.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

USAGE_TAG = "[llm_usage]"
TURN_TAG = "[turn_summary]"
ANATOMY_TAG = "[prompt_anatomy]"

# Anton system-prompt constants worth sizing. These are the static blocks
# assembled into every call's system prompt (see the context-optimization
# investigation); measuring them by import keeps the anatomy line exact
# without rebuilding the prompt (which could trigger a memory-filter LLM call).
_ANTON_PROMPT_CONSTANTS = (
    "CHAT_SYSTEM_PROMPT",
    "BACKEND_GENERATION_PROMPT",
    "ARTIFACTS_PROMPT",
    "BASE_VISUALIZATIONS_PROMPT",
    "VISUALIZATIONS_HTML_OUTPUT_FORMAT_PROMPT",
    "VISUALIZATIONS_MARKDOWN_OUTPUT_FORMAT_PROMPT",
    "CONVERSATION_DISCIPLINE_ACT_FIRST",
    "CONVERSATION_DISCIPLINE_ASK_FIRST",
)


@lru_cache(maxsize=1)
def anton_static_section_chars() -> dict[str, int]:
    """Char size of anton's static system-prompt blocks, measured once.

    Degrades to {} if the installed anton renames/moves the constants —
    the anatomy line then simply lacks the static breakdown.
    """
    try:
        from anton.core.llm import prompts
    except Exception:
        return {}
    sizes: dict[str, int] = {}
    for name in _ANTON_PROMPT_CONSTANTS:
        value = getattr(prompts, name, None)
        if isinstance(value, str):
            sizes[name] = len(value)
    return sizes


def log_llm_usage(payload: dict) -> None:
    logger.info("%s %s", USAGE_TAG, json.dumps(payload, default=str))


def log_turn_summary(payload: dict) -> None:
    logger.info("%s %s", TURN_TAG, json.dumps(payload, default=str))


def build_prompt_anatomy(
    *,
    conversation_id,
    turn_id: int,
    initial_history: list[dict],
    suffix_parts: dict[str, str | None],
    tool_defs: list[dict],
) -> dict:
    """Size every prompt component cowork controls or can measure statically.

    The true per-call total comes from [llm_usage] input_tokens; this line
    explains its composition (chars ≈ tokens × 4).
    """
    tool_chars = {
        str(t.get("name") or f"tool_{i}"): len(json.dumps(t, default=str))
        for i, t in enumerate(tool_defs)
    }
    payload = {
        "conversation_id": str(conversation_id),
        "turn_id": turn_id,
        "history_messages": len(initial_history),
        "history_chars": sum(len(str(m.get("content") or "")) for m in initial_history),
        "suffix_chars": {k: len(v or "") for k, v in suffix_parts.items()},
        "tool_chars": tool_chars,
        "tools_total_chars": sum(tool_chars.values()),
        "anton_static_chars": anton_static_section_chars(),
    }
    return payload


def log_prompt_anatomy(payload: dict) -> None:
    logger.info("%s %s", ANATOMY_TAG, json.dumps(payload, default=str))
