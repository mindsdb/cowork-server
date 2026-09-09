"""Rebuilding the assistant's answer text from a turn's SSE events.

Every consumer that needs the finished answer — the two streaming handlers, the
non-streaming one, the channels runtime — accumulates it from the same events
and then persists the result. Sharing the rule is what keeps them from
drifting: a consumer that honours the delta but not the reset persists an
answer the user was never shown whole.
"""

from __future__ import annotations


def accumulate_answer_text(collected: list[str], event_type: str, data: dict) -> None:
    """Apply one formatter event to an answer-text accumulator, in place.

    `response.output_text.reset` drops everything before it: anton's completion
    verifier forced a continuation, so the text that follows replaces the answer
    already streamed rather than continuing it. The formatter emits it only
    immediately before that replacement text, so this never empties an
    accumulator it does not go on to refill.
    """
    if event_type == "response.output_text.delta":
        collected.append(data.get("delta", ""))
    elif event_type == "response.output_text.reset":
        collected.clear()
