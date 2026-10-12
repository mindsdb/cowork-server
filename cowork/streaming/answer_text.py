"""Rebuilding the assistant's answer text from a turn's SSE events.

Every consumer that needs the finished answer — the two streaming handlers, the
non-streaming one, the channels runtime — accumulates it from the same events
and then persists the result. Sharing the rule is what keeps them from
drifting: a consumer that honours the delta but not the reset persists an
answer the user was never shown whole.

The same events are stored for a reload with their adjacent text deltas
merged (coalesce_text_deltas), which leaves the text this rule rebuilds as it
was.
"""

from __future__ import annotations

from typing import Any

from cowork.schemas.responses import StreamingResponseEvent

_TEXT_DELTA = StreamingResponseEvent.output_text_delta.value

# Every key a plain text delta carries: the formatter's delta event
# (anton_harness/stream_formatter.py) with the at_ms stamp its _event adds, and
# the connector probe's (handlers/probe.py). A delta with any other key, such
# as a direct answer's response_route, is stored as it is.
_PLAIN_DELTA_KEYS = frozenset({"type", "sequence_number", "item_id", "delta", "at_ms"})


def accumulate_answer_text(collected: list[str], event_type: str, data: dict) -> None:
    """Apply one formatter event to an answer-text accumulator, in place.

    `response.answer_reset` drops everything before it: anton's completion
    verifier forced a continuation, so the text that follows replaces the answer
    already streamed rather than continuing it. The formatter emits it only
    immediately before that replacement text, so this never empties an
    accumulator it does not go on to refill.

    `response.answer_restore` undoes one reset, carrying the text back with it.
    A continuation normally narrates before its first tool call, so the delta
    that spent the boundary is not always the promised answer; when the turn
    hands back instead, the answer that was already read has to return.
    """
    if event_type == _TEXT_DELTA:
        collected.append(data.get("delta", ""))
    elif event_type == "response.answer_reset":
        collected.clear()
    elif event_type == "response.answer_restore":
        collected.insert(0, data.get("text", ""))


def _is_plain_delta(event: Any) -> bool:
    return (
        isinstance(event, dict)
        and event.get("type") == _TEXT_DELTA
        and isinstance(event.get("delta"), str)
        and event.keys() <= _PLAIN_DELTA_KEYS
    )


def coalesce_text_deltas(events: list[dict]) -> list[dict]:
    """A turn's events as they are stored: each run of adjacent plain text
    deltas of one item becomes one delta carrying the run's text.

    The live stream stays one frame per delta; only the stored log is merged.
    Every reader of the stored log (a reload's replay, the answer text, the
    scratchpad replay) concatenates adjacent deltas anyway, so the merge saves
    rows without changing what any of them rebuilds. A merged delta keeps the
    first delta's sequence_number, item_id and at_ms. Any other event ends a
    run and passes through as the same object in the same place.

    Never mutates its input: those dicts are the ones the live SSE frames were
    serialized from.
    """
    out: list[dict] = []
    run: list[dict] = []

    def flush() -> None:
        if len(run) == 1:
            out.append(run[0])
        elif run:
            out.append({**run[0], "delta": "".join(event["delta"] for event in run)})
        run.clear()

    for event in events:
        if not _is_plain_delta(event):
            flush()
            out.append(event)
            continue
        if run and run[0].get("item_id") != event.get("item_id"):
            flush()
        run.append(event)
    flush()
    return out
