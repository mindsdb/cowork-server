"""Validate the turn-history rows a semi-trusted pod sent us (ENG-1808).

The pod runs tenant code, and these rows land in two places that make sloppy
input expensive: the `messages` table, and the LLM history replayed on every
later turn of the conversation.

Rejection is always all-or-nothing. tool_use/tool_result pairing is a property
of the SET, not of a single row, so dropping one bad row would leave an orphan
block and make every subsequent turn fail with a 400 from the provider. Dropping
the whole set costs that one turn its tool detail and nothing else: it replays
text-only, exactly the degradation the in-process harness already applies after
a mid-turn compaction.

The one exception is an oversized tool_result, where the payload is replaced by
a placeholder instead of the row being dropped — the call stays visible and
pairing stays intact.

There is NO upstream clip on this path: anton's MAX_STEP_CHARS bounds the
`tool_result` STEP event rendered in the UI, not `session.history`, which is
where these rows come from. The caps below are the only limit, so expect the
placeholder to fire on ordinary large cell output.

Changing the caps affects NEW turns only — rows already written to the messages
table are never recomputed.
"""

from __future__ import annotations

import json
import logging
from collections import Counter

logger = logging.getLogger(__name__)

#: Per-tool_result payload budget. Sized for "the gist of the call" rather than
#: a whole cell dump; oversize is replaced by a placeholder, not dropped.
_MAX_RESULT_BYTES = 16 * 1024
#: Whole-turn budget. ~40 heavy turns fit inside the pod request's 10 MiB cap.
_MAX_TURN_BYTES = 256 * 1024

#: The splitter only ever produces these two pairings, so anything else is a
#: sign the payload did not come from it.
_BLOCK_FOR_ROLE = {"assistant": "tool_use", "user": "tool_result"}
_ID_FIELD = {"tool_use": "id", "tool_result": "tool_use_id"}


def _wire_size(obj) -> int:
    """Bytes this object will occupy on the wire.

    Same measure as the producer's _request_wire_size, so these budgets are
    directly comparable to the pod request's 10 MiB cap.
    """
    return len(json.dumps(obj, separators=(",", ":")).encode("utf-8"))


def _reject(reason: str, *args) -> list:
    logger.warning("[turn_history] dropping this turn's rows: " + reason, *args)
    return []


def _cap_tool_result(block: dict) -> dict:
    """Replace an oversized tool_result payload with a placeholder.

    Returns a new dict; the caller's block is never mutated. The placeholder is
    a plain string, which is valid whether the original content was a string or
    a list of blocks.
    """
    size = _wire_size(block.get("content"))
    if size <= _MAX_RESULT_BYTES:
        return block
    return {**block, "content": (
        f"[tool output omitted: {size} bytes over the "
        f"{_MAX_RESULT_BYTES}-byte replay cap]"
    )}


def sanitize_turn_history_rows(rows: object) -> list[dict]:
    """Return usable `{role, content}` rows, or `[]` to fall back to text-only.

    `[]` is a valid, safe answer for every failure mode — callers should pass
    the result straight to `save_assistant_turn(tool_rows=...)`.
    """
    if not isinstance(rows, list):
        return _reject("payload is %s, not a list", type(rows).__name__)

    clean: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            return _reject("row is %s, not a dict", type(row).__name__)
        role = row.get("role")
        want = _BLOCK_FOR_ROLE.get(role)
        if want is None:
            return _reject("unexpected role %r", role)
        content = row.get("content")
        if not isinstance(content, list) or not content:
            return _reject("role %s carries %s, not a non-empty list of blocks",
                           role, type(content).__name__)
        blocks = []
        for block in content:
            if not isinstance(block, dict):
                return _reject("role %s carries a %s block", role, type(block).__name__)
            if block.get("type") != want:
                return _reject("role %s carries a %r block, expected %r",
                               role, block.get("type"), want)
            if not block.get(_ID_FIELD[want]):
                return _reject("%s block without %s", want, _ID_FIELD[want])
            blocks.append(_cap_tool_result(block) if want == "tool_result" else block)
        clean.append({"role": role, "content": blocks})

    if not clean:
        return []

    # Counter, not set: two tool_use blocks sharing an id against one
    # tool_result would pass a set comparison and still replay invalidly.
    uses = Counter(
        b["id"] for r in clean if r["role"] == "assistant" for b in r["content"]
    )
    results = Counter(
        b["tool_use_id"] for r in clean if r["role"] == "user" for b in r["content"]
    )
    if uses != results:
        return _reject("tool_use ids %s do not pair with tool_result ids %s",
                       sorted(uses.elements()), sorted(results.elements()))

    total = _wire_size(clean)
    if total > _MAX_TURN_BYTES:
        return _reject("rows are %d bytes, over the %d-byte turn cap",
                       total, _MAX_TURN_BYTES)
    return clean
