"""Rebuild Anton `Cell` lists from persisted assistant streaming events."""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from anton.core.backends.base import Cell

_SCRATCHPAD_END = "thought.scratchpad.end"
_SCRATCHPAD_RESULT = "thought.scratchpad.result"

# The only thought roles extract_scratchpad_cells reacts to. The harness reads
# just the events carrying one of them, so a role the extractor starts to
# read has to be added here too, or its events never reach it.
SCRATCHPAD_REPLAY_ROLES = (_SCRATCHPAD_END, _SCRATCHPAD_RESULT)

# Cell's outputs that have no default, for a result cut before them.
_NO_OUTPUT = {"stdout": "", "stderr": "", "error": None}

# JSON's whitespace between tokens.
_SPACE = re.compile(r"[ \t\n\r]*")

# The start of a cell's result JSON, which json.dumps(asdict(cell)) opens
# with `code`. A result that starts this way and doesn't parse was cut.
_CELL_JSON_START = re.compile(r'\{[ \t\n\r]*"code"[ \t\n\r]*:')


@dataclass(frozen=True)
class _SentCall:
    """A scratchpad call as its end event stored it: the action the model
    sent, and its code when the stored event kept that whole."""

    action: str
    code: str | None


def _parse_json(raw: Any) -> dict[str, Any] | None:
    """The JSON object in ``raw``, or the fields it kept whole if it was cut."""
    if not isinstance(raw, str) or not (t := raw.strip()):
        return None
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        return _whole_fields(text=t) or None
    return data if isinstance(data, dict) else None


def _whole_fields(*, text: str) -> dict[str, Any]:
    """The fields of a JSON object cut short that the cut left whole.

    The stream formatter stores at most 65,536 characters of each event, so a
    cell with long output, most often its logs, is stored as JSON that no
    longer parses. json.dumps(asdict(cell)) writes ``code`` first and ``logs``
    last, so ``code`` and the fields after it up to the cut still read whole.
    A field counts only once the comma or brace after it is in the text, so
    a number the cut shortened is not taken for the whole one.
    """
    decoder = json.JSONDecoder()
    fields: dict[str, Any] = {}
    if not text.startswith("{"):
        return fields
    pos = 1
    while True:
        try:
            key, pos = decoder.raw_decode(text, _SPACE.match(text, pos).end())
            pos = _SPACE.match(text, pos).end()
            if not isinstance(key, str) or not text.startswith(":", pos):
                return fields
            value, pos = decoder.raw_decode(text, _SPACE.match(text, pos + 1).end())
        except json.JSONDecodeError:
            return fields
        pos = _SPACE.match(text, pos).end()
        if not text.startswith((",", "}"), pos):
            return fields
        fields[key] = value
        if text.startswith("}", pos):
            return fields
        pos += 1


def _sent_call(*, content: Any) -> _SentCall | None:
    """The call an end event stored, or None when its action isn't readable."""
    parsed = _parse_json(content)
    action = parsed.get("action") if parsed else None
    if not isinstance(action, str):
        return None
    code = parsed.get("code")
    return _SentCall(action=action, code=code if isinstance(code, str) else None)


def _ran_cell(*, content: Any, sent_code: str | None) -> Cell | None:
    """The cell an exec result reports, or None for a result that is no cell.

    Plain text is none: anton closes a cell it never ran, such as a write the
    user didn't approve, with its "Not run: ..." reason as plain text. A
    result cut before its code ends (see _whole_fields) makes a cell from the
    code the model sent, `sent_code`, with no outputs, when the call's end
    event kept that code whole. It often does: json.dumps escapes each
    non-ASCII character to six, while the end event keeps the model's tool
    input as written.
    """
    payload = _parse_json(content)
    if payload is None:
        cut_in_code = isinstance(content, str) and _CELL_JSON_START.match(content.strip())
        if not (cut_in_code and sent_code is not None):
            return None
        payload = {"code": sent_code}
    # A result cut before its outputs still makes a cell, so the write check
    # sees the code it ran.
    with contextlib.suppress(TypeError, ValueError):
        return Cell(**{**_NO_OUTPUT, **payload})
    return None


def extract_scratchpad_cells(events: Iterable[Any]) -> list[Cell]:
    """Rebuild `Cell` instances from persisted assistant streaming events,
    given in history order.

    Each stored event_data dict has a flat structure with a `thought_role` key
    and a `content` key carrying the JSON payload for scratchpad events. A
    result pairs with the end event of its own `tool_use_id`: when the model
    sends several calls at once, every end event comes before the first
    result. A result stored without one pairs with the end event just before
    it.
    """
    cells: list[Cell] = []
    # The calls whose result hasn't come yet, by tool_use_id.
    sent: dict[str, _SentCall] = {}
    # The last call, for a result stored without a tool_use_id.
    last: _SentCall | None = None

    for data in events:
        if not isinstance(data, dict):
            continue
        role = data.get("thought_role")
        tool_use_id = data.get("tool_use_id")
        tool_use_id = tool_use_id if isinstance(tool_use_id, str) else ""

        if role == _SCRATCHPAD_END:
            call = _sent_call(content=data.get("content"))
            if call is None:
                continue
            if call.action == "reset":
                # The calls sent before it ran before it, so it cleared them too.
                cells.clear()
                sent.clear()
                last = None
            else:
                last = call
                if tool_use_id:
                    sent[tool_use_id] = call

        elif role == _SCRATCHPAD_RESULT:
            call = sent.pop(tool_use_id, None) if tool_use_id else last
            last = None
            if call is not None and call.action == "exec":
                cell = _ran_cell(content=data.get("content"), sent_code=call.code)
                if cell is not None:
                    cells.append(cell)

    return cells
