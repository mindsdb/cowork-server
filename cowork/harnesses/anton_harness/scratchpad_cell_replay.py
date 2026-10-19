"""Rebuild Anton `Cell` lists from persisted assistant streaming events."""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Iterable
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


def extract_scratchpad_cells(events: Iterable[Any]) -> list[Cell]:
    """Rebuild `Cell` instances from persisted assistant streaming events,
    given in history order.

    Each stored event_data dict has a flat structure with a `thought_role` key
    and a `content` key carrying the JSON payload for scratchpad events.
    """
    cells: list[Cell] = []
    pending_action: str | None = None

    for data in events:
        if not isinstance(data, dict):
            continue
        role = data.get("thought_role")

        if role == _SCRATCHPAD_END:
            parsed = _parse_json(data.get("content"))
            if parsed:
                act = parsed.get("action")
                if act == "reset":
                    cells.clear()
                    pending_action = None
                elif isinstance(act, str):
                    pending_action = act

        elif role == _SCRATCHPAD_RESULT:
            if pending_action == "exec":
                payload = _parse_json(data.get("content"))
                if payload is not None:
                    # A result cut before its outputs still makes a cell, so
                    # the write check sees the code it ran.
                    with contextlib.suppress(TypeError, ValueError):
                        cells.append(Cell(**{**_NO_OUTPUT, **payload}))
            pending_action = None

    return cells
