"""Rebuild Anton `Cell` lists from persisted assistant streaming events."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterable
from typing import Any

from anton.core.backends.base import Cell

_SCRATCHPAD_END = "thought.scratchpad.end"
_SCRATCHPAD_RESULT = "thought.scratchpad.result"

# The only thought roles extract_scratchpad_cells reacts to. The harness reads
# just the events carrying one of them, so a role the extractor starts to
# read has to be added here too, or its events never reach it.
SCRATCHPAD_REPLAY_ROLES = (_SCRATCHPAD_END, _SCRATCHPAD_RESULT)


def _parse_json(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, str) or not (t := raw.strip()):
        return None
    try:
        data = json.loads(t)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


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
                    with contextlib.suppress(TypeError, ValueError):
                        cells.append(Cell(**payload))
            pending_action = None

    return cells
