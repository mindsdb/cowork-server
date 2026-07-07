"""Rebuild Anton `Cell` lists from persisted assistant streaming events."""

from __future__ import annotations

import contextlib
import json
from typing import Any

from anton.core.backends.base import Cell

_SCRATCHPAD_END = "thought.scratchpad.end"
_SCRATCHPAD_RESULT = "thought.scratchpad.result"


def _parse_json(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, str) or not (t := raw.strip()):
        return None
    try:
        data = json.loads(t)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def extract_scratchpad_cells_from_message_events(messages: list[Any]) -> list[Cell]:
    """Rebuild `Cell` instances from persisted assistant streaming events.

    Each stored event_data dict has a flat structure with a `thought_role` key
    and a `content` key carrying the JSON payload for scratchpad events.
    """
    cells: list[Cell] = []
    pending_action: str | None = None

    for message in messages:
        events = getattr(message, "message_events", None) or []
        for event in events:
            data = event.event_data if hasattr(event, "event_data") else event
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
