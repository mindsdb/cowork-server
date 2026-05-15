"""Rebuild Anton `Cell` lists from persisted assistant streaming events."""

from __future__ import annotations

import contextlib
import json
from typing import Any

from anton.core.backends.base import Cell

# Must match ``minds.schemas.chat.Role`` without importing ``minds.schemas`` (package
# ``__init__`` pulls FastAPI and more — problematic for tight unit tests).
_SCRATCHPAD_END = "thought.scratchpad.end"
_SCRATCHPAD_RESULT = "thought.scratchpad.result"


def _first_stream_output(event: dict | None) -> dict | None:
    if not event:
        return None
    output = event.get("response", {}).get("output")
    if not output:
        return None
    return output[0]


def _parse_first_content_json(output: dict) -> dict[str, Any] | None:
    """Parse JSON from the first text block in a streamed output item."""
    content = output.get("content")
    if not content:
        return None
    raw = (content[0] or {}).get("text")
    if not isinstance(raw, str) or not (t := raw.strip()):
        return None
    try:
        data = json.loads(t)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return None
    return None


def extract_scratchpad_cells_from_message_events(messages: list[Any]) -> list[Cell]:
    """Rebuild `Cell` instances from persisted assistant streaming events.

    Streaming emits ``thought.scratchpad.end`` (tool JSON) *before* progress chunks,
    then ``thought.scratchpad.result``. Pairing uses pending tool action, not
    event adjacency (progress breaks ``previous_event`` pairing).
    """
    cells: list[Cell] = []
    pending_action: str | None = None

    for message in messages:
        events = getattr(message, "events", None) or []
        for event in events:
            output = _first_stream_output(event)
            if not output:
                continue
            role = output.get("role")

            if role == _SCRATCHPAD_END:
                data = _parse_first_content_json(output)
                if data:
                    act = data.get("action")
                    if act == "reset":
                        cells.clear()
                        pending_action = None
                    elif isinstance(act, str):
                        pending_action = act

            elif role == _SCRATCHPAD_RESULT:
                if pending_action == "exec":
                    payload = _parse_first_content_json(output)
                    if payload is not None:
                        with contextlib.suppress(TypeError, ValueError):
                            cells.append(Cell(**payload))
                pending_action = None

    return cells
