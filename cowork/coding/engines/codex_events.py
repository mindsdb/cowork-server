from __future__ import annotations

import re
from typing import Any

from cowork.coding.contracts import CodingEvent, EventType
from cowork.coding.redaction import redact_secrets, redact_text, sanitize


def map_codex_notification(method: str, payload: object) -> CodingEvent | None:
    raw = payload_dict(payload)
    item_id = string(raw.get("itemId") or raw.get("item_id")) or None
    turn_id = string(raw.get("turnId") or raw.get("turn_id")) or None
    if method == "item/agentMessage/delta":
        return CodingEvent(type=EventType.agent_message, text=string(raw.get("delta")), phase="progress", item_id=item_id, turn_id=turn_id)
    if method in {"item/reasoning/summaryTextDelta", "item/reasoning/textDelta"}:
        return CodingEvent(type=EventType.reasoning, title="Reasoning", text=string(raw.get("delta")), phase="progress", item_id=item_id, turn_id=turn_id)
    if method == "turn/plan/updated":
        return CodingEvent(type=EventType.plan, title="Plan updated", phase="progress", turn_id=turn_id, data=sanitize(raw))
    if method == "item/commandExecution/outputDelta":
        return CodingEvent(type=EventType.command, title="Command output", text=string(raw.get("delta")), phase="progress", item_id=item_id, turn_id=turn_id)
    if method == "item/fileChange/outputDelta":
        return CodingEvent(type=EventType.file_change, title="File change", text=string(raw.get("delta")), phase="progress", item_id=item_id, turn_id=turn_id)
    if method == "turn/diff/updated":
        return CodingEvent(type=EventType.diff, title="Working diff updated", text=string(raw.get("diff")), phase="progress", turn_id=turn_id)
    if method == "thread/tokenUsage/updated":
        return CodingEvent(type=EventType.usage, title="Usage updated", phase="progress", turn_id=turn_id, data=sanitize(raw.get("tokenUsage") or raw))
    if method in {"item/started", "item/completed"}:
        item = raw.get("item") if isinstance(raw.get("item"), dict) else {}
        item_type = string(item.get("type")) or "tool"
        phase = "started" if method.endswith("started") else "completed"
        return CodingEvent(
            type=event_type_for_item(item_type),
            title=item_title(item_type, item),
            phase=phase,
            item_id=string(item.get("id")) or item_id,
            turn_id=turn_id,
            data=sanitize(item),
        )
    if method == "error":
        nested = raw.get("error") if isinstance(raw.get("error"), dict) else {}
        message = raw.get("message") or nested.get("message") or nested.get("additionalDetails") or "The coding agent reported an error."
        return CodingEvent(type=EventType.error, title="Agent error", text=redact_text(string(message)), phase="failed", turn_id=turn_id)
    if method == "turn/completed":
        turn = raw.get("turn") if isinstance(raw.get("turn"), dict) else {}
        status = string(turn.get("status")) or "completed"
        error = turn.get("error") if isinstance(turn.get("error"), dict) else {}
        return CodingEvent(
            type=EventType.session,
            title={"completed": "Task completed", "interrupted": "Task interrupted", "failed": "Task failed"}.get(status, "Turn finished"),
            text=string(error.get("message")),
            phase="completed" if status == "completed" else "failed",
            turn_id=string(turn.get("id")) or turn_id,
            data={"status": status},
        )
    if method == "turn/started":
        return CodingEvent(type=EventType.session, title="Agent started", phase="started", turn_id=turn_id)
    return None


def payload_dict(payload: object) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "model_dump"):
        dumped = payload.model_dump(by_alias=True, mode="json", exclude_none=True)
        return dumped if isinstance(dumped, dict) else {}
    params = getattr(payload, "params", None)
    return params if isinstance(params, dict) else {}


def event_type_for_item(item_type: str) -> EventType:
    lowered = item_type.lower()
    if "collab" in lowered or "subagent" in lowered or "sub_agent" in lowered:
        return EventType.child_work
    if "command" in lowered:
        return EventType.command
    if "file" in lowered or "patch" in lowered:
        return EventType.file_change
    if "reason" in lowered:
        return EventType.reasoning
    if "agentmessage" in lowered:
        return EventType.agent_message
    return EventType.tool


def item_title(item_type: str, item: dict) -> str:
    if event_type_for_item(item_type) == EventType.child_work:
        for key in ("description", "prompt", "message"):
            value = string(item.get(key))
            if value:
                return value.splitlines()[0][:160]
        return "Parallel work"
    for key in ("name", "command", "path", "title"):
        value = string(item.get(key))
        if value:
            return value[:240]
    words = re.sub(r"(?<!^)(?=[A-Z])", " ", item_type).replace("_", " ").strip()
    return words[:240].capitalize() or "Agent activity"


def string(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def enum_value(value: object) -> str:
    return string(getattr(value, "value", value))


def redact_event(event: CodingEvent, secrets: tuple[str, ...]) -> CodingEvent:
    event.title = redact_text(event.title, secrets)
    event.text = redact_text(event.text, secrets)
    event.data = redact_secrets(event.data, secrets)
    return event
