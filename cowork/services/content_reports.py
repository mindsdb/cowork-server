"""Accept authenticated reports without exposing the moderation queue."""
import json
import math
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from uuid import UUID
from fastapi import HTTPException
from cowork.db.scoped import ScopedSession
from cowork.models.content_report import ContentReport
from cowork.models.message import Message
from cowork.models.message_event import MessageEvent
from cowork.schemas.responses import Role
from cowork.services.conversations import ConversationService, _is_tool_row


# Persist only fields shown on cards or needed to identify them. In particular,
# tool arguments/results, sharing credentials and private turn history stay out.
_CARD_FIELDS = {
    "response.artifact_created": ("artifact", ("id", "projectId", "slug", "title", "description", "type")),
    "response.skill_created": ("skill", ("slug", "name", "description")),
    "response.connector_drafted": ("connector", ("connector_id", "name", "description")),
}


def _visible_cards(event: dict) -> Iterator[dict]:
    kind = event.get("type")
    if not isinstance(kind, str):
        return
    if kind in _CARD_FIELDS:
        key, fields = _CARD_FIELDS[kind]
        payload = event.get(key)
    elif kind == "response.quick_edit_group" and isinstance(event.get("edits"), list):
        for edit in event["edits"]:
            if isinstance(edit, dict) and isinstance(edit.get("summary"), str):
                yield {"event": kind, "summary": edit["summary"][:512]}
        return
    elif kind == "response.ask_user":
        fields, payload = ("prompt",), event
    elif kind == "response.in_progress" and event.get("thought_role") == Role.thought_tool_call_message.value:
        # A tool-authored message addressed to the user is a visible reply.
        fields, payload = ("content",), event
    elif kind == "response.in_progress" and isinstance(event.get("generated_images"), list):
        for image in event["generated_images"]:
            if isinstance(image, dict) and isinstance(image.get("path"), str):
                yield {"event": kind, "image": image["path"][:512]}
        return
    else:
        return
    if isinstance(payload, dict):
        card = {k: payload[k][:512] for k in fields if isinstance(payload.get(k), str)}
        if card:
            yield {"event": kind, **card}


def _event_evidence(session: ScopedSession, message_id: UUID) -> list[dict]:
    rows = session.exec(
        session.select(MessageEvent).with_only_columns(MessageEvent.event_data)
        .where(MessageEvent.message_id == message_id)
        .order_by(MessageEvent.sequence_number).execution_options(yield_per=32)
    )
    cards = []
    # json.dumps' default separators add up to two characters per card.
    remaining = 65536 - len('{"cards": []}')
    for event in rows:
        if not isinstance(event, dict):
            continue
        for card in _visible_cards(event):
            remaining -= len(json.dumps(card)) + 2
            if remaining < 0 or len(cards) == 16:
                return cards
            cards.append(card)
    return cards


def submit_report(session: ScopedSession, *, conversation_id: UUID, message_id: UUID) -> dict:
    scope = session.scope
    if not scope.org_mode or not scope.org_id or not scope.user_id:
        raise HTTPException(401, "Sign in to report content.")
    try:
        ConversationService(session).get_conversation(conversation_id)
    except ValueError:
        raise HTTPException(404, "Content not found.") from None
    message = session.get(Message, message_id)
    # Messages carry no org of their own, so the match against the caller's
    # conversation is what keeps another tenant's message out of reach. Tool
    # rows are saved as assistant too, but are replay history the reader never
    # saw, and hold raw tool arguments.
    if (
        message is None
        or message.conversation_id != conversation_id
        or message.role != "assistant"
        or _is_tool_row(message.content)
    ):
        raise HTTPException(404, "Content not found.")
    content = message.content.model_dump() if hasattr(message.content, "model_dump") else message.content
    if content:
        evidence = json.dumps(content, ensure_ascii=False)
    else:
        evidence = json.dumps({"cards": _event_evidence(session, message_id)})
    return record_report(session, target_key=f"message:{message_id}", conversation_id=conversation_id,
                         message_id=message_id, evidence=evidence)


def record_report(session: ScopedSession, *, target_key: str, evidence: str, **target) -> dict:
    scope = session.scope
    if not scope.org_mode or not scope.org_id or not scope.user_id:
        raise HTTPException(401, "Sign in to report content.")
    existing = session.exec(session.select(ContentReport).where(
        ContentReport.user_id == scope.user_id, ContentReport.target_key == target_key,
    )).first()
    if existing:
        return {"id": str(existing.id), "status": "received"}
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=1)
    recent = session.exec(session.select(ContentReport).with_only_columns(ContentReport.created_at).where(
        ContentReport.user_id == scope.user_id,
        ContentReport.created_at > window_start,
    ).order_by(ContentReport.created_at.desc()).limit(100)).all()
    if len(recent) >= 100:
        # SQLite returns naive UTC timestamps. PostgreSQL preserves the timezone.
        oldest = recent[-1]
        if oldest.tzinfo is None:
            oldest = oldest.replace(tzinfo=timezone.utc)
        retry_after = max(1, math.ceil((oldest - window_start).total_seconds()))
        raise HTTPException(429, "Too many reports. Please try again later.",
                            headers={"Retry-After": str(retry_after)})
    report = ContentReport(org_id=scope.org_id, user_id=scope.user_id,
        target_key=target_key, evidence=evidence[:65536], **target)
    session.add(report)
    session.flush()
    return {"id": str(report.id), "status": "received"}
