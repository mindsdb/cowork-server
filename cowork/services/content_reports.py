"""Accept authenticated reports without exposing the moderation queue."""
import json
from datetime import datetime, timedelta, timezone
from uuid import UUID
from fastapi import HTTPException
from cowork.db.scoped import ScopedSession
from cowork.models.content_report import ContentReport
from cowork.models.message import Message
from cowork.services.conversations import ConversationService


def submit_report(session: ScopedSession, *, conversation_id: UUID, message_id: UUID) -> dict:
    scope = session.scope
    if not scope.org_mode or not scope.org_id or not scope.user_id:
        raise HTTPException(401, "Sign in to report content.")
    try:
        ConversationService(session).get_conversation(conversation_id)
    except ValueError:
        raise HTTPException(404, "Content not found.") from None
    message = session.get(Message, message_id)
    if message is None or message.conversation_id != conversation_id or message.role != "assistant":
        raise HTTPException(404, "Content not found.")
    content = message.content.model_dump() if hasattr(message.content, "model_dump") else message.content
    return record_report(session, target_key=f"message:{message_id}", conversation_id=conversation_id,
                         message_id=message_id, evidence=json.dumps(content, ensure_ascii=False)[:65536])


def record_report(session: ScopedSession, *, target_key: str, evidence: str, **target) -> dict:
    scope = session.scope
    if not scope.org_mode or not scope.org_id or not scope.user_id:
        raise HTTPException(401, "Sign in to report content.")
    existing = session.exec(session.select(ContentReport).where(
        ContentReport.user_id == scope.user_id, ContentReport.target_key == target_key,
    )).first()
    if existing:
        return {"id": str(existing.id), "status": "received"}
    recent = session.exec(session.select(ContentReport).where(
        ContentReport.user_id == scope.user_id,
        ContentReport.created_at >= datetime.now(timezone.utc) - timedelta(days=1),
    ).limit(100)).all()
    if len(recent) >= 100:
        raise HTTPException(429, "Too many reports. Please try again later.", headers={"Retry-After": "3600"})
    report = ContentReport(org_id=scope.org_id, user_id=scope.user_id,
        target_key=target_key, evidence=evidence[:65536], **target)
    session.add(report)
    session.flush()
    return {"id": str(report.id), "status": "received"}
