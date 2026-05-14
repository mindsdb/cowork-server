from __future__ import annotations

from uuid import UUID

from sqlmodel import Session, select

from cowork.models.conversation import Conversation
from cowork.models.message import Message
from cowork.models.message_event import MessageEvent
from cowork.services.projects import GENERAL_PROJECT_ID


class ConversationService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def list_conversations(
        self,
        project_id: UUID | None = None,
        limit: int = 50,
    ) -> list[Conversation]:
        effective_project_id = project_id or GENERAL_PROJECT_ID
        return list(
            self.session.exec(
                select(Conversation)
                .where(Conversation.project_id == effective_project_id)
                .limit(limit)
            ).all()
        )

    def get_conversation(self, conversation_id: UUID) -> Conversation:
        conversation = self.session.get(Conversation, conversation_id)
        if conversation is None:
            raise ValueError("Conversation not found")
        return conversation

    def create_conversation(
        self,
        topic: str,
        project_id: UUID | None = None,
    ) -> Conversation:
        conversation = Conversation(
            topic=topic,
            project_id=project_id or GENERAL_PROJECT_ID,
        )
        self.session.add(conversation)
        self.session.commit()
        self.session.refresh(conversation)
        return conversation

    def update_conversation(
        self,
        conversation_id: UUID,
        topic: str | None = None,
        project_id: UUID | None = None,
    ) -> Conversation:
        conversation = self.session.get(Conversation, conversation_id)
        if conversation is None:
            raise ValueError("Conversation not found")
        if topic is not None:
            conversation.topic = topic
        if project_id is not None:
            conversation.project_id = project_id
        self.session.add(conversation)
        self.session.commit()
        self.session.refresh(conversation)
        return conversation

    def delete_conversation(self, conversation_id: UUID) -> bool:
        conversation = self.session.get(Conversation, conversation_id)
        if conversation is None:
            return False
        messages = self.session.exec(
            select(Message).where(Message.conversation_id == conversation_id)
        ).all()
        for message in messages:
            for event in self.session.exec(
                select(MessageEvent).where(MessageEvent.message_id == message.id)
            ).all():
                self.session.delete(event)
            self.session.delete(message)
        self.session.delete(conversation)
        self.session.commit()
        return True

    def get_messages(self, conversation_id: UUID) -> list[dict]:
        self.get_conversation(conversation_id)  # raises if not found
        messages = self.session.exec(
            select(Message).where(Message.conversation_id == conversation_id)
        ).all()
        result = []
        for message in messages:
            events = self.session.exec(
                select(MessageEvent)
                .where(MessageEvent.message_id == message.id)
                .order_by(MessageEvent.sequence_number)
            ).all()
            result.append({
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at,
                "events": [
                    {
                        "id": e.id,
                        "sequence_number": e.sequence_number,
                        "event_data": e.event_data,
                    }
                    for e in events
                ],
            })
        return result
