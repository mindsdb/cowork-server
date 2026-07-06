from __future__ import annotations

from uuid import UUID

from sqlmodel import Session, select

from cowork.models.conversation import Conversation
from cowork.models.message import Message
from cowork.models.message_event import MessageEvent
from cowork.models.project import Project
from cowork.services.projects import GENERAL_PROJECT_ID


class ConversationService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def list_conversations(
        self,
        project_id: UUID | None = None,
        limit: int = 50,
        all_projects: bool = False,
    ) -> list[Conversation]:
        stmt = select(Conversation)
        if not all_projects:
            stmt = stmt.where(Conversation.project_id == (project_id or GENERAL_PROJECT_ID))
        stmt = stmt.order_by(Conversation.created_at.desc()).limit(limit)  # type: ignore[union-attr]
        return list(self.session.exec(stmt).all())

    def get_conversation(self, conversation_id: UUID) -> Conversation:
        conversation = self.session.get(Conversation, conversation_id)
        if conversation is None:
            raise ValueError("Conversation not found")
        return conversation

    def create_conversation(
        self,
        topic: str,
        project_id: UUID | None = None,
        conversation_id: UUID | None = None,
    ) -> Conversation:
        """`conversation_id` lets the caller adopt a client-allocated id —
        the composer allocates one up front so attachments can be uploaded
        against it before the first stream creates the conversation."""
        conversation = Conversation(
            topic=topic,
            project_id=project_id or GENERAL_PROJECT_ID,
        )
        if conversation_id is not None:
            conversation.id = conversation_id
        self.session.add(conversation)
        self.session.commit()
        self.session.refresh(conversation)
        return conversation

    def project_by_name(self, name: str | None) -> Project | None:
        if not name:
            return None
        return self.session.exec(select(Project).where(Project.name == name)).first()

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
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
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

    def delete_turn(self, conversation_id: UUID, turn_index: int) -> int:
        """Delete a turn and everything after it.

        turn_index is 0-based counting only assistant messages. The turn's
        preceding user message (if any) and all subsequent messages are
        removed. Returns the number of messages deleted.
        """
        self.get_conversation(conversation_id)  # raises if not found
        messages = list(
            self.session.exec(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at)
            ).all()
        )
        # Find the Nth assistant message (0-based).
        assistant_count = -1
        cut_from = None
        for i, m in enumerate(messages):
            if m.role.value == "assistant":
                assistant_count += 1
                if assistant_count == turn_index:
                    # Include the preceding user message in the cut if it
                    # exists and is immediately before this assistant msg.
                    if i > 0 and messages[i - 1].role.value == "user":
                        cut_from = i - 1
                    else:
                        cut_from = i
                    break
        if cut_from is None:
            raise ValueError(f"Turn {turn_index} not found")
        to_delete = messages[cut_from:]
        for msg in to_delete:
            for event in self.session.exec(
                select(MessageEvent).where(MessageEvent.message_id == msg.id)
            ).all():
                self.session.delete(event)
            self.session.delete(msg)
        self.session.commit()
        return len(to_delete)

    def save_assistant_turn(
        self,
        conversation_id: UUID,
        text: str,
        events: list[dict],
        harness: str | None = None,
    ) -> None:
        """Persist an assistant message and its streaming events.

        A turn that produced no text (every provider rate-limited, a CLI
        coworker errored before output, etc.) must still leave an
        assistant row: the user message was already committed, so
        dropping the turn would put two consecutive user messages in the
        history — which downstream harnesses reject or mishandle (the
        "consecutive Role.user" warning). Persist a visible placeholder
        instead so `(user, assistant)` pairing always holds. Empty turns
        that carry no events at all (nothing happened) are still skipped.
        """
        if not text and not events:
            return
        content = text or "[No response was produced for this turn.]"
        assistant_msg = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            harness=harness,
        )
        self.session.add(assistant_msg)
        self.session.commit()
        self.session.refresh(assistant_msg)

        for seq, event_data in enumerate(events):
            self.session.add(MessageEvent(
                message_id=assistant_msg.id,
                sequence_number=seq,
                event_data=event_data,
            ))
        if events:
            self.session.commit()

    def get_messages(self, conversation_id: UUID) -> list[dict]:
        self.get_conversation(conversation_id)  # raises if not found
        messages = self.session.exec(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at, Message.id)
        ).all()
        result = []
        for message in messages:
            events = self.session.exec(
                select(MessageEvent)
                .where(MessageEvent.message_id == message.id)
                .order_by(MessageEvent.sequence_number)
            ).all()
            item = {
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at,
                "events": [e.event_data for e in events],
            }
            if message.harness:
                item["harness"] = message.harness
            result.append(item)
        return result
