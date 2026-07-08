from __future__ import annotations

from uuid import UUID

from sqlmodel import Session, select

from cowork.models.conversation import Conversation
from cowork.models.message import Message
from cowork.models.message_event import MessageEvent
from cowork.models.project import Project
from cowork.services.projects import GENERAL_PROJECT_ID

# Visible stand-in for a turn that produced no assistant text — keeps the
# (user, assistant) pairing intact (see save_assistant_turn docstring).
EMPTY_TURN_PLACEHOLDER = "[No response was produced for this turn.]"


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
        content = text or EMPTY_TURN_PLACEHOLDER
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

    def begin_assistant_turn(
        self,
        conversation_id: UUID,
        harness: str | None = None,
    ) -> Message:
        """Create the assistant row up front so streamed events have a
        durable parent before their SSE strings leave the server
        (write-ahead persistence — see ResponsesHandler._stream).

        Content starts as the placeholder and is overwritten by
        finalize_assistant_turn once the turn's text is known, so even a
        crash mid-turn leaves a well-formed (user, assistant) pair whose
        events replay the progress made so far.
        """
        assistant_msg = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=EMPTY_TURN_PLACEHOLDER,
            harness=harness,
        )
        self.session.add(assistant_msg)
        self.session.commit()
        self.session.refresh(assistant_msg)
        return assistant_msg

    def append_event(
        self,
        message_id: UUID,
        sequence_number: int,
        event_data: dict,
    ) -> None:
        """Durably persist one streaming event (committed immediately).

        Callers rely on write-ahead ordering: this must be called BEFORE
        the event's SSE string is yielded to the client, so a disconnect
        can only lose bytes on the wire — never recorded progress.
        """
        self.session.add(MessageEvent(
            message_id=message_id,
            sequence_number=sequence_number,
            event_data=event_data,
        ))
        self.session.commit()

    def finalize_assistant_turn(self, message_id: UUID, text: str) -> None:
        """Stamp the assistant text once the stream ends (or is cut)."""
        message = self.session.get(Message, message_id)
        if message is None:
            return
        message.content = text or EMPTY_TURN_PLACEHOLDER
        self.session.add(message)
        self.session.commit()

    def latest_assistant_message(self, conversation_id: UUID) -> Message | None:
        """Newest assistant turn of a conversation — the turn
        /responses/tail replays."""
        return self.session.exec(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .where(Message.role == "assistant")
            .order_by(Message.created_at.desc(), Message.id.desc())  # type: ignore[union-attr]
            .limit(1)
        ).first()

    def get_turn_events(self, message_id: UUID, from_seq: int = 0) -> list[MessageEvent]:
        """Events of one assistant turn from `from_seq` on, in sequence
        order — the replay path /responses/tail serves. Sequence numbers
        are the 0-based message_events row numbering (same numbering
        save_assistant_turn and append_event write), so a client that
        resumes with `from_seq = last_seen + 1` gets a gapless tail."""
        return list(self.session.exec(
            select(MessageEvent)
            .where(MessageEvent.message_id == message_id)
            .where(MessageEvent.sequence_number >= from_seq)
            .order_by(MessageEvent.sequence_number)
        ).all())

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
