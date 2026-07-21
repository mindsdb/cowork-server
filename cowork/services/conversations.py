from __future__ import annotations

from uuid import UUID

from sqlalchemy import case
from sqlmodel import select

from cowork.db.scoped import ScopedSession
from cowork.models.conversation import Conversation
from cowork.models.message import Message
from cowork.models.message_event import MessageEvent
from cowork.models.project import Project
from cowork.schemas.responses import Role
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.task_objects import TaskObjectService

# Streaming turns persist user + assistant in one persist() call; on SQLite
# both rows often share the same created_at (second precision). Tie-break
# user before assistant, then id.
_MESSAGE_ORDER = (
    Message.created_at,
    case((Message.role == Role.user, 0), else_=1),
    Message.id,
)


class ConversationService:
    def __init__(self, session: ScopedSession) -> None:
        self.session = session

    def list_conversations(
        self,
        project_id: UUID | None = None,
        limit: int = 50,
        all_projects: bool = False,
    ) -> list[Conversation]:
        stmt = self.session.select(Conversation)
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
        # Anchor the parent: the target project must be visible in scope —
        # otherwise org A could link a conversation to org B's project and
        # leak its name/path through serialization.
        target_project_id = project_id or GENERAL_PROJECT_ID
        if self.session.get(Project, target_project_id) is None:
            raise ValueError("Project not found")
        conversation = Conversation(
            topic=topic,
            project_id=target_project_id,
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
        return self.session.exec(
            self.session.select(Project).where(Project.name == name)
        ).first()

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
            # Anchor the move target: the project must be visible in scope.
            if self.session.get(Project, project_id) is None:
                raise ValueError("Project not found")
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
            self.session.select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(*_MESSAGE_ORDER)
        ).all()
        for message in messages:
            for event in self.session.exec(
                self.session.select(MessageEvent).where(MessageEvent.message_id == message.id)
            ).all():
                self.session.delete(event)
            self.session.delete(message)
        # Drop the conversation's object index too — otherwise the rows
        # outlive the conversation as orphans pointing at artifacts no
        # task owns anymore.
        TaskObjectService(self.session).delete_for_conversation(conversation)
        # Drop the conversation's uploaded attachments (rows + bytes) — they're
        # keyed by conversation id and would otherwise orphan in the file store
        # forever, invisible in any UI (ENG-701). Stage the row deletes into
        # THIS transaction (single commit below, so a crash can't leave a
        # half-deleted "ghost" conversation), then unlink the bytes only after
        # the commit succeeds.
        from cowork.services.files import (
            FileService,
            attachment_purpose,
            unlink_file_dirs,
        )
        attachment_dirs = FileService(self.session).delete_by_purpose(
            attachment_purpose(str(conversation_id))
        )
        self.session.delete(conversation)
        self.session.commit()
        unlink_file_dirs(attachment_dirs)
        return True

    def delete_turn(self, conversation_id: UUID, turn_index: int) -> int:
        """Delete a turn and everything after it.

        turn_index is 0-based counting only assistant messages. The turn's
        preceding user message (if any) and all subsequent messages are
        removed. Returns the number of messages deleted.
        """
        conversation = self.get_conversation(conversation_id)  # raises if not found
        messages = list(
            self.session.exec(
                self.session.select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(*_MESSAGE_ORDER)
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
                self.session.select(MessageEvent).where(MessageEvent.message_id == msg.id)
            ).all():
                self.session.delete(event)
            self.session.delete(msg)
        # Clearing the whole history (truncate from turn 0) is the UI's
        # "delete chat history". When nothing remains, the conversation no
        # longer owns anything it produced — drop its object index so a
        # cleared chat doesn't keep resurfacing old artifacts. A partial
        # truncation leaves the index alone (rows aren't turn-scoped, and
        # surviving turns may still reference the artifact).
        if cut_from == 0:
            TaskObjectService(self.session).delete_for_conversation(conversation)
        self.session.commit()
        return len(to_delete)

    def save_assistant_turn(
        self,
        conversation_id: UUID,
        text: str,
        events: list[dict],
        harness: str | None = None,
    ) -> None:
        """Persist an assistant message and its streaming events."""
        # Persist when there's body text OR any events — an artifact-only turn
        # (the agent writes a file and says little/nothing) carries no text but
        # emits a `response.artifact_created` event, and that event must survive
        # reload so the inline card replays identically.
        if not text and not events:
            return
        # Anchor the write to a parent loaded through THIS session's scope —
        # detached writers (producer) call this on a fresh session, and the
        # conversation may be gone or out-of-scope by now.
        self.get_conversation(conversation_id)
        assistant_msg = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=text,
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
            self.session.select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(*_MESSAGE_ORDER)
        ).all()
        result = []
        for message in messages:
            events = self.session.exec(
                self.session.select(MessageEvent)
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
