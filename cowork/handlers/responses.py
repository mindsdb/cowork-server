from __future__ import annotations

from collections.abc import AsyncGenerator
from uuid import UUID

from fastapi import HTTPException
from sqlmodel import Session

from cowork.harnesses.base import get_harness
from cowork.models.message import Message as DBMessage
from cowork.models.message_event import MessageEvent
from cowork.schemas.responses import (
    Content,
    ContentType,
    Response,
    ResponseOutput,
    ResponseOutputContent,
    ResponseStatus,
    ResponsesRequest,
    Role,
)
from cowork.services.conversations import ConversationService
from cowork.services.files import FileService


class ResponsesHandler:
    def __init__(self, session: Session) -> None:
        self.session = session
        # TODO: Get the harness from settings? Request context?
        self.harness = get_harness("anton")

    async def handle(self, request: ResponsesRequest) -> AsyncGenerator[str, None] | Response:
        conversation_service = ConversationService(self.session)

        harness_input = self._build_harness_input(request)
        original_content = self._extract_original_content(request)

        if request.conversation:
            conversation = conversation_service.get_conversation(UUID(request.conversation))
        else:
            conversation = conversation_service.create_conversation(topic=self._prompt_text(harness_input)[:80])

        user_message = DBMessage(
            conversation_id=conversation.id,
            role=Role.user,
            content=original_content,
        )
        self.session.add(user_message)
        self.session.commit()
        self.session.refresh(user_message)

        stream = self.harness.stream_response(
            conversation=conversation,
            input=harness_input,
            model=request.model,
        )

        if request.stream:
            return self._stream(stream, conversation.id, request.model)

        return await self._collect(stream, conversation.id, request.model, str(user_message.id))

    async def _stream(
        self,
        stream,
        conversation_id: UUID,
        model: str,
    ) -> AsyncGenerator[str, None]:
        collected_text: list[str] = []
        collected_events: list[dict] = []

        def event_sink(event_type: str, data: dict) -> None:
            collected_events.append(data)
            if event_type == "response.output_text.delta":
                collected_text.append(data.get("delta", ""))

        async for sse_string in self.harness.formatter(stream, model, event_sink):
            yield sse_string

        self._save_assistant_turn(conversation_id, "".join(collected_text), collected_events)

    async def _collect(
        self,
        stream,
        conversation_id: UUID,
        model: str,
        output_item_id: str,
    ) -> Response:
        collected_text: list[str] = []
        collected_events: list[dict] = []

        def event_sink(event_type: str, data: dict) -> None:
            collected_events.append(data)
            if event_type == "response.output_text.delta":
                collected_text.append(data.get("delta", ""))

        async for _ in self.harness.formatter(stream, model, event_sink):
            pass

        assistant_text = "".join(collected_text)
        self._save_assistant_turn(conversation_id, assistant_text, collected_events)

        return Response(
            status=ResponseStatus.completed,
            model=model,
            output=[self._build_output(output_item_id, assistant_text)],
        )

    def _save_assistant_turn(
        self,
        conversation_id: UUID,
        text: str,
        events: list[dict],
    ) -> None:
        if not text:
            return
        assistant_msg = DBMessage(
            conversation_id=conversation_id,
            role=Role.assistant,
            content=text,
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

    def _build_harness_input(self, request: ResponsesRequest) -> list[dict]:
        if isinstance(request.input, str):
            return [{"type": "text", "text": request.input}]
        if isinstance(request.input, list):
            for msg in reversed(request.input):
                if msg.role == Role.user and msg.content:
                    if isinstance(msg.content, str):
                        return [{"type": "text", "text": msg.content}]
                    if isinstance(msg.content, list):
                        blocks: list[dict] = []
                        for item in msg.content:
                            if isinstance(item, Content):
                                if item.type == ContentType.text and item.text:
                                    blocks.append({"type": "text", "text": item.text})
                                elif item.type == ContentType.file and item.file_id:
                                    file = FileService(self.session).get_file(UUID(item.file_id))
                                    if file is None:
                                        raise HTTPException(status_code=404, detail=f"File {item.file_id!r} not found")
                                    blocks.append({"type": "file", "path": file.path, "filename": file.filename})
                        return blocks
        return [{"type": "text", "text": ""}]

    @staticmethod
    def _prompt_text(harness_input: list[dict]) -> str:
        return " ".join(b["text"] for b in harness_input if b.get("type") == "text")

    @staticmethod
    def _extract_original_content(request: ResponsesRequest) -> str | list:
        if isinstance(request.input, str):
            return request.input
        if isinstance(request.input, list):
            for msg in reversed(request.input):
                if msg.role == Role.user and msg.content:
                    if isinstance(msg.content, str):
                        return msg.content
                    if isinstance(msg.content, list):
                        return [item.model_dump() if isinstance(item, Content) else item for item in msg.content]
        return ""

    @staticmethod
    def _build_output(item_id: str, text: str) -> ResponseOutput:
        return ResponseOutput(
            id=item_id,
            status=ResponseStatus.completed,
            content=[ResponseOutputContent(text=text)],
        )
