from __future__ import annotations

from collections.abc import AsyncGenerator
from uuid import UUID

from sqlmodel import Session

from cowork.harnesses.base import get_harness
from cowork.models.message import Message as DBMessage
from cowork.models.message_event import MessageEvent
from cowork.schemas.responses import (
    Response,
    ResponseOutput,
    ResponseOutputContent,
    ResponseStatus,
    ResponsesRequest,
    Role,
)
from cowork.services.conversations import ConversationService


class ResponsesHandler:
    def __init__(self, session: Session) -> None:
        self.session = session
        # TODO: Get the harness from settings? Request context?
        self.harness = get_harness("anton")

    async def handle(self, request: ResponsesRequest) -> AsyncGenerator[str, None] | Response:
        conversation_service = ConversationService(self.session)

        if request.conversation:
            conversation = conversation_service.get_conversation(UUID(request.conversation))
        else:
            conversation = conversation_service.create_conversation(topic=self._extract_prompt(request)[:80])

        prompt = self._extract_prompt(request)

        user_message = DBMessage(
            conversation_id=conversation.id,
            role=Role.user,
            content=prompt,
        )
        self.session.add(user_message)
        self.session.commit()
        self.session.refresh(user_message)

        stream = self.harness.stream_response(
            conversation=conversation,
            prompt=prompt,
            model=request.model,
        )

        if request.stream:
            return self._stream(stream, conversation.id, request.model, str(user_message.id))

        return await self._collect(stream, conversation.id, request.model, str(user_message.id))

    async def _stream(
        self,
        stream,
        conversation_id: UUID,
        model: str,
        output_item_id: str,
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

    @staticmethod
    def _extract_prompt(request: ResponsesRequest) -> str:
        if isinstance(request.input, str):
            return request.input
        if isinstance(request.input, list):
            for msg in reversed(request.input):
                if msg.role == Role.user and msg.content:
                    return msg.content if isinstance(msg.content, str) else str(msg.content)
        return ""

    @staticmethod
    def _build_output(item_id: str, text: str) -> ResponseOutput:
        return ResponseOutput(
            id=item_id,
            status=ResponseStatus.completed,
            content=[ResponseOutputContent(text=text)],
        )
