from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncGenerator
from pathlib import Path
from uuid import UUID

from fastapi import HTTPException
from sqlmodel import Session

from cowork.common.settings.user_settings import get_user_settings
from cowork.db.session import get_open_session
from cowork.harnesses.base import get_harness
from cowork.models.message import Message as DBMessage
from cowork.streaming import new_buffer, registry
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
from cowork.handlers.turn_errors import (
    AUTH_ERROR_CODE,
    GENERIC_TURN_ERROR_CODE,
    GENERIC_TURN_ERROR_MESSAGE,
    auth_error_detail,
    friendly_turn_error,
    response_failed_payload,
    response_failed_sse,
)
from cowork.services.conversations import ConversationService
from cowork.services.files import FileService
from cowork.services.projects import GENERAL_PROJECT_ID, ProjectService
from cowork.services.skills import SkillService


import logging

logger = logging.getLogger(__name__)


class ResponsesHandler:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.harness = get_harness(get_user_settings().harness)
        self.last_conversation_id: str | None = None

    async def handle(self, request: ResponsesRequest) -> AsyncGenerator[str, None] | Response:
        logger.info("[responses] handle() called — conversation=%s, stream=%s", request.conversation, request.stream)
        # A per-conversation coworker pick (composer) overrides the global
        # default. Resolved here (not in __init__) so the same handler
        # instance's default construction cost is only paid once per
        # request regardless of whether an override is present.
        if request.harness:
            try:
                self.harness = get_harness(request.harness)
            except ValueError:
                logger.warning("Requested harness '%s' is not registered; using default '%s'",
                                request.harness, self.harness.id)
        await self.harness.sync_skills(SkillService(self.session).list_skills())

        conversation_service = ConversationService(self.session)
        project_id = self._resolve_project_id(request)

        harness_input = self._build_harness_input(request)
        original_content = self._extract_original_content(request)

        if request.conversation:
            try:
                conv_id = UUID(request.conversation)
            except ValueError:
                conv_id = None
            if conv_id is not None:
                try:
                    conversation = conversation_service.get_conversation(conv_id)
                except ValueError:
                    # Unknown UUID — the composer allocates a conversation id
                    # up front so attachments can be uploaded against it before
                    # the first stream. Adopt it, otherwise those uploads strand
                    # under an id no conversation ever gets (ENG-264).
                    conversation = conversation_service.create_conversation(
                        topic=self._prompt_text(harness_input)[:80],
                        project_id=project_id,
                        conversation_id=conv_id,
                    )
            else:
                # Client sent a non-UUID id (e.g. the legacy timestamp
                # allocator, or a name-based format) — it can't become the
                # row id, so create a fresh conversation and re-link any
                # attachments uploaded against the client's id (ENG-264).
                conversation = conversation_service.create_conversation(
                    topic=self._prompt_text(harness_input)[:80],
                    project_id=project_id,
                )
                self._relink_attachments(request.conversation, conversation)
        else:
            conversation = conversation_service.create_conversation(
                topic=self._prompt_text(harness_input)[:80],
                project_id=project_id,
            )

        self.last_conversation_id = str(conversation.id)

        # Pre-load messages before adding the new user message so the ORM
        # cache (and thus the harness's initial_history) doesn't include the
        # current turn's input — it's passed separately via `input`.
        _ = conversation.messages
        # turn_id: prior message count. The current user message is NOT
        # persisted yet (deferred to the producer for the streaming path), so
        # this is a stable per-conversation index for the buffer file.
        turn_id = len(conversation.messages)

        disabled = (
            [dc.model_dump() for dc in request.disabled_connections]
            if request.disabled_connections else None
        )

        if request.stream:
            # Detached + resumable. The agent run executes in a background
            # task that writes events to a per-turn buffer; this request just
            # tails the buffer. Closing the connection never reaches the
            # producer — only an explicit /cancel does. The user message is
            # persisted by the producer at the end (deferred), so the harness
            # reads history WITHOUT the current turn (see _produce).
            buffer = new_buffer(str(conversation.id), turn_id)
            await registry.start(
                conversation_id=str(conversation.id),
                turn_id=turn_id,
                buffer=buffer,
                producer_coro=self._produce(
                    conv_id=conversation.id,
                    harness_input=harness_input,
                    original_content=original_content,
                    model=request.model,
                    disabled=disabled,
                    harness_name=get_user_settings().harness,
                    harness_id=getattr(self.harness, "id", None),
                    buffer=buffer,
                ),
            )
            return sse_from_buffer(buffer, 0)

        # Non-streaming (legacy/rare): persist the user message inline (via FK,
        # not the relationship, so the cached history above stays clean) and
        # run synchronously within the request.
        user_message = DBMessage(
            conversation_id=conversation.id,
            role=Role.user,
            content=original_content,
        )
        self.session.add(user_message)
        self.session.commit()
        self.session.refresh(user_message)

        # `request.model` is a "{provider_slug}/{model_id}" pick from the
        # composer's model picker (see cowork.services.provider_registry).
        # When absent or unresolvable, each harness falls back to its own
        # default routing.
        stream = self.harness.stream_response(
            conversation=conversation,
            input=harness_input,
            model=request.model,
            disabled_connections=[dc.model_dump() for dc in request.disabled_connections]
            if request.disabled_connections else None,
        )
        return await self._collect(stream, conversation.id, request.model, str(user_message.id))

    async def _produce(
        self,
        *,
        conv_id: UUID,
        harness_input: list[dict],
        original_content,
        model: str,
        disabled: list[dict] | None,
        harness_name: str,
        harness_id: str | None,
        buffer,
    ) -> None:
        """Detached producer: run the turn and write events to the buffer.

        Runs in its OWN DB session (it outlives the request). Persistence is
        deferred to the end so the harness reads history WITHOUT the current
        user message; on terminal we persist user + assistant together.
        Never reaches the HTTP response — readers tail the buffer.
        """
        own = get_open_session()
        collected_text: list[str] = []
        pending_events: list[dict] = []

        def event_sink(event_type: str, data: dict) -> None:
            pending_events.append(data)
            if event_type == "response.output_text.delta":
                collected_text.append(data.get("delta", ""))

        conversation_service = ConversationService(self.session)
        harness_id = getattr(self.harness, 'id', None)
        assistant_message_id: UUID | None = None
        next_seq = 0

        event_count = 0
        try:
            async for sse_string in self.harness.formatter(stream, model, event_sink):
                event_count += 1
                if event_count <= 3 or "response.completed" in sse_string:
                    logger.info("[responses] SSE event #%d (first 120 chars): %s", event_count, sse_string[:120].replace('\n', '\\n'))
                # Inject conversation_id and harness into the response.created
                # event so the client learns the canonical id and which agent
                # generated this response.
                if "response.created" in sse_string and "conversation_id" not in sse_string:
                    try:
                        lines = sse_string.strip().split("\n")
                        data_line = next(l for l in lines if l.startswith("data:"))
                        payload = json.loads(data_line[5:])
                        payload["conversation_id"] = str(conversation_id)
                        if harness_id:
                            payload["harness"] = harness_id
                        sse_string = f"event: response.created\ndata: {json.dumps(payload)}\n\n"
                    except Exception:
                        pass
                # Write-ahead: durably commit every event the formatter
                # recorded for this SSE string BEFORE it reaches the client.
                # A disconnect mid-turn can then only lose bytes on the
                # wire — never recorded progress — so the tail/items replay
                # from message_events is always complete. The assistant row
                # is created lazily on the first event so an eventless turn
                # still leaves no row (matches save_assistant_turn).
                if pending_events:
                    if assistant_message_id is None:
                        assistant_message_id = conversation_service.begin_assistant_turn(
                            conversation_id, harness=harness_id,
                        ).id
                    for event_data in pending_events:
                        conversation_service.append_event(
                            assistant_message_id, next_seq, event_data,
                        )
                        next_seq += 1
                    pending_events.clear()
                yield sse_string

            logger.info("[responses] stream finished — %d events, %d chars of text", event_count, len("".join(collected_text)))
        finally:
            # Runs on normal completion AND when the client disconnects or
            # cancels (GeneratorExit/CancelledError at the yield above).
            # Events are already durable; stamp the text collected so far.
            if assistant_message_id is not None:
                conversation_service.finalize_assistant_turn(
                    assistant_message_id, "".join(collected_text),
                )

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

        try:
            async for _ in self.harness.formatter(stream, model, event_sink):
                pass
        except Exception as exc:
            # Mirror the streaming path: a recognised failure (e.g. an
            # unsupported image) surfaces its curated message with a 400;
            # anything else stays a generic 500 so provider internals never
            # leak. (cowork PR #156.)
            friendly = friendly_turn_error(exc)
            if friendly is not None:
                _, message = friendly
                logger.info("[responses] user-facing turn error: %s", exc)
                raise HTTPException(status_code=400, detail=message)
            logger.exception("[responses] turn failed")
            raise HTTPException(status_code=500, detail=GENERIC_TURN_ERROR_MESSAGE)

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
        harness_id = getattr(self.harness, 'id', None)
        ConversationService(self.session).save_assistant_turn(
            conversation_id, text, events, harness=harness_id,
        )

    def _build_harness_input(self, request: ResponsesRequest) -> list[dict]:
        blocks: list[dict] = []

        # Resolve attachment_ids to image/file blocks
        if request.attachment_ids:
            file_svc = FileService(self.session)
            for aid in request.attachment_ids:
                try:
                    content_type, filename, filepath = file_svc.get_file_content(UUID(aid))
                except (ValueError, Exception):
                    continue
                if content_type and content_type.startswith("image/"):
                    blocks.append(self._image_block(filepath, content_type))
                else:
                    blocks.append({"type": "file", "path": str(filepath), "filename": filename})

        # Extract text input
        if isinstance(request.input, str):
            blocks.append({"type": "text", "text": request.input})
        elif isinstance(request.input, list):
            for msg in reversed(request.input):
                if msg.role == Role.user and msg.content:
                    if isinstance(msg.content, str):
                        blocks.append({"type": "text", "text": msg.content})
                    elif isinstance(msg.content, list):
                        for item in msg.content:
                            if isinstance(item, Content):
                                if item.type == ContentType.text and item.text:
                                    blocks.append({"type": "text", "text": item.text})
                                elif item.type == ContentType.file and item.file_id:
                                    try:
                                        content_type, filename, filepath = FileService(self.session).get_file_content(UUID(item.file_id))
                                    except ValueError:
                                        raise HTTPException(status_code=404, detail=f"File {item.file_id!r} not found")
                                    if content_type and content_type.startswith("image/"):
                                        blocks.append(self._image_block(filepath, content_type))
                                    else:
                                        blocks.append({"type": "file", "path": str(filepath), "filename": filename})
                    break

        return blocks or [{"type": "text", "text": ""}]

    def _relink_attachments(self, client_session_id: str, conversation) -> None:
        """Repoint attachments uploaded against a client-side session id to
        the conversation that actually got created, so the Task Uploads
        rail (which queries by the live conversation id) still finds them."""
        from cowork.services.files import FileService, attachment_purpose

        try:
            project_name = conversation.project.name
        except Exception:
            return
        moved = FileService(self.session).relink_purpose(
            attachment_purpose(project_name, client_session_id),
            attachment_purpose(project_name, str(conversation.id)),
        )
        if moved:
            logger.info(
                "[responses] relinked %d attachment(s) from client session %r to conversation %s",
                moved, client_session_id, conversation.id,
            )

    def _resolve_project_id(self, request: ResponsesRequest) -> UUID:
        if request.project_id is not None:
            return request.project_id
        if request.project:
            try:
                return ProjectService(self.session).get_project_by_name(request.project).id
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=f"Project not found: {request.project}") from exc
        return GENERAL_PROJECT_ID

    @staticmethod
    def _image_block(filepath: Path, media_type: str) -> dict:
        data = filepath.read_bytes()
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64.standard_b64encode(data).decode("ascii"),
            },
        }

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


async def sse_from_buffer(buffer, from_seq: int = 0) -> AsyncGenerator[str, None]:
    """Serialize a turn buffer to the SSE wire, replaying from ``from_seq``
    then live-tailing. Used by both the initial POST /responses stream
    (from_seq=0) and reconnects via GET /responses/tail. The terminal record
    just ends the stream — the harness's own response.completed/failed frame
    was already written as a normal record."""
    async for rec in buffer.tail(from_seq):
        if rec.is_terminal:
            return
        sse = rec.data.get("sse")
        if sse:
            yield sse
