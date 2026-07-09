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
    MODEL_ACCESS_DENIED_CODE,
    MODEL_DISABLED_CODE,
    auth_error_detail,
    friendly_turn_error,
    model_unavailable_info,
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
                    trace_tags=request.trace_tags,
                    trace_metadata=request.trace_metadata,
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

        stream = self.harness.stream_response(
            conversation=conversation,
            input=harness_input,
            disabled_connections=disabled,
            trace_tags=request.trace_tags,
            trace_metadata=request.trace_metadata,
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
        trace_tags: list[str] | None = None,
        trace_metadata: dict[str, str] | None = None,
    ) -> None:
        """Detached producer: run the turn and write events to the buffer.

        Runs in its OWN DB session (it outlives the request). Persistence is
        deferred to the end so the harness reads history WITHOUT the current
        user message; on terminal we persist user + assistant together.
        Never reaches the HTTP response — readers tail the buffer.
        """
        own = get_open_session()
        collected_text: list[str] = []
        collected_events: list[dict] = []
        persisted = False

        def event_sink(event_type: str, data: dict) -> None:
            collected_events.append(data)
            if event_type == "response.output_text.delta":
                collected_text.append(data.get("delta", ""))

        def persist() -> None:
            nonlocal persisted
            if persisted:
                return
            persisted = True
            try:
                own.add(DBMessage(conversation_id=conv_id, role=Role.user, content=original_content))
                own.commit()
                ConversationService(own).save_assistant_turn(
                    conv_id, "".join(collected_text), collected_events, harness=harness_id,
                )
            except Exception:
                logger.exception("[responses] failed to persist turn for conversation %s", conv_id)

        try:
            conv = ConversationService(own).get_conversation(conv_id)
            harness = get_harness(harness_name)
            stream = harness.stream_response(
                conversation=conv, input=harness_input, disabled_connections=disabled,
                trace_tags=trace_tags, trace_metadata=trace_metadata,
            )
            event_count = 0
            async for sse_string in harness.formatter(stream, model, event_sink):
                event_count += 1
                sse_string = self._inject_created(sse_string, conv_id, harness_id)
                await buffer.append("sse", {"sse": sse_string})
            logger.info("[responses] turn %s finished — %d events", conv_id, event_count)
            persist()
            await buffer.close("completed")
        except asyncio.CancelledError:
            # Nothing special is emitted on cancellation.
            # The partial text and evennts generated before cancellation are persisted.
            persist()
            await buffer.close("cancelled")
            return
        except Exception as exc:
            # Resolve the model-403 info once and hand it to friendly_turn_error
            # so it isn't computed twice on this path (reused by the extras below).
            model_info = model_unavailable_info(exc)
            friendly = friendly_turn_error(exc, model_info=model_info)
            if friendly is not None:
                code, message = friendly
                logger.info("[responses] user-facing turn error: %s", exc)
            else:
                code, message = GENERIC_TURN_ERROR_CODE, GENERIC_TURN_ERROR_MESSAGE
                logger.exception("[responses] turn failed for conversation %s", conv_id)
            # For an auth failure, tell the client which provider failed so it
            # offers the right action: "Reconnect" only for MindsHub (we can
            # re-provision the key in place), "Open Settings" for a BYOK key the
            # user owns. Without this the renderer would always say "Reconnect
            # MindsHub" — wrong for BYOK users.
            extra: dict = {}
            if code == AUTH_ERROR_CODE:
                # Resolving the provider must never break the error handler —
                # if it raises we just fall back to the generic auth message
                # (no reconnectable flag), so the stream still closes cleanly.
                try:
                    from cowork.common.settings.user_settings import Provider
                    provider = get_user_settings().resolved_planning_provider
                    reconnectable = provider == Provider.MINDS_CLOUD
                    message = auth_error_detail(provider.label, reconnectable)
                    extra = {"reconnectable": reconnectable, "provider_label": provider.label}
                except Exception:
                    logger.exception("[responses] could not resolve provider for auth error")
            elif code in (MODEL_ACCESS_DENIED_CODE, MODEL_DISABLED_CODE):
                # Model-403: tell the client WHICH model was rejected so the card
                # can name it ("Sonnet isn't included in your plan"). No
                # provider_label — the ModelUnavailableCard doesn't render it, and
                # resolved_planning_provider would name the wrong provider when
                # the *coding* model was the one rejected.
                extra = {"model": model_info[1] if model_info else ""}
            failed = response_failed_payload(message, code, **extra)
            await buffer.append("sse", {"sse": response_failed_sse(message, code, **extra)})
            collected_events.append(failed)
            persist()
            await buffer.close("error")
        finally:
            own.close()

    @staticmethod
    def _inject_created(sse_string: str, conversation_id: UUID, harness_id: str | None) -> str:
        """Inject conversation_id + harness into the response.created event so
        the client learns the canonical id and which agent generated this."""
        if "response.created" in sse_string and "conversation_id" not in sse_string:
            try:
                lines = sse_string.strip().split("\n")
                data_line = next(l for l in lines if l.startswith("data:"))
                payload = json.loads(data_line[5:])
                payload["conversation_id"] = str(conversation_id)
                if harness_id:
                    payload["harness"] = harness_id
                return f"event: response.created\ndata: {json.dumps(payload)}\n\n"
            except Exception:
                pass
        return sse_string

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
