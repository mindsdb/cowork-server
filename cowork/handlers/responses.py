from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from fastapi import HTTPException
from sqlmodel import Session

from cowork.build_info import build_trace_metadata
from cowork.common.settings.app_settings import TurnQueueSettings
from cowork.common.settings.user_settings import get_user_settings, use_settings_scope
from cowork.db.session import get_open_session
from cowork.harnesses.base import get_harness
from cowork.streaming import new_buffer, registry
from cowork.turnqueue.producer import produce_remote_turn
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
    PROVIDER_OVERLOADED_CODE,
    auth_error_detail,
    friendly_turn_error,
    model_unavailable_info,
    provider_overloaded_info,
    response_failed_payload,
    response_failed_sse,
)
from cowork.db.scoped import ScopedSession, scope_from_principal
from cowork.principal import Principal, identity_trace_metadata
from cowork.services.conversations import ConversationService
from cowork.services.files import FileService
from cowork.services.memory import apply_turn_memory, build_turn_memory
from cowork.services.projects import GENERAL_PROJECT_ID, ProjectService
from cowork.services.skills import SkillService, build_turn_skills


import logging

logger = logging.getLogger(__name__)


class ResponsesHandler:
    def __init__(self, session: Session, principal: Principal | None = None) -> None:
        self.session = session
        self.principal = principal
        self.scope = scope_from_principal(principal)
        self.scoped = ScopedSession(session, self.scope)
        # Resolve settings once; reuse the harness name in handle()/the producer.
        self.harness_name = get_user_settings(self.scope).harness
        self.harness = get_harness(self.harness_name)
        self.last_conversation_id: str | None = None

    async def handle(self, request: ResponsesRequest) -> AsyncGenerator[str, None] | Response:
        logger.info("[responses] handle() called — conversation=%s, stream=%s", request.conversation, request.stream)

        # Identity + the running build into the run's trace metadata;
        # server-derived keys win. The build stamp (ENG-1279) is what lets a
        # metric be attributed to a release instead of to a date on which
        # several changes happened to ship together.
        trace_metadata = build_trace_metadata(identity_trace_metadata(self.principal, request.trace_metadata))

        conversation_service = ConversationService(self.scoped)

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
                        project_id=self._resolve_project_id(request),
                        conversation_id=conv_id,
                    )
            else:
                # Client sent a non-UUID id (e.g. the legacy timestamp
                # allocator, or a name-based format) — it can't become the
                # row id, so create a fresh conversation and re-link any
                # attachments uploaded against the client's id (ENG-264).
                conversation = conversation_service.create_conversation(
                    topic=self._prompt_text(harness_input)[:80],
                    project_id=self._resolve_project_id(request),
                )
                self._relink_attachments(request.conversation, conversation)
        else:
            conversation = conversation_service.create_conversation(
                topic=self._prompt_text(harness_input)[:80],
                project_id=self._resolve_project_id(request),
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
            # producer — only an explicit /cancel does.
            #
            # The user message is persisted (pending) as the producer's FIRST
            # action, not here (ENG-1231). registry.start() dedups a duplicate
            # start for an already-in-flight conversation and discards the second
            # producer coroutine unawaited — persisting here (before that dedup)
            # would commit a pending row whose producer never runs and never
            # finalizes, stranding it out of LLM history. Persisting inside the
            # producer ties the write to the one coroutine that actually runs, so
            # there is at most one pending row per conversation.
            buffer = new_buffer(str(conversation.id), turn_id)
            producer_coro = self._select_producer(
                conv_id=conversation.id,
                harness_input=harness_input,
                original_content=original_content,
                model=request.model,
                disabled=disabled,
                harness_name=self.harness_name,
                harness_id=getattr(self.harness, "id", None),
                buffer=buffer,
                trace_tags=request.trace_tags,
                trace_metadata=trace_metadata,
            )
            await registry.start(
                conversation_id=str(conversation.id),
                turn_id=turn_id,
                buffer=buffer,
                org_id=self.scoped.scope.org_id,
                user_id=self.scoped.scope.user_id,
                producer_coro=producer_coro,
            )
            return sse_from_buffer(buffer, 0)

        # Non-streaming (legacy/rare): run synchronously within the request.
        # The user message is persisted by _collect after the turn (deferred),
        # so the harness reads history WITHOUT the current turn — otherwise the
        # fresh-query history would replay it AND resend it as the live input.
        with use_settings_scope(self.scope):
            stream = self.harness.stream_response(
                conversation=conversation,
                input=harness_input,
                disabled_connections=disabled,
                trace_tags=request.trace_tags,
                trace_metadata=trace_metadata,
            )
            return await self._collect(stream, conversation.id, request.model, original_content)

    def _select_producer(
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
    ):
        """Choose the streaming producer coroutine.

        `COWORK_TURN_BACKEND=remote` (`TurnQueueSettings().backend`) routes
        the turn through the Redis-backed remote producer. Otherwise (the
        default, "inprocess"), this runs in-process: the `self._produce(...)`
        call below is unchanged from before this branch existed.
        """
        if TurnQueueSettings().backend == "remote":
            return self._produce_remote(
                conv_id=conv_id,
                input_text=self._prompt_text(harness_input),
                original_content=original_content,
                model=model,
                harness_id=harness_id,
                buffer=buffer,
            )
        return self._produce(
            conv_id=conv_id,
            harness_input=harness_input,
            original_content=original_content,
            model=model,
            disabled=disabled,
            harness_name=harness_name,
            harness_id=harness_id,
            buffer=buffer,
            trace_tags=trace_tags,
            trace_metadata=trace_metadata,
        )

    @staticmethod
    def _remote_memory(session: ScopedSession, conv_id: UUID) -> dict:
        """This org's memory slots for the pod. A read error degrades to a turn
        without memory rather than failing the turn."""
        try:
            conversation = ConversationService(session).get_conversation(conv_id)
            return build_turn_memory(session.scope, conversation.project.path)
        except Exception:
            logger.exception("[responses] failed to read memory for conversation %s", conv_id)
            return {}

    @staticmethod
    def _remote_skills(session: ScopedSession, conv_id: UUID) -> dict:
        """This org's skills for the pod, filtered to the conversation's project.
        A read error degrades to a turn without skills rather than failing the
        turn."""
        try:
            conversation = ConversationService(session).get_conversation(conv_id)
            return build_turn_skills(session.scope, conversation.project.path)
        except Exception:
            logger.exception("[responses] failed to read skills for conversation %s", conv_id)
            return {}

    @staticmethod
    def _persist_turn_memory(session: ScopedSession, conv_id: UUID, entries: list) -> None:
        """Apply what the pod asked to remember, re-anchoring the conversation
        first (like persist(): one deleted mid-turn must not write memory).

        Never fails the turn — a lost memory is recoverable, a lost reply isn't.
        """
        if not entries:
            return
        try:
            conversation = ConversationService(session).get_conversation(conv_id)
            applied = apply_turn_memory(session.scope, conversation.project.path, entries)
            logger.info("[responses] applied %d memory entr(ies) for conversation %s",
                        applied, conv_id)
        except Exception:
            logger.exception("[responses] failed to apply memory for conversation %s", conv_id)

    @staticmethod
    def _remote_history(session, conv_id) -> list[dict]:
        """Prior user/assistant messages in canonical order, as OpenAI-shaped
        dicts (mode="json": the payload gets json.dumps'd into the Redis job)."""
        ordered = ConversationService(session).get_ordered_messages(conv_id)
        return [
            m.to_openai_message().model_dump(mode="json")
            for m in ordered
            if m.role in {"user", "assistant"}
        ]

    async def _produce_remote(
        self,
        *,
        conv_id: UUID,
        input_text: str,
        original_content,
        model: str | None,
        harness_id: str | None,
        buffer,
    ) -> None:
        """Remote-backend counterpart of _produce: run the turn through the
        queue and persist user + assistant together on terminal (deferred, so
        _remote_history reads prior turns without the current input)."""
        producer_session = ScopedSession(get_open_session(), scope_from_principal(self.principal))
        collected_text: list[str] = []
        collected_events: list[dict] = []
        persisted = False
        pending_message_id: UUID | None = None

        def on_event(kind: str, data: dict) -> None:
            # Mirror the SSE frames the producer streams, so the client
            # rebuilds the same turn from the events log on reload.
            if kind == "turn_delta":
                text = data.get("text", "")
                collected_text.append(text)
                collected_events.append({"type": "response.output_text.delta", "delta": text})
            elif kind == "turn_memory":
                self._persist_turn_memory(producer_session, conv_id, data.get("entries") or [])
            elif kind == "turn_completed":
                collected_events.append({"type": "response.completed"})
            elif kind == "turn_failed":
                # Producer attaches the classified (code, message) it streamed.
                collected_events.append(response_failed_payload(
                    data.get("message") or GENERIC_TURN_ERROR_MESSAGE,
                    data.get("code") or GENERIC_TURN_ERROR_CODE,
                ))

        def persist() -> None:
            nonlocal persisted
            if persisted:
                return
            persisted = True
            try:
                # Re-anchor first: the conversation may be gone or out of scope.
                svc = ConversationService(producer_session)
                svc.get_conversation(conv_id)
                # The user message was already persisted (pending) at turn start
                # (ENG-1231). Clear the flag first — even if save_assistant_turn
                # early-returns on an empty turn — so the question rejoins replayed
                # history; then persist the assistant turn. Scope to THIS turn's
                # row so a completing turn can't absorb a pending row stranded by
                # an earlier crashed turn into history. If the pending persist
                # never succeeded (id unset), this turn owns no row — skip
                # finalize rather than fall back to clearing every pending row.
                if pending_message_id is not None:
                    svc.finalize_pending(conv_id, pending_message_id)
                svc.save_assistant_turn(
                    conv_id, "".join(collected_text), collected_events, harness=harness_id,
                )
            except Exception:
                logger.exception("[responses] failed to persist remote turn for conversation %s", conv_id)

        try:
            # Persist the user message (pending) as the first thing this producer
            # does (ENG-1231) — see the note in handle(). Committed here, before
            # streaming, so a refresh/reconnect mid-turn shows the question via
            # /items. _remote_history reads get_ordered_messages, which excludes
            # pending, so the current input isn't replayed into the remote job.
            pending_message_id = ConversationService(producer_session).save_user_message(
                conv_id, original_content, pending=True,
            ).id
            await produce_remote_turn(
                conversation_id=str(conv_id),
                org_id=self.scoped.scope.org_id,
                user_id=self.scoped.scope.user_id,
                input_text=input_text,
                model=model,
                buffer=buffer,
                # Producer session, NOT self.scoped: this coroutine is detached
                # and the request session may be closed by the time it runs.
                history=self._remote_history(producer_session, conv_id),
                harness_id=harness_id,
                memory=self._remote_memory(producer_session, conv_id),
                skills=self._remote_skills(producer_session, conv_id),
                on_event=on_event,
            )
            persist()
        except asyncio.CancelledError:
            # Partial text generated before cancellation is persisted.
            persist()
            await buffer.close("cancelled")
        except Exception:
            logger.exception("[responses] remote turn failed for conversation %s", conv_id)
            await buffer.append("sse", {"sse": response_failed_sse(
                GENERIC_TURN_ERROR_MESSAGE, GENERIC_TURN_ERROR_CODE)})
            persist()
            await buffer.close("error")
        finally:
            producer_session.close()

    async def _produce(self, **kwargs) -> None:
        # Detached task: bind the turn's org scope so every settings reader in the
        # harness/provider/publish subtree resolves this org's config.
        with use_settings_scope(scope_from_principal(self.principal)):
            await self._run_turn(**kwargs)

    async def _run_turn(
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

        Runs in its OWN DB session (it outlives the request). Persists the user
        message (pending) as its first action so a mid-turn refresh shows the
        question, reads history via get_ordered_messages (which excludes pending,
        so the current input isn't double-fed), and on terminal finalizes the
        pending flag + persists the assistant turn (ENG-1231). Never reaches the
        HTTP response — readers tail the buffer.
        """
        # Fresh session (outlives the request), scoped from the immutable
        # principal captured at handler construction — never request state.
        producer_session = ScopedSession(get_open_session(), scope_from_principal(self.principal))
        collected_text: list[str] = []
        collected_events: list[dict] = []
        turn_rows: list[dict] = []
        persisted = False
        # Send time captured before the turn
        sent_at = datetime.now(timezone.utc)
        pending_message_id: UUID | None = None

        def event_sink(event_type: str, data: dict) -> None:
            # Tool block-rows are for LLM-history persistence, not UI replay —
            # keep them out of the events log the client rebuilds from.
            if event_type == "response.turn_history":
                turn_rows[:] = data.get("rows") or []
                return
            collected_events.append(data)
            if event_type == "response.output_text.delta":
                collected_text.append(data.get("delta", ""))

        def persist() -> None:
            nonlocal persisted
            if persisted:
                return
            persisted = True
            try:
                # Re-anchor before ANY write: the conversation may be gone
                # (deleted mid-turn) or out of scope on this fresh session.
                svc = ConversationService(producer_session)
                svc.get_conversation(conv_id)
                # The user message was already persisted (pending) at turn start
                # (ENG-1231). Clear the flag first — even if save_assistant_turn
                # early-returns on an empty turn — so the question rejoins replayed
                # history; then persist the assistant turn. Scope to THIS turn's
                # row so a completing turn can't absorb a pending row stranded by
                # an earlier crashed turn into history. If the pending persist
                # never succeeded (id unset), this turn owns no row — skip
                # finalize rather than fall back to clearing every pending row.
                if pending_message_id is not None:
                    svc.finalize_pending(conv_id, pending_message_id)
                svc.save_assistant_turn(
                    conv_id, "".join(collected_text), collected_events, harness=harness_id,
                    tool_rows=turn_rows,
                )
            except Exception:
                logger.exception("[responses] failed to persist turn for conversation %s", conv_id)

        try:
            conv = ConversationService(producer_session).get_conversation(conv_id)
            # Persist the user message (pending) as the first thing this producer
            # does (ENG-1231) — see the note in handle(). The harness reads history
            # via get_ordered_messages, which excludes pending, so this write isn't
            # replayed into the turn as duplicate context.
            pending_message_id = ConversationService(producer_session).save_user_message(
                conv_id, original_content, created_at=sent_at, pending=True,
            ).id
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
            elif code == PROVIDER_OVERLOADED_CODE:
                # Transient-incident timeout (ENG-673): give the card the failing
                # model AND the active provider, and flag whether the user is
                # already routed through MindsHub. reconnectable=True → on managed
                # (all upstreams down; just Retry); False → BYOK/direct, so the
                # card can nudge toward MindsHub's cross-provider failover. Never
                # break the handler — fall back to the bare message on any error.
                overloaded_info = provider_overloaded_info(exc)
                failed_model = overloaded_info[1] if overloaded_info else ""
                extra = {"model": failed_model}
                try:
                    from cowork.common.settings.user_settings import Provider
                    s = get_user_settings()
                    # The nudge keys on WHICH provider overloaded. anton passes the
                    # actual failing model (planning OR coding); map it back to its
                    # provider so a coding-model incident on a DIFFERENT provider
                    # than planning isn't mislabeled — e.g. planning=MindsHub +
                    # coding=BYOK overloads must NOT read as reconnectable=True and
                    # suppress the failover nudge (Sam's review). Falls back to
                    # planning when the model is unknown or both roles share a
                    # provider (then the two agree anyway).
                    if (
                        failed_model
                        and failed_model == s.resolved_coding_model
                        and failed_model != s.resolved_planning_model
                    ):
                        provider = s.resolved_coding_provider
                    else:
                        provider = s.resolved_planning_provider
                    extra["provider_label"] = provider.label
                    extra["reconnectable"] = provider == Provider.MINDS_CLOUD
                except Exception:
                    logger.exception("[responses] could not resolve provider for overload error")
            failed = response_failed_payload(message, code, **extra)
            await buffer.append("sse", {"sse": response_failed_sse(message, code, **extra)})
            collected_events.append(failed)
            persist()
            await buffer.close("error")
        finally:
            producer_session.close()

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
        original_content,
    ) -> Response:
        collected_text: list[str] = []
        collected_events: list[dict] = []
        turn_rows: list[dict] = []
        # Send time captured before the turn
        sent_at = datetime.now(timezone.utc)

        def event_sink(event_type: str, data: dict) -> None:
            if event_type == "response.turn_history":
                turn_rows[:] = data.get("rows") or []
                return
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
        # Persist the user message now — after the harness has read history for
        # this turn — so it isn't replayed into the turn as duplicate context.
        user_message = ConversationService(self.scoped).save_user_message(
            conversation_id, original_content, created_at=sent_at,
        )
        self._save_assistant_turn(conversation_id, assistant_text, collected_events, turn_rows)

        return Response(
            status=ResponseStatus.completed,
            model=model,
            output=[self._build_output(str(user_message.id), assistant_text)],
        )

    def _save_assistant_turn(
        self,
        conversation_id: UUID,
        text: str,
        events: list[dict],
        tool_rows: list[dict] | None = None,
    ) -> None:
        harness_id = getattr(self.harness, 'id', None)
        ConversationService(self.scoped).save_assistant_turn(
            conversation_id, text, events, harness=harness_id, tool_rows=tool_rows,
        )

    def _build_harness_input(self, request: ResponsesRequest) -> list[dict]:
        blocks: list[dict] = []

        # Resolve attachment_ids to image/file blocks
        if request.attachment_ids:
            file_svc = FileService(self.scoped)
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
                                        content_type, filename, filepath = FileService(self.scoped).get_file_content(UUID(item.file_id))
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
        from cowork.services.files import attachment_purpose

        moved = FileService(self.scoped).relink_purpose(
            attachment_purpose(client_session_id),
            attachment_purpose(str(conversation.id)),
        )
        if moved:
            logger.info(
                "[responses] relinked %d attachment(s) from client session %r to conversation %s",
                moved, client_session_id, conversation.id,
            )

    def _resolve_project_id(self, request: ResponsesRequest) -> UUID:
        """Project for a conversation being CREATED this turn.

        Only called on the creation paths: an existing conversation already
        pins its project via conversation.project_id, and the client-held
        name it echoes can be stale after a project rename — resolving it
        eagerly used to 404 every later turn of the task (ENG-1028).
        """
        service = ProjectService(self.scoped)
        if request.project_id is not None:
            return request.project_id
        if request.project:
            try:
                return service.get_project_by_name(request.project).id
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=f"Project not found: {request.project}") from exc
        # Bootstrap site: a turn can be the org's first request — adopt the
        # seeded GENERAL project before defaulting to it.
        service.ensure_general_for_scope()
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
