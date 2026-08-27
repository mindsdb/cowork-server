from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import HTTPException
from sqlmodel import Session

from cowork.build_info import build_trace_metadata
from cowork.common.chat_session import in_process_agent_allowed
from cowork.common.settings.app_settings import MINDS_FREE_MODEL, TurnQueueSettings
from cowork.common.settings.user_settings import (
    Provider,
    get_user_settings,
    provider_api_key,
    use_settings_scope,
)
from cowork.db.session import get_open_session
from cowork.harnesses.base import available_harness_ids, get_harness
from cowork.handlers.response_routing import (
    DELEGATED_AGENTIC,
    DIRECT_CONTEXT,
    RouteDecision,
    RouterBinding,
    decide_route,
    ineligible_reason,
)
from cowork.harnesses.anton_harness.stream_formatter import SkillCreated, format_responses_stream
from cowork.streaming import TurnLifecycle, new_buffer, registry, sse_frame
from cowork.turnqueue.producer import step_stream_events, stream_remote_replies
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
from cowork.handlers._turn_history import sanitize_turn_history_rows
from cowork.handlers.turn_errors import (
    AUTH_ERROR_CODE,
    CONTENT_RECOVERY_CODE,
    GENERIC_TURN_ERROR_CODE,
    GENERIC_TURN_ERROR_MESSAGE,
    MODEL_UNAVAILABLE_CODES,
    ALLOWANCE_EXHAUSTED_CODE,
    PROVIDER_OVERLOADED_CODE,
    RATE_LIMITED_CODE,
    auth_error_detail,
    friendly_turn_error,
    model_unavailable_info,
    provider_overloaded_info,
    allowance_reset_at,
    response_failed_payload,
    retry_after_seconds,
    retry_at_instant,
    response_failed_sse,
)
from cowork.db.scoped import ScopedSession, scope_from_principal
from cowork.principal import Principal, identity_trace_metadata
from cowork.services.conversations import ConversationService
from cowork.services.files import FileService
from cowork.services.memory import apply_turn_memory, build_turn_memory
from cowork.services.projects import ProjectService
from cowork.services.skills import SkillService
from cowork.services.task_objects import remote_skill_draft_result


import logging

logger = logging.getLogger(__name__)


class _RemoteTurnFailed(Exception):
    """Terminal turn_failed reply; payload rides the enclosing scope."""

# Statuses of the `response.ask_user_answered` event, as the harness emits them
# (cowork/harnesses/anton_harness/stream_formatter.py). "cancelled" is the one
# this module synthesizes when a turn is stopped while a question is on screen.
ASK_USER_EVENT = "response.ask_user"
ASK_USER_ANSWERED_EVENT = "response.ask_user_answered"


def cancelled_ask_user_retirements(events: list[dict]) -> list[dict]:
    """Retirement events for every question in *events* that nothing retires.

    Why the server has to do this: anton's ``elicit()`` emits
    ``StreamAskUserAnswered`` from an ``except Exception`` branch, and
    ``CancelledError`` is a ``BaseException`` — so pressing Stop while a
    question is open skips the retirement. Emitting it from anton on that path
    would not help either: once the turn is cancelled ``session.emit`` only
    puts the event on a queue nobody drains any more. The server, by contrast,
    knows the turn is over and knows exactly which questions it published.

    Without this, ``_run_turn``'s cancellation path persists a ``response.ask_user``
    with nothing that retires it, i.e. an internally inconsistent event log:
    every consumer that replays it has to infer the missing half. Note the
    shape carries no ``sequence_number`` — there is no live counter left to
    draw one from, and the client keys purely on ``type`` + ``question_id``
    (same as the synthesized ``response.failed`` payload next to it).
    """
    retired = {
        e.get("question_id")
        for e in events
        if e.get("type") == ASK_USER_ANSWERED_EVENT
    }
    synthesized: list[dict] = []
    for event in events:
        if event.get("type") != ASK_USER_EVENT:
            continue
        question_id = event.get("question_id")
        if question_id in retired:
            continue
        # Guard against a duplicated publish of the same id producing two
        # retirements for one card.
        retired.add(question_id)
        synthesized.append({
            "type": ASK_USER_ANSWERED_EVENT,
            "question_id": question_id,
            "status": "cancelled",
            "values": [],
            "text": "",
        })
    return synthesized


async def _seal_unterminated_buffer(buffer, lifecycle: "TurnLifecycle", conv_id) -> None:
    """Guarantee a terminal record so a producer that ended WITHOUT closing its
    buffer can't leave the client's tail (and its shared stream slot) hanging
    forever.

    Every producer branch closes the buffer on its own path, but the terminal is
    not guaranteed: an exception escaping the ``except Exception`` handler (e.g.
    the error-classification helpers raising), or a ``BaseException`` that
    matches no ``except`` clause, skips ``buffer.close()`` — the buffer stays
    open with no terminal, the desktop's in-process tail blocks forever, and
    every later message strands at "Queued". Unlike the duration bound, this
    covers a turn that FAILS FAST, well before any timeout.

    ``close()`` is idempotent, so this is a no-op on every normal path. The
    ``discarded`` path is skipped: its buffer file was already deleted and
    closing would recreate it. Both awaits are guarded so a seal failure can't
    mask the original exception propagating out of ``finally``. ``is_closed`` is
    abstract on ``StreamBuffer`` but a minimal test double may lack it — treat an
    absent flag as already-terminated so a stub can't trigger a spurious second
    terminal.
    """
    if lifecycle.discarded or getattr(buffer, "is_closed", True):
        return
    logger.error(
        "[responses] turn for conversation %s ended without a terminal record; "
        "sealing the buffer so the client releases its stream slot", conv_id,
    )
    try:
        await buffer.append("sse", {"sse": response_failed_sse(
            GENERIC_TURN_ERROR_MESSAGE, GENERIC_TURN_ERROR_CODE)})
    except Exception:
        logger.exception("[responses] could not emit terminal error frame while sealing")
    try:
        await buffer.close("error")
    except Exception:
        logger.exception("[responses] could not seal unterminated turn buffer")


class ResponsesHandler:
    def __init__(self, session: Session, principal: Principal | None = None) -> None:
        self.session = session
        self.principal = principal
        self.scope = scope_from_principal(principal)
        self.scoped = ScopedSession(session, self.scope)
        # Resolve the selected harness name now, but initialize Anton lazily only
        # after Cowork delegates a turn. Direct context responses must never build
        # the Anton harness.
        self.harness_name = get_user_settings(self.scope).harness
        self.harness = None
        self.last_conversation_id: str | None = None

    def _get_harness(self):
        if self.harness is None:
            self.harness = get_harness(self.harness_name)
        return self.harness

    async def handle(self, request: ResponsesRequest) -> AsyncGenerator[str, None] | Response:
        logger.info("[responses] handle() called — conversation=%s, stream=%s", request.conversation, request.stream)

        # A per-conversation harness pick (Coding Mode's composer pill)
        # overrides the account default for THIS call only — mirrors the
        # per-conversation model override below. Ignored (not raised) when it
        # doesn't name a currently-registered/available harness: a stale
        # client cache (e.g. Hermes got uninstalled since the picker last
        # loaded) must never fail the turn, it just falls back to the
        # account default. self.harness stays None either way — still lazy,
        # only self.harness_name (which harness _get_harness() will build)
        # changes here.
        if request.harness and request.harness in available_harness_ids():
            self.harness_name = request.harness

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
                        harness=self.harness_name,
                        model=request.model,
                    )
            else:
                # Client sent a non-UUID id (e.g. the legacy timestamp
                # allocator, or a name-based format) — it can't become the
                # row id, so create a fresh conversation and re-link any
                # attachments uploaded against the client's id (ENG-264).
                conversation = conversation_service.create_conversation(
                    topic=self._prompt_text(harness_input)[:80],
                    project_id=self._resolve_project_id(request),
                    harness=self.harness_name,
                    model=request.model,
                )
                self._relink_attachments(request.conversation, conversation)
        else:
            conversation = conversation_service.create_conversation(
                topic=self._prompt_text(harness_input)[:80],
                project_id=self._resolve_project_id(request),
                harness=self.harness_name,
                model=request.model,
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
        route, turn_llm = await self._route_request(
            conversation_id=conversation.id,
            harness_input=harness_input,
            has_attachments=bool(request.attachment_ids),
            has_disabled_connections=bool(disabled),
        )
        trace_metadata = {
            **trace_metadata,
            "response_route": route.route,
            "response_route_reason": route.reason,
            **({"response_router_provider": route.provider} if route.provider else {}),
            **({"response_router_model": route.model} if route.model else {}),
            **({"response_route_fallback": "true"} if route.fallback else {}),
        }
        logger.info(
            "[responses] route=%s reason=%s fallback=%s provider=%s model=%s conversation=%s",
            route.route, route.reason, route.fallback, route.provider, route.model, conversation.id,
        )

        if route.route == DIRECT_CONTEXT:
            return await self._handle_direct_response(
                request=request,
                conversation_id=conversation.id,
                turn_id=turn_id,
                original_content=original_content,
                route=route,
            )

        harness = self._get_harness()

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
            # Created here, before the coroutine, and handed to BOTH: it is the
            # only channel by which a turn delete can tell this producer that
            # the history it is writing into no longer exists (the handle it
            # will be registered under does not exist yet).
            lifecycle = TurnLifecycle()
            producer_coro = self._select_producer(
                lifecycle=lifecycle,
                conv_id=conversation.id,
                harness_input=harness_input,
                original_content=original_content,
                model=request.model,
                disabled=disabled,
                harness_name=self.harness_name,
                harness_id=getattr(harness, "id", None),
                buffer=buffer,
                turn_id=turn_id,
                trace_tags=request.trace_tags,
                trace_metadata=trace_metadata,
                turn_llm=turn_llm,
            )
            await registry.start(
                conversation_id=str(conversation.id),
                turn_id=turn_id,
                buffer=buffer,
                org_id=self.scoped.scope.org_id,
                user_id=self.scoped.scope.user_id,
                producer_coro=producer_coro,
                lifecycle=lifecycle,
            )
            return sse_from_buffer(buffer, 0)

        # Non-streaming (legacy/rare): run synchronously within the request.
        # There is nowhere to run it in org mode. Only the streaming branch
        # above has a remote producer (_select_producer dispatches the turn to
        # a worker); this branch drives the harness in this process, and
        # AntonHarness.stream_response refuses in org mode because doing so
        # would execute agent-written code here. Without this check that
        # refusal surfaces as an unhandled RuntimeError from _collect and the
        # client sees an opaque 500. 501 with a concrete instruction instead:
        # this deployment really does not implement a non-streaming turn, which
        # is a statement about what the server can do. The org-mode tenancy
        # guards answer 403 because they refuse a caller rather than admit a
        # missing capability. `stream` defaults to False in ResponsesRequest, so
        # a client can land here by simply omitting the field.
        if not in_process_agent_allowed():
            raise HTTPException(
                status_code=501,
                detail=(
                    "This deployment only serves streaming turns. "
                    'Retry the request with "stream": true.'
                ),
            )
        # The user message is persisted by _collect after the turn (deferred),
        # so the harness reads history WITHOUT the current turn — otherwise the
        # fresh-query history would replay it AND resend it as the live input.
        with use_settings_scope(self.scope):
            stream = harness.stream_response(
                conversation=conversation,
                input=harness_input,
                model=request.model,
                disabled_connections=disabled,
                trace_tags=request.trace_tags,
                trace_metadata=trace_metadata,
            )
            return await self._collect(stream, conversation.id, request.model, original_content)

    async def _route_request(
        self,
        *,
        conversation_id: UUID,
        harness_input: list[dict],
        has_attachments: bool,
        has_disabled_connections: bool,
    ) -> tuple[RouteDecision, dict | None]:
        """Run Cowork's narrow pre-Anton gate with only safe text context.

        The composer's per-conversation model pick (`request.model`) is
        deliberately not passed down: it drives Anton's turn, not the gate
        (see `UserSettings.resolved_gate_model`).

        Returns the decision plus pre-minted turn credentials
        (`{"correlation_id", "llm"}`) for a delegated remote turn to reuse."""
        has_non_text_input = any(block.get("type") != "text" for block in harness_input)
        # Shape checks first: ineligible turns skip the history query.
        reason = ineligible_reason(
            has_non_text_input=has_non_text_input,
            has_attachments=has_attachments,
            has_disabled_connections=has_disabled_connections,
        )
        if reason:
            return RouteDecision(route=DELEGATED_AGENTIC, reason=reason), None
        try:
            history = [
                message.to_openai_message().model_dump()
                for message in ConversationService(self.scoped).get_ordered_messages(conversation_id)
                if message.role in {"user", "assistant"}
            ]
            history.append({"role": "user", "content": self._prompt_text(harness_input)})
            # The gate resolves the router role + key ambiently; bind the org scope.
            with use_settings_scope(self.scope):
                binding, turn_llm = await self._router_binding()
                decision = await decide_route(
                    history=history,
                    has_non_text_input=has_non_text_input,
                    has_attachments=has_attachments,
                    has_disabled_connections=has_disabled_connections,
                    binding=binding,
                )
            return decision, turn_llm
        except Exception:
            # Any gate-path failure (history query, mint, settings) fails open.
            logger.exception("[responses] routing gate failed — delegating")
            return RouteDecision(
                route=DELEGATED_AGENTIC, reason="router_unavailable", fallback=True
            ), None

    async def _router_binding(self) -> tuple[RouterBinding | None, dict | None]:
        """Hosted orgs keep no stored Minds key (remote turns mint one), so the
        gate mints its own per-turn key here and hands it back for the
        delegated turn to reuse. Everywhere else the stored settings apply."""
        if TurnQueueSettings().backend != "remote":
            return None, None
        settings = get_user_settings(self.scope)
        if (settings.resolved_router_provider is not Provider.MINDS_CLOUD
                or provider_api_key(settings, Provider.MINDS_CLOUD) is not None):
            return None, None
        from anton.core.llm.openai import OpenAIProvider
        from cowork.turnqueue.producer import _mint_llm_block

        corr = str(uuid4())
        block = await _mint_llm_block(
            org_id=self.scoped.scope.org_id,
            user_id=self.scoped.scope.user_id,
            correlation_id=corr,
            settings=TurnQueueSettings(),
        )
        provider = OpenAIProvider(
            api_key=block["api_key"],
            base_url=block["base_url"],
            flavor=OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH,
        )
        binding = RouterBinding(
            provider=provider,
            model=settings.resolved_gate_model or MINDS_FREE_MODEL,
            label=Provider.MINDS_CLOUD.value,
        )
        return binding, {"correlation_id": corr, "llm": block}

    async def _handle_direct_response(
        self,
        *,
        request: ResponsesRequest,
        conversation_id: UUID,
        turn_id: int,
        original_content,
        route: RouteDecision,
    ) -> AsyncGenerator[str, None] | Response:
        """Return the router model's direct answer without initializing Anton."""
        if not request.stream:
            user_message = ConversationService(self.scoped).save_user_message(
                conversation_id, original_content,
            )
            events = [{
                "type": "response.output_text.delta",
                "delta": route.text,
                "response_route": route.route,
                "response_route_reason": route.reason,
            }, {"type": "response.completed"}]
            ConversationService(self.scoped).save_assistant_turn(
                conversation_id, route.text, events, harness="cowork-direct",
            )
            return Response(
                status=ResponseStatus.completed,
                model=route.model,
                output=[self._build_output(str(user_message.id), route.text)],
            )

        # turn_id comes from handle(): same numbering as the delegated path.
        buffer = new_buffer(str(conversation_id), turn_id)
        lifecycle = TurnLifecycle()
        await registry.start(
            conversation_id=str(conversation_id),
            turn_id=turn_id,
            buffer=buffer,
            org_id=self.scoped.scope.org_id,
            user_id=self.scoped.scope.user_id,
            producer_coro=self._produce_direct(
                lifecycle=lifecycle,
                conv_id=conversation_id,
                original_content=original_content,
                route=route,
                buffer=buffer,
            ),
            lifecycle=lifecycle,
        )
        return sse_from_buffer(buffer, 0)

    async def _produce_direct(
        self,
        *,
        lifecycle: TurnLifecycle,
        conv_id: UUID,
        original_content,
        route: RouteDecision,
        buffer,
    ) -> None:
        """Persist and emit a direct answer using the normal detached lifecycle.

        The full answer exists up front, so persistence happens before any
        frame is emitted — the client can never see a completed turn the DB
        does not have, and no pending row is needed."""
        producer_session = None
        try:
            producer_session = ScopedSession(get_open_session(), scope_from_principal(self.principal))
            item_id = f"msg-{conv_id.hex[:12]}"
            delta = {
                "type": "response.output_text.delta",
                "sequence_number": 2,
                "item_id": item_id,
                "delta": route.text,
                "response_route": route.route,
                "response_route_reason": route.reason,
            }
            svc = ConversationService(producer_session)
            svc.save_user_message(conv_id, original_content)
            svc.save_assistant_turn(
                conv_id, route.text, [delta, {"type": "response.completed"}],
                harness="cowork-direct",
            )

            response = Response(status=ResponseStatus.created, model=route.model)
            # conversation_id/harness sit at the event root, like both
            # delegated paths — the GUI reads them there.
            await buffer.append("sse", {"sse": sse_frame("response.created", {
                "type": "response.created",
                "sequence_number": 1,
                "conversation_id": str(conv_id),
                "harness": "cowork-direct",
                "response": response.model_dump(),
            })})
            await buffer.append("sse", {"sse": sse_frame("response.output_text.delta", delta)})
            completed_response = Response(
                id=response.id,
                created_at=response.created_at,
                status=ResponseStatus.completed,
                model=route.model,
                output=[self._build_output(item_id, route.text)],
            ).model_dump()
            await buffer.append("sse", {"sse": sse_frame("response.completed", {
                "type": "response.completed", "sequence_number": 3, "response": completed_response,
            })})
            await buffer.close("completed")
        except asyncio.CancelledError:
            if lifecycle.discarded:
                # Same reasoning as _produce_remote's discarded branch.
                logger.info("[responses] discarded direct turn %s — not persisting", conv_id)
                return
            await buffer.close("cancelled")
        except Exception:
            logger.exception("[responses] direct turn failed for conversation %s", conv_id)
            await buffer.append("sse", {"sse": response_failed_sse(GENERIC_TURN_ERROR_MESSAGE, GENERIC_TURN_ERROR_CODE)})
            await buffer.close("error")
        finally:
            await _seal_unterminated_buffer(buffer, lifecycle, conv_id)
            if producer_session is not None:
                producer_session.close()

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
        turn_id: int = 0,
        trace_tags: list[str] | None = None,
        trace_metadata: dict[str, str] | None = None,
        lifecycle: TurnLifecycle | None = None,
        turn_llm: dict | None = None,
    ):
        """Choose the streaming producer coroutine.

        `COWORK_TURN_BACKEND=remote` (`TurnQueueSettings().backend`) routes
        the turn through the Redis-backed remote producer. Otherwise (the
        default, "inprocess"), this runs in-process: the `self._produce(...)`
        call below is unchanged from before this branch existed.

        `lifecycle` is shared with the RunHandle so a turn delete can stop
        either producer from persisting into truncated history; it defaults to
        a fresh one so a directly-called producer (tests) still has a flag to
        read.
        """
        lifecycle = lifecycle if lifecycle is not None else TurnLifecycle()
        if TurnQueueSettings().backend == "remote":
            return self._produce_remote(
                lifecycle=lifecycle,
                conv_id=conv_id,
                input_text=self._prompt_text(harness_input),
                original_content=original_content,
                model=model,
                harness_id=harness_id,
                buffer=buffer,
                turn_id=turn_id,
                turn_llm=turn_llm,
            )
        return self._produce(
            lifecycle=lifecycle,
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
    def _stage_remote_workspace_files(session: ScopedSession, conv_id: UUID) -> None:
        """Stage the project-level files the pod can't otherwise see — the
        conversation's attachments and the project's anton.md instructions —
        into the conversation workspace on the shared mount, and seed this
        org's skill store with the packaged builtins if it hasn't been yet.

        The seeding call belongs here, not just behind ``GET /skills``: the pod
        reads skills straight off the shared mount (no payload), so this is the
        only place that runs on every remote turn and can catch an org that
        chats before it ever opens the skills menu. Never fails the turn: a
        staging error degrades to a turn without the missing piece."""
        try:
            from cowork.services.files import stage_project_instructions

            conversation = ConversationService(session).get_conversation(conv_id)
            project_path = conversation.project.path
            FileService(session).stage_conversation_attachments(conv_id, project_path)
            stage_project_instructions(project_path, conv_id)
            SkillService(session.scope).ensure_builtin_skills()
        except Exception:
            logger.exception("[responses] failed to stage workspace files for conversation %s", conv_id)

    @staticmethod
    def _remote_workspace(session: ScopedSession, conv_id: UUID) -> dict:
        """The conversation's project as a path relative to the org root.

        Absolute paths must not cross the wire: cowork-server sees the shared
        tree at ``<root>/<org_id>`` and the pod mounts its own org's access
        point at ``<root>``, so an absolute path built here names nothing
        inside the pod. Both sides join their own root to this.

        A lookup failure degrades to the org's default project rather than
        failing the turn, matching how memory and skills used to degrade.
        """
        from pathlib import Path

        from cowork.common.settings.app_settings import get_app_settings
        from cowork.db.scoped import scoped_storage_root

        try:
            conversation = ConversationService(session).get_conversation(conv_id)
            org_root = scoped_storage_root(
                Path(get_app_settings().project.root_dir), session.scope, store="projects"
            ).parent
            rel = Path(conversation.project.path).relative_to(org_root).as_posix()
            return {"project_id": str(conversation.project.id), "workspace_rel_path": rel}
        except Exception:
            logger.exception("[responses] failed to resolve workspace for conversation %s", conv_id)
            return {}

    @staticmethod
    def _remote_artifacts_context(session: ScopedSession, conv_id: UUID):
        """`(conversation, artifacts_base, project_id, project_name)` for the
        remote turn's end-of-turn artifact bookkeeping, or None if unavailable.

        The pod writes artifacts into `<project>/.anton/artifacts/` on the shared
        mount, so cowork-server reads the same directory the worker just wrote —
        this is the whole reason the in-process flow transplants onto the remote
        path unchanged. `conversation` comes back attached to `session` because
        `index_turn_artifacts` recovers the tenant scope from the session the row
        is bound to; the ids and the project name are read here, while it is
        unambiguously attached, rather than after the turn.

        None on any failure: no artifact card and no autopublish is a recoverable
        outcome (the next turn in this project reconciles), a failed turn is not.
        """
        from cowork.services.artifact_roots import conversation_artifacts_base

        try:
            conversation = ConversationService(session).get_conversation(conv_id)
            # Conversation-scoped in org mode: the pod's workspace is
            # <project>/conversations/<id>, not the project, so that is where the
            # worker's artifacts land. Resolved through artifact_roots so this and
            # the artifacts list agree on the layout.
            artifacts_base = conversation_artifacts_base(conversation.project.path, conv_id)
            return (
                conversation,
                artifacts_base,
                str(conversation.project_id) if conversation.project_id else None,
                conversation.project.name,
            )
        except Exception:
            logger.exception(
                "[responses] failed to resolve artifacts context for conversation %s", conv_id)
            return None

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
        turn_id: int = 0,
        lifecycle: TurnLifecycle | None = None,
        turn_llm: dict | None = None,
    ) -> None:
        """Remote-backend counterpart of _produce: pipe the turn's replies
        through the same SSE formatter as the in-process path (full step /
        thinking parity, live and in the persisted events log) and persist
        user + assistant together on terminal (deferred, so _remote_history
        reads prior turns without the current input)."""
        lifecycle = lifecycle if lifecycle is not None else TurnLifecycle()
        producer_session = ScopedSession(get_open_session(), scope_from_principal(self.principal))
        collected_text: list[str] = []
        collected_events: list[dict] = []
        # This turn's tool block-rows, for LLM-history persistence only. Kept
        # out of collected_events: the client rebuilds its UI from those, and
        # tool rows are hidden from the UI (mirrors _run_turn's event_sink).
        turn_rows: list[dict] = []
        persisted = False
        pending_message_id: UUID | None = None
        failure: dict = {}

        def event_sink(event_type: str, data: dict) -> None:
            # Same event log the in-process path records, so the client
            # rebuilds the thinking block + steps identically on reload.
            # at_ms is stamped at receipt (the pod sends no timestamps), so
            # replayed durations are approximate under consumer lag.
            collected_events.append(data)
            if event_type == "response.output_text.delta":
                collected_text.append(data.get("delta", ""))

        # Stage attachments + project instructions into the workspace before the
        # pod runs, so it can read them off the shared mount (no other channel).
        # Off the event loop: the copies are blocking fs I/O (multi-MB uploads,
        # EFS latency) and would otherwise stall every other SSE stream on this
        # worker. Safe to share the producer session with the thread — nothing
        # else touches it until the reply stream below starts.
        await asyncio.to_thread(self._stage_remote_workspace_files, producer_session, conv_id)

        async def replies_as_stream_events():
            from anton.core.llm.provider import StreamTaskProgress, StreamTextDelta
            from cowork.harnesses.anton_harness.stream_formatter import ArtifactCreated
            from cowork.services.task_objects import (
                index_turn_artifacts,
                publish_and_card_turn_artifacts,
                snapshot_artifact_state,
            )

            # The worker writes artifacts into the shared tree while this turn
            # runs, so cowork-server does the same before/after diff it does for
            # an in-process turn. Snapshotting here rather than in the caller is
            # what makes it a genuine "before": stream_remote_replies below only
            # enqueues the job once this generator is first iterated.
            artifacts = self._remote_artifacts_context(producer_session, conv_id)
            before_slugs, before_mtimes = (
                snapshot_artifact_state(artifacts[1]) if artifacts else (set(), {})
            )
            new_slugs: list[str] = []
            touched_slugs: set[str] = set()
            turn_scope = None

            try:
                async for kind, data in stream_remote_replies(
                    conversation_id=str(conv_id),
                    org_id=self.scoped.scope.org_id,
                    user_id=self.scoped.scope.user_id,
                    input_text=input_text,
                    model=model,
                    turn_id=turn_id,
                    # Producer session, NOT self.scoped: this coroutine is detached
                    # and the request session may be closed by the time it runs.
                    history=self._remote_history(producer_session, conv_id),
                    # Skills and memory are NOT sent: the pod reads them off the
                    # shared mount. Only the org-relative project path travels.
                    **self._remote_workspace(producer_session, conv_id),
                    correlation_id=(turn_llm or {}).get("correlation_id"),
                    llm=(turn_llm or {}).get("llm"),
                ):
                    if kind == "turn_delta":
                        yield StreamTextDelta(text=data.get("text", ""))
                    elif kind == "turn_step":
                        for event in step_stream_events(data):
                            yield event
                    elif kind == "turn_memory":
                        self._persist_turn_memory(producer_session, conv_id, data.get("entries") or [])
                    elif kind == "turn_history":
                        # Slice-assign, not extend: the pod emits one frame per
                        # turn, so a repeated frame must replace the rows rather
                        # than double them. Sanitized at the boundary — the pod
                        # is semi-trusted and these rows reach both the DB and
                        # every later turn's LLM context.
                        turn_rows[:] = sanitize_turn_history_rows(data.get("rows"))
                    elif kind == "turn_skill":
                        # Not persisted like memory: a draft is the user's decision.
                        # Yielding SkillCreated puts it through the same formatter the
                        # in-process path uses, so the card renders — and replays off
                        # the events log — identically to a desktop one. A rejected
                        # draft, or a sibling file quietly excluded from an otherwise
                        # saved one, also gets a StreamTaskProgress notice — the
                        # generic "thought_progress" role already rendered inline,
                        # so a loss is visible in the turn instead of only in the
                        # server log.
                        for entry in data.get("entries") or []:
                            payload, reasons = remote_skill_draft_result(entry)
                            if payload is not None:
                                yield SkillCreated(payload)
                            for reason in reasons:
                                yield StreamTaskProgress(phase="skill_draft_dropped", message=reason)
                    elif kind == "turn_completed":
                        # `break`, not `return`: the publish/card block below the
                        # try must still run on a clean finish.
                        break
                    elif kind == "turn_failed":
                        failure.update(data)
                        raise _RemoteTurnFailed()
            finally:
                # Mirrors the in-process harness: indexing runs on EVERY exit so
                # an artifact the worker wrote is recorded even when the turn
                # failed or was stopped, and it is synchronous because an await
                # in a generator's finally is skipped on cancellation.
                if artifacts is not None:
                    new_slugs, touched_slugs, turn_scope = index_turn_artifacts(
                        artifacts[0], conv_id, artifacts[2], artifacts[1],
                        before_slugs, before_mtimes,
                    )

            # Clean completion only — a raise inside the try skips this, matching
            # the in-process path where Stop/error produce no cards and the next
            # turn in the project heals the publish.
            if artifacts is not None:
                for card in await publish_and_card_turn_artifacts(
                    artifacts[1],
                    new_slugs=new_slugs,
                    touched_slugs=touched_slugs,
                    scope=turn_scope,
                    project_id=artifacts[2],
                    project_name=artifacts[3],
                ):
                    yield ArtifactCreated(card)

        def persist(*, clean: bool = False) -> None:
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
                    # Tool rows only on a clean finish. They arrive before the
                    # terminal event, so a turn can carry rows and then fail or
                    # be cancelled — and this is called on those paths too. The
                    # default is False so a future except-branch inherits
                    # text-only instead of silently persisting a torn turn.
                    tool_rows=turn_rows if clean else None,
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
            first = True
            async for sse in format_responses_stream(
                replies_as_stream_events(), model or "", event_sink,
            ):
                if first:
                    # The formatter's created frame lacks conversation_id +
                    # harness; inject them like the in-process path does.
                    sse = self._inject_created(sse, conv_id, harness_id)
                    first = False
                await buffer.append("sse", {"sse": sse})
            persist(clean=True)
            await buffer.close("completed")
        except _RemoteTurnFailed:
            message = failure.get("message") or GENERIC_TURN_ERROR_MESSAGE
            code = failure.get("code") or GENERIC_TURN_ERROR_CODE
            collected_events.append(response_failed_payload(message, code))
            await buffer.append("sse", {"sse": response_failed_sse(message, code)})
            if code == CONTENT_RECOVERY_CODE:
                # ENG-1992: the remote/org path's twin of the streaming
                # handler's repair — producer.py already classified this via
                # remote_turn_error from the pod's scrubbed error string, so
                # `code` alone is enough to act on here.
                try:
                    repaired = ConversationService(producer_session).repair_image_content(conv_id)
                    logger.warning(
                        "[responses] content validation error on remote conversation %s — "
                        "repaired %d message(s) with image content: %s",
                        conv_id, len(repaired), failure.get("error"),
                    )
                except Exception:
                    logger.exception(
                        "[responses] failed to repair conversation %s after remote content validation error",
                        conv_id,
                    )
            persist()
            await buffer.close("error")
        except asyncio.CancelledError:
            if lifecycle.discarded:
                # Same reasoning as _run_turn's discarded branch — see there.
                logger.info("[responses] discarded remote turn %s — not persisting", conv_id)
                return
            # Partial text generated before cancellation is persisted.
            persist()
            await buffer.close("cancelled")
        except Exception:
            logger.exception("[responses] remote turn failed for conversation %s", conv_id)
            collected_events.append(response_failed_payload(
                GENERIC_TURN_ERROR_MESSAGE, GENERIC_TURN_ERROR_CODE))
            await buffer.append("sse", {"sse": response_failed_sse(
                GENERIC_TURN_ERROR_MESSAGE, GENERIC_TURN_ERROR_CODE)})
            persist()
            await buffer.close("error")
        finally:
            await _seal_unterminated_buffer(buffer, lifecycle, conv_id)
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
        turn_id: int = 0,
        trace_tags: list[str] | None = None,
        trace_metadata: dict[str, str] | None = None,
        lifecycle: TurnLifecycle | None = None,
    ) -> None:
        """Detached producer: run the turn and write events to the buffer.

        Runs in its OWN DB session (it outlives the request). Persists the user
        message (pending) as its first action so a mid-turn refresh shows the
        question, reads history via get_ordered_messages (which excludes pending,
        so the current input isn't double-fed), and on terminal finalizes the
        pending flag + persists the assistant turn (ENG-1231). Never reaches the
        HTTP response — readers tail the buffer.
        """
        lifecycle = lifecycle if lifecycle is not None else TurnLifecycle()
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
                conversation=conv, input=harness_input, model=model, disabled_connections=disabled,
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
            if lifecycle.discarded:
                # This cancellation came from a turn delete (registry.discard),
                # not from Stop: the messages this turn belongs to are already
                # gone. Persisting would write rows into truncated history, and
                # closing the buffer would recreate the file that
                # discard_conversation just removed — which the next turn would
                # then tail, since turn_id == message count is reused after a
                # truncation. So drop the turn entirely.
                logger.info("[responses] discarded turn %s — not persisting", conv_id)
                return
            # Nothing special is emitted on cancellation.
            # The partial text and events generated before cancellation are persisted.
            # A question that was on screen when Stop was pressed never got its
            # `response.ask_user_answered` (see cancelled_ask_user_retirements),
            # so retire it here — otherwise the persisted log holds a published
            # question that nothing in it ever closes.
            collected_events.extend(cancelled_ask_user_retirements(collected_events))
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
            if code == CONTENT_RECOVERY_CODE:
                # ENG-1992: the provider permanently rejected an image block in
                # this conversation's stored history — repair the DATA once,
                # here, rather than special-case every future replay. Never
                # lets a repair failure mask the turn's real outcome; the
                # user-facing message above already went out either way.
                try:
                    repaired = ConversationService(producer_session).repair_image_content(conv_id)
                    logger.warning(
                        "[responses] content validation error on conversation %s — "
                        "repaired %d message(s) with image content: %s",
                        conv_id, len(repaired), exc,
                    )
                except Exception:
                    logger.exception(
                        "[responses] failed to repair conversation %s after content validation error",
                        conv_id,
                    )
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
            elif code in MODEL_UNAVAILABLE_CODES:
                # The model was rejected (legacy 403 gate, or a 404 for a model
                # the provider can't serve): tell the client WHICH model so the
                # card can name it ("Sonnet isn't included in your plan",
                # "deepseek-v4-flash isn't a model on this provider"). Naming it
                # is the whole point for model_not_found — the id is usually one
                # the user typed or pasted, and seeing it is what makes the
                # mistake obvious (ENG-1358). No provider_label — the
                # ModelUnavailableCard doesn't render it, and
                # resolved_planning_provider would name the wrong provider when
                # the *coding* model was the one rejected.
                extra = {"model": model_info[1] if model_info else ""}
            elif code == ALLOWANCE_EXHAUSTED_CODE:
                # When the free grant refreshes (ENG-1537). The gate sends it on
                # this denial and only this one, so the card can offer waiting as
                # a real alternative to paying instead of only asking for money.
                _reset = allowance_reset_at(exc)
                if _reset is not None:
                    extra = {"reset_at": _reset}
            elif code == RATE_LIMITED_CODE:
                # Pass the server's own wait interval so the card can time-gate
                # its Retry (ENG-1537). An ungated Retry re-sends a large
                # context into the limiter that just refused it — the same
                # amplification this fix removed, only user-initiated. Absent
                # header → no gate, which is honest rather than invented.
                # Never break the handler — same rule the auth and overloaded
                # branches state below. Anything raised here skips the
                # response.failed frame AND buffer.close(), stranding the
                # client on keepalives with no error (ENG-1537 review round 3).
                try:
                    _after = retry_after_seconds(exc)
                    if _after is not None:
                        # `retry_at` is the absolute anchor the card gates on —
                        # the renderer has no trustworthy one of its own, since
                        # created_at is serialised offset-less and JS reads it
                        # as local time. `retry_after` rides along for
                        # non-desktop consumers; no cowork code reads it.
                        extra = {"retry_after": _after, "retry_at": retry_at_instant(_after)}
                except Exception:
                    logger.exception("[responses] could not resolve the retry hint")
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
            await _seal_unterminated_buffer(buffer, lifecycle, conv_id)
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
            async for _ in self._get_harness().formatter(stream, model, event_sink):
                pass
        except Exception as exc:
            # Mirror the streaming path: a recognised failure (e.g. an
            # unsupported image) surfaces its curated message with a 400;
            # anything else stays a generic 500 so provider internals never
            # leak. (cowork PR #156.)
            friendly = friendly_turn_error(exc)
            if friendly is not None:
                code, message = friendly
                logger.info("[responses] user-facing turn error: %s", exc)
                if code == CONTENT_RECOVERY_CODE:
                    # ENG-1992: see the streaming path's twin for the full
                    # rationale — repair the conversation's stored history
                    # once here rather than special-case every future replay.
                    try:
                        repaired = ConversationService(self.scoped).repair_image_content(conversation_id)
                        logger.warning(
                            "[responses] content validation error on conversation %s — "
                            "repaired %d message(s) with image content: %s",
                            conversation_id, len(repaired), exc,
                        )
                    except Exception:
                        logger.exception(
                            "[responses] failed to repair conversation %s after content validation error",
                            conversation_id,
                        )
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
        harness_id = getattr(self._get_harness(), 'id', None)
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
                # Provisions the org's default when the name is `general` — a fresh
                # org may not have its row yet on the turn that first names it.
                return service.get_or_provision_by_name(request.project).id
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=f"Project not found: {request.project}") from exc
        # Bootstrap site: a turn can be the org's first request, and each org has
        # its OWN default row — the fixed constant resolves to None in org mode.
        return service.default_project_id()

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


# A card waiting on a human produces no events, so the stream can be quiet for
# the whole question timeout. Cloudflare (proxied, in the path for every cloud
# instance) drops quiet connections; its documented threshold covers
# time-to-first-byte rather than mid-stream idle, so the exact mid-stream bound
# is unpublished. 20 s is chosen to be below any plausible one — Cloudflare's
# own published timeouts start at 100 s and no proxy in common use idles out
# under 30 s — and sits inside the design's 15-30 s window. If a stream is ever
# observed dropping mid-question in the cloud, the thing to measure is the
# elapsed time between the last byte written and the disconnect, at the edge;
# do not tune this value from a local test, where no proxy is in the path.
SSE_KEEPALIVE_SECONDS = 20.0


async def sse_from_buffer(buffer, from_seq: int = 0) -> AsyncGenerator[str, None]:
    """Serialize a turn buffer to the SSE wire, replaying from ``from_seq``
    then live-tailing. Used by both the initial POST /responses stream
    (from_seq=0) and reconnects via GET /responses/tail. The terminal record
    just ends the stream — the harness's own response.completed/failed frame
    was already written as a normal record.

    Emits a comment heartbeat whenever the buffer has been quiet for
    ``SSE_KEEPALIVE_SECONDS``, so an intermediary cannot mistake a pending
    ask_user card for a dead connection.

    Prefetch semantics: unlike a plain ``async for``, this loop keeps one
    ``__anext__()`` in flight while the current record is being yielded, so it
    runs one record ahead of the wire. That is safe only because
    ``buffer.tail(from_seq)`` is replayable — if the consumer goes away and a
    prefetched record is discarded unrendered, the client reconnects via
    ``GET /responses/tail`` from its last rendered seq and the record is
    replayed. Do not introduce a non-replayable source under this loop.
    """
    records = buffer.tail(from_seq).__aiter__()
    pending = asyncio.ensure_future(records.__anext__())
    try:
        while True:
            try:
                rec = await asyncio.wait_for(
                    asyncio.shield(pending), timeout=SSE_KEEPALIVE_SECONDS
                )
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            except StopAsyncIteration:
                return
            # Checked BEFORE prefetching: the terminal record ends the stream,
            # so scheduling another __anext__() here would only be cancelled.
            if rec.is_terminal:
                return
            pending = asyncio.ensure_future(records.__anext__())
            sse = rec.data.get("sse")
            if sse:
                yield sse
    finally:
        # Let the cancellation actually land before closing. Task.cancel() is
        # asynchronous, so closing immediately after it hits the underlying async
        # generator while ag_running_async is still set, and aclose() raises
        # "RuntimeError: aclose(): asynchronous generator is already running" —
        # which then REPLACES the CancelledError on the real disconnect path
        # (StreamingResponse cancels this task), leaving the iterator open.
        pending.cancel()
        # Narrower than suppress(BaseException) and exactly as wide as needed:
        # asyncio.wait() returns (done, pending) *sets* and never re-raises the
        # awaited task's exception, so nothing the prefetch did can surface
        # here. The only reachable exception is a cancellation of the enclosing
        # task arriving during this await — swallowing that is deliberate, so
        # that aclose() below still runs.
        cancelled: asyncio.CancelledError | None = None
        try:
            await asyncio.wait([pending])
        except asyncio.CancelledError as exc:
            cancelled = exc
        aclose = getattr(records, "aclose", None)
        if aclose is not None:
            await aclose()
        # ...but do not LOSE it. Task.__step has already cleared must_cancel by
        # the time we catch it, so on the normal-exhaustion path the generator
        # would return cleanly, the consumer's `async for` would end normally,
        # and the cancellation would vanish. Re-raise now that the iterator is
        # closed.
        if cancelled is not None:
            raise cancelled
