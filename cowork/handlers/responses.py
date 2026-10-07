from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from cowork.build_info import build_trace_metadata
from cowork.common.chat_session import in_process_agent_allowed
from cowork.common.history_scrub import scrub_credentials, scrubbed_openai_dump
from cowork.common.settings.app_settings import MINDS_FREE_MODEL, TurnQueueSettings, get_app_settings
from cowork.common.settings.user_settings import (
    Provider,
    UserSettings,
    get_user_settings,
    provider_api_key,
    use_settings_scope,
    use_turn_settings,
)
from cowork.db.units import busy_retry_seconds, conversation_writes, run_db, run_to_completion
from cowork.harnesses.base import available_harness_ids, get_harness
from cowork.handlers import jev_shadow
from cowork.handlers.response_routing import (
    DELEGATED_AGENTIC,
    DIRECT_CONTEXT,
    _MAX_HISTORY_MESSAGES,
    _text_history,
    RouteDecision,
    RouterBinding,
    decide_route,
    ineligible_reason,
)
from cowork.harnesses.anton_harness.stream_formatter import SkillCreated, format_responses_stream
from cowork.models.conversation import Conversation
from cowork.models.message import Message
from cowork.streaming import StreamBuffer, TurnLifecycle, new_buffer, registry, sse_frame
from cowork.streaming.answer_text import accumulate_answer_text
from cowork.streaming.backend import get_backend
from cowork.streaming.records import TerminalReason
from cowork.streaming.turn_index import record_turn
from cowork.turnqueue.producer import step_stream_events, stream_remote_replies
from cowork.turnqueue.redis_client import cancel_flag_key, get_redis
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
from cowork.handlers._turn_history import (
    reject_unreplayable_tool_rows,
    sanitize_turn_history_rows,
)
from cowork.handlers.turn_errors import (
    AUTH_ERROR_CODE,
    CONTENT_REPAIR_CODES,
    GENERIC_TURN_ERROR_CODE,
    GENERIC_TURN_ERROR_MESSAGE,
    INTERRUPTED_TURN_MESSAGE,
    MODEL_UNAVAILABLE_CODES,
    PROVIDER_OVERLOADED_CODE,
    RATE_LIMITED_CODE,
    REMOTE_CANCEL_LITERAL,
    REMOTE_CANCEL_VIA_FAIL_JOB,
    RESET_AT_CODES,
    SERVER_BUSY_CODE,
    auth_error_detail,
    friendly_turn_error,
    gate_reset_at,
    model_unavailable_info,
    provider_overloaded_info,
    response_failed_payload,
    retry_after_seconds,
    retry_at_instant,
    response_failed_sse,
    server_busy_message,
)
from cowork.db.scoped import (
    ScopedSession,
    scope_from_principal,
    unsafe_unscoped_session,
)
from cowork.principal import Principal, identity_trace_metadata
from cowork.services.connectors.vault_secrets import register_vault_secrets
from cowork.services.conversations import ConversationService
from cowork.services.files import FileService
from cowork.services.product_permissions import (
    ProductPermissionDenied, ProductPermissionUnavailable, require_product_permission,
)
from cowork.services.memory import apply_turn_memory, build_turn_memory
from cowork.services.projects import ProjectService
from cowork.services.settings import SettingService
from cowork.services.skills import SkillService
from cowork.services.task_objects import ArtifactChanges, remote_skill_draft_result


import logging

logger = logging.getLogger(__name__)

# Strong references for fire-and-forget probe tasks: asyncio holds only a weak
# reference to a task once nothing else does, so a bare `create_task` result
# that's dropped can be garbage-collected mid-flight. Discarded on completion
# via the done-callback below.
_jev_shadow_tasks: set[asyncio.Task] = set()


def _spawn_jev_shadow_probe(
    *, conversation_id: UUID, correlation_id: str | None, messages: list[dict],
    llm_block: dict | None, settings: TurnQueueSettings,
) -> None:
    """Runs `jev_shadow.probe` detached from the request path and logs its
    own result. Never awaited by the caller, so a slow or hung probe cannot
    delay the turn it's shadowing.

    `create_task` copies the caller's context, so the probe runs under the
    gate's anton TraceContext and `jev_shadow` attributes its Langfuse trace to
    the same conversation and correlation_id."""

    async def _run() -> None:
        jev_result = await jev_shadow.probe(messages=messages, llm_block=llm_block, settings=settings)
        if jev_result is None:
            return
        fields = " ".join(f"{k}={v}" for k, v in jev_result.items())
        logger.warning(
            "[jev-shadow] conversation=%s correlation_id=%s %s", conversation_id, correlation_id, fields,
        )

    task = asyncio.create_task(_run())
    _jev_shadow_tasks.add(task)
    task.add_done_callback(_jev_shadow_tasks.discard)


class _RemoteTurnFailed(Exception):
    """Terminal turn_failed reply; payload rides the enclosing scope."""

# Statuses of the `response.ask_user_answered` event, as the harness emits them
# (cowork/harnesses/anton_harness/stream_formatter.py). "cancelled" is the one
# this module synthesizes when a turn is stopped while a question is on screen.
ASK_USER_EVENT = "response.ask_user"
ASK_USER_ANSWERED_EVENT = "response.ask_user_answered"

#: Budget for a pod-reported history summary. Well above what the summarizer's
#: own output cap can produce, and the summary is sticky: it is replayed on
#: every later turn, so an oversized one wedges the conversation for good.
_MAX_COMPACTION_SUMMARY_BYTES = 64 * 1024


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


async def _remote_cancel_confirmed(error: str | None, correlation_id: str) -> bool:
    """Whether a remote ``turn_failed`` is the user's own Stop, not a failure.

    ``REMOTE_CANCEL_LITERAL`` is proof on its own. ``REMOTE_CANCEL_VIA_FAIL_JOB``
    is not: a controller replica taking SIGTERM mid-turn unwinds through the
    same handler that a keepalive-driven cancel does, so the string cannot tell
    a Stop from a rolling deploy killing the turn.

    The cancel flag separates them. ``/cancel`` writes it, the producer clears
    a stale one before each turn, and the keepalive-driven path never clears
    it — so its presence means a user asked for this turn to stop. A shutdown
    abort leaves no flag and stays a failure, which is the outcome that earns
    an error frame and a lookup id.
    """
    if error == REMOTE_CANCEL_LITERAL:
        return True
    if error != REMOTE_CANCEL_VIA_FAIL_JOB:
        return False
    try:
        return bool(await get_redis().exists(cancel_flag_key(correlation_id)))
    except Exception:
        # Redis is how the reply being classified arrived, so this is close to
        # unreachable. Read as a failure rather than a cancel: a visible error
        # carrying an id is the recoverable way to be wrong here.
        logger.exception(
            "[responses] could not read the cancel flag for correlation_id=%s; "
            "treating the turn as failed", correlation_id,
        )
        return False


async def _seal_unterminated_buffer(
    buffer, lifecycle: "TurnLifecycle", conv_id, *, request_id: str | None = None,
) -> None:
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

    ``request_id`` is the producing turn's own correlation id — the remote
    one for ``_produce_remote``, the locally minted one for ``_run_turn``.
    This is the hardest-failing turn (one that escaped every named ``except``),
    so it's exactly the one a user is most likely to report. Still optional:
    the direct/channel producers have no such id to offer.
    """
    if lifecycle.discarded or getattr(buffer, "is_closed", True):
        return
    logger.error(
        "[responses] turn for conversation %s (correlation_id=%s) ended without a "
        "terminal record; sealing the buffer so the client releases its stream slot",
        conv_id, request_id, extra={"request_id": request_id},
    )
    try:
        await buffer.append("sse", {"sse": response_failed_sse(
            GENERIC_TURN_ERROR_MESSAGE, GENERIC_TURN_ERROR_CODE, request_id=request_id)})
    except Exception:
        logger.exception("[responses] could not emit terminal error frame while sealing")
    try:
        await buffer.close("error")
    except Exception:
        logger.exception("[responses] could not seal unterminated turn buffer")


def _auth_failure_provider(settings, role: str | None) -> Provider | None:
    """Whose credential the failing call was using, or None if unattributable.

    Anton's `LLMClient` stamps the role on every confirmed refusal that leaves
    it, so the three named roles cover normal operation. An unstamped auth error
    can still arrive from a version-skewed anton, whose 401 is an untyped
    `ConnectionError` that `turn_errors.is_auth_error` still recognizes. Guessing
    "planning" there would name the wrong provider and offer the wrong remedy in
    a mixed configuration — a MindsHub "Reconnect" card for a failing BYOK
    Anthropic key, or the reverse.

    When the required roles resolve to the same provider there is nothing to get
    wrong, so answer. When they disagree, decline and let the caller fall back to
    the generic auth copy.
    """
    if role == "coding":
        return settings.resolved_coding_provider
    if role == "router":
        # Defensive, and not reachable today: anton's `summarize()` is the only
        # router-role call that stamps a refusal, and it swallows a confirmed one
        # rather than propagating (pinned by its
        # `test_failed_summarize_reports_no_compaction`). `decide_route` below
        # streams the provider directly, outside anton's client, so it never
        # stamps a role at all. Kept so a future propagating router path
        # attributes the card to the provider that failed rather than to planning.
        return settings.resolved_router_provider
    if role == "planning":
        return settings.resolved_planning_provider
    planning = settings.resolved_planning_provider
    return planning if planning == settings.resolved_coding_provider else None

def _turn_anchor_id(user_message: Message, assistant_message: Message | None) -> UUID:
    """The id a non-streaming caller can hand back to DELETE .../turns/{id}.

    The output item carries the assistant's text, so its id is the assistant
    row's — which is also the anchor delete_turn expects for an answered turn.
    When nothing was persisted for the assistant (an empty turn), the user row
    is the whole turn and is itself a valid orphan anchor, so it stands in.
    Previously this was always the user row's id, which named the wrong message
    for the content and 404s as an anchor once the turn has a reply.
    """
    return assistant_message.id if assistant_message is not None else user_message.id


@dataclass(frozen=True)
class _TurnStart:
    """What a request reads, as one unit, before its gate runs."""

    settings: UserSettings
    harness_name: str
    harness_input: list[dict]
    conversation_id: UUID
    turn_id: int
    # The gate's history: user and assistant rows, oldest first, at most
    # _MAX_HISTORY_MESSAGES. Empty when the turn's shape skips the gate, and
    # None when the read failed, which sends the turn to the agent.
    gate_rows: list[Message] | None


@dataclass(frozen=True)
class _StartedTurn:
    """What a turn's first unit read and wrote: the conversation, detached with
    the project the harness reads, and the id of the question it saved."""

    conversation: Conversation
    question_id: UUID


@dataclass(frozen=True)
class _SavedAnswer:
    """What the turn-end unit did with the answer.

    ``message_id`` names the assistant row it wrote, and ``failure`` is why the
    save failed. Neither means there was nothing to save: an empty turn, a turn
    whose question was never saved, or an answer this turn already saved.
    """

    message_id: UUID | None = None
    failure: Exception | None = None

    @property
    def assistant_message_id(self) -> str | None:
        """The row's id as the terminal frames carry it, or None."""
        return str(self.message_id) if self.message_id is not None else None


@dataclass(frozen=True)
class _DirectTurn:
    """The rows a direct answer saved."""

    user_message_id: UUID
    assistant_message_id: UUID | None


def _conversation_for_harness(session: ScopedSession, *, conversation_id: UUID) -> Conversation:
    """The conversation, with the project the harness reads loaded, so both
    stay readable after the unit's session closes. The harness reads the
    conversation's history and attachments in units of its own."""
    conversation = ConversationService(session).get_conversation(conversation_id)
    _ = conversation.project
    return conversation


def _start_turn(
    session: ScopedSession, *, lifecycle: TurnLifecycle, conversation_id: UUID, content, sent_at: datetime,
) -> _StartedTurn | None:
    """A streamed turn's first unit: load the conversation, then save the
    question as pending (ENG-1231), so a refresh mid-turn shows it while
    replayed history (get_ordered_messages) leaves it out."""
    if lifecycle.discarded:
        return None
    conversation = _conversation_for_harness(session, conversation_id=conversation_id)
    question = ConversationService(session).save_user_message(
        conversation_id, content, created_at=sent_at, pending=True,
    )
    return _StartedTurn(conversation=conversation, question_id=question.id)


def _save_answer(
    session: ScopedSession,
    *,
    lifecycle: TurnLifecycle,
    conversation_id: UUID,
    question_id: UUID,
    text: str,
    events: tuple[dict, ...],
    tool_rows: tuple[dict, ...],
    harness: str | None,
) -> UUID | None:
    """A streamed turn's last unit: the saved assistant row's id, or None when
    it saved no answer (an empty turn, or a deleted one)."""
    if lifecycle.discarded:
        # The turn was deleted while this save waited for a connection. Its
        # rows would land in the history the delete cut.
        return None
    service = ConversationService(session)
    # Re-anchor before ANY write: the conversation may be gone (deleted
    # mid-turn) or out of scope.
    service.get_conversation(conversation_id)
    # The question's pending flag is cleared in the commit that saves the
    # answer, so the two land together or not at all: a failed save leaves the
    # question pending and out of replayed history, never an answer whose
    # question stays pending. Scoped to THIS turn's row, so a completing turn
    # can't absorb into history a pending row stranded by an earlier crashed
    # turn. An empty turn saves no row and commits nothing itself; the unit's
    # own commit lands the flag, so the question rejoins replayed history.
    service.clear_pending(conversation_id, message_id=question_id)
    message = service.save_assistant_turn(
        conversation_id, text, list(events), harness=harness, tool_rows=list(tool_rows),
    )
    return message.id if message is not None else None


@dataclass(frozen=True)
class _RemoteArtifactsDir:
    """The project artifacts directory a remote turn's pod writes into."""

    base: Path
    project_id: str | None
    project_name: str
    # The conversation's creator, recorded in org mode as the owner of each
    # artifact the turn creates.
    creator: str | None


@dataclass(frozen=True)
class _RemoteTurnInputs:
    """What a remote turn's pod is seeded with, read in one unit
    (ResponsesHandler._read_remote_turn_inputs)."""

    history: list[dict]
    # Maps the pod's compaction count back onto our message ids
    # (_remote_seed_history); None when compaction is off.
    seed_info: dict | None
    artifacts: _RemoteArtifactsDir | None
    memory: dict | None
    # stream_remote_replies' project_id and workspace_rel_path, or empty when
    # they could not be resolved (_remote_workspace).
    workspace: dict
    started_at: str | None


@dataclass
class _RemoteTurnArtifacts:
    """A remote turn's view of its project's artifacts directory: how it stood
    before the pod ran, whether the pod's workspace may keep what it writes
    there (persistent), and whether the turn has recorded what changed."""

    directory: _RemoteArtifactsDir
    before_slugs: set[str]
    before_mtimes: dict[str, int]
    writes_allowed: bool = False
    recorded: bool = False


def _save_remote_question(
    session: ScopedSession, *, lifecycle: TurnLifecycle, conversation_id: UUID, content,
) -> UUID | None:
    """A remote turn's first write: the question, saved as pending
    so a refresh mid-turn shows it while replayed history
    (get_ordered_messages) leaves it out. None when the turn was deleted
    while this unit waited for a connection: its row would land in the
    history the delete cut."""
    if lifecycle.discarded:
        return None
    return ConversationService(session).save_user_message(
        conversation_id, content, pending=True,
    ).id


def _failed_frame_for(exc: Exception, *, request_id: str | None = None) -> str:
    """The response.failed frame for a turn failure that is the server's own:
    no database connection freed in time (server_busy, carrying the wait as
    rate_limited does, so a card can time its Retry), or anything else (the
    generic error, which leaks no internals)."""
    if isinstance(exc, PoolTimeoutError):
        retry_after = busy_retry_seconds()
        return response_failed_sse(
            server_busy_message(retry_after), SERVER_BUSY_CODE,
            retry_after=retry_after, retry_at=retry_at_instant(retry_after), request_id=request_id,
        )
    return response_failed_sse(GENERIC_TURN_ERROR_MESSAGE, GENERIC_TURN_ERROR_CODE, request_id=request_id)


class ResponsesHandler:
    def __init__(self, *, principal: Principal | None = None, interactive: bool = True) -> None:
        self.principal = principal
        self.scope = scope_from_principal(principal)
        # The harness this turn runs: the request's own pick or the stored
        # setting, resolved by handle() from the settings its first unit reads.
        # Anton itself is built lazily, only after Cowork delegates a turn:
        # direct context responses must never build the Anton harness.
        self.harness_name: str | None = None
        self.harness = None
        self.last_conversation_id: str | None = None
        # Whether a person is watching this turn and can answer ask_user cards.
        self.interactive = interactive

    def _get_harness(self):
        if self.harness is None:
            self.harness = get_harness(self.harness_name)
        return self.harness

    async def handle(self, request: ResponsesRequest) -> AsyncGenerator[str, None] | Response:
        logger.info("[responses] handle() called — conversation=%s, stream=%s", request.conversation, request.stream)

        await require_product_permission(self.scope, "product.execute")

        # Identity + the running build into the run's trace metadata;
        # server-derived keys win. The build stamp (ENG-1279) is what lets a
        # metric be attributed to a release instead of to a date on which
        # several changes happened to ship together.
        trace_metadata = build_trace_metadata(identity_trace_metadata(self.principal, request.trace_metadata))
        original_content = self._extract_original_content(request)
        disabled = (
            [dc.model_dump() for dc in request.disabled_connections]
            if request.disabled_connections else None
        )

        # Before the reads: the gate and the producer task both scrub history
        # in this request's context, so the vault's secrets are registered
        # first. It needs only the scope.
        await register_vault_secrets(self.scope)

        # The request's reads run as one unit in a worker thread, which holds a
        # connection only while it reads: none is held across the gate's model
        # call, and a wait for one never stalls the event loop.
        start = await run_db(
            partial(self._read_turn_start, request=request, has_disabled_connections=bool(disabled)),
            scope=self.scope,
        )
        self.harness_name = start.harness_name
        self.last_conversation_id = str(start.conversation_id)

        # Every settings read for this turn, in the gate and in the producer
        # task created below (create_task copies this context), is served from
        # the snapshot the unit loaded.
        with use_turn_settings(self.scope, start.settings):
            route, turn_llm = await self._route_request(
                conversation_id=start.conversation_id,
                harness_input=start.harness_input,
                gate_rows=start.gate_rows,
                has_attachments=bool(request.attachment_ids),
                has_disabled_connections=bool(disabled),
                trace_metadata=trace_metadata,
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
                route.route, route.reason, route.fallback, route.provider, route.model,
                start.conversation_id,
            )

            if route.route == DIRECT_CONTEXT:
                return await self._handle_direct_response(
                    request=request,
                    conversation_id=start.conversation_id,
                    turn_id=start.turn_id,
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
                # action, not here (ENG-1231). registry.start() refuses a second
                # turn for a conversation before it builds this turn's buffer or
                # producer, so persisting here, before that check, would commit a
                # pending row whose producer never runs and never finalizes,
                # stranding it out of LLM history. Persisting inside the producer
                # ties the write to the one coroutine that actually runs, so there
                # is at most one pending row per conversation.
                #
                # Created here, before the coroutine, and handed to BOTH: it is
                # the only channel by which a turn delete can tell this producer
                # that the history it is writing into no longer exists (the handle
                # it will be registered under does not exist yet).
                lifecycle = TurnLifecycle()

                def produce(buffer: StreamBuffer):
                    return self._select_producer(
                        lifecycle=lifecycle,
                        conv_id=start.conversation_id,
                        harness_input=start.harness_input,
                        original_content=original_content,
                        model=request.model,
                        reasoning_effort=request.reasoning_effort,
                        disabled=disabled,
                        harness_name=self.harness_name,
                        harness_id=getattr(harness, "id", None),
                        buffer=buffer,
                        turn_id=start.turn_id,
                        trace_tags=request.trace_tags,
                        trace_metadata=trace_metadata,
                        turn_llm=turn_llm,
                    )

                handle = await registry.start(
                    conversation_id=str(start.conversation_id),
                    turn_id=start.turn_id,
                    open_buffer=partial(new_buffer, str(start.conversation_id), start.turn_id),
                    produce=produce,
                    org_id=self.scope.org_id,
                    user_id=self.scope.user_id,
                    lifecycle=lifecycle,
                )
                return sse_from_buffer(handle.buffer, 0)

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
            #
            # The harness gets the conversation detached, and reads its history
            # in units of its own.
            conversation = await run_db(
                partial(_conversation_for_harness, conversation_id=start.conversation_id),
                scope=self.scope,
            )
            with use_settings_scope(self.scope):
                stream = harness.stream_response(
                    conversation=conversation,
                    input=start.harness_input,
                    model=request.model,
                    reasoning_effort=request.reasoning_effort,
                    disabled_connections=disabled,
                    trace_tags=request.trace_tags,
                    trace_metadata=trace_metadata,
                )
                return await self._collect(stream, start.conversation_id, request.model, original_content)

    def _read_turn_start(
        self, session: ScopedSession, *, request: ResponsesRequest, has_disabled_connections: bool,
    ) -> _TurnStart:
        """The request's database work, run as one unit: the turn's settings,
        its attachments, the conversation (read, adopted or created), the turn
        number and the gate's history."""
        # SettingService routes each key to its global, org or user row itself,
        # so it reads through the raw session with the scope passed explicitly.
        settings = SettingService(unsafe_unscoped_session(session), session.scope).load()
        # A per-conversation harness pick (Coding Mode's composer pill)
        # overrides the account default for THIS call only — mirrors the
        # per-conversation model override. Ignored (not raised) when it
        # doesn't name a currently-registered/available harness: a stale
        # client cache (a harness removed since the picker last loaded)
        # must never fail the turn, it just falls back to the
        # account default.
        harness_name = settings.harness
        if request.harness and request.harness in available_harness_ids():
            harness_name = request.harness

        harness_input = self._build_harness_input(request, session=session)
        conversation_service = ConversationService(session)

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
                    try:
                        conversation = conversation_service.create_conversation(
                            topic=self._prompt_text(harness_input)[:80],
                            project_id=self._resolve_project_id(request, session=session),
                            conversation_id=conv_id,
                            harness=harness_name,
                            model=request.model,
                            reasoning_effort=request.reasoning_effort,
                        )
                    except IntegrityError:
                        # A second first send for this new conversation (a
                        # double click, a client retry) inserted it between
                        # the read above and this insert. Go on with that row:
                        # the registry refuses whichever question comes
                        # second, as it does any duplicate send.
                        session.rollback()
                        conversation = conversation_service.get_conversation(conv_id)
            else:
                # Client sent a non-UUID id (e.g. the legacy timestamp
                # allocator, or a name-based format) — it can't become the
                # row id, so create a fresh conversation and re-link any
                # attachments uploaded against the client's id (ENG-264).
                conversation = conversation_service.create_conversation(
                    topic=self._prompt_text(harness_input)[:80],
                    project_id=self._resolve_project_id(request, session=session),
                    harness=harness_name,
                    model=request.model,
                    reasoning_effort=request.reasoning_effort,
                )
                self._relink_attachments(request.conversation, conversation, session=session)
        else:
            conversation = conversation_service.create_conversation(
                topic=self._prompt_text(harness_input)[:80],
                project_id=self._resolve_project_id(request, session=session),
                harness=harness_name,
                model=request.model,
                reasoning_effort=request.reasoning_effort,
            )

        conversation_id = conversation.id
        # turn_id: prior message count. The current user message is NOT
        # persisted yet (deferred to the producer for the streaming path), so
        # this is a stable per-conversation index for the buffer file.
        turn_id = len(conversation.messages)

        # Shape checks first: a turn the gate cannot route skips the history
        # read, as _route_request skips the gate.
        has_non_text_input = any(block.get("type") != "text" for block in harness_input)
        gate_rows: list[Message] | None = []
        if ineligible_reason(
            has_non_text_input=has_non_text_input,
            has_attachments=bool(request.attachment_ids),
            has_disabled_connections=has_disabled_connections,
        ) is None:
            gate_rows = self._read_gate_rows(session, conversation_id=conversation_id)

        return _TurnStart(
            settings=settings,
            harness_name=harness_name,
            harness_input=harness_input,
            conversation_id=conversation_id,
            turn_id=turn_id,
            gate_rows=gate_rows,
        )

    @staticmethod
    def _read_gate_rows(session: ScopedSession, *, conversation_id: UUID) -> list[Message] | None:
        """The rows decide_route can use: user and assistant messages, at most
        _MAX_HISTORY_MESSAGES, so scrubbing never pays for the whole
        conversation. None when the read fails: routing failures send the turn
        to the agent rather than failing it."""
        try:
            rows = ConversationService(session).get_ordered_messages(conversation_id)
        except PoolTimeoutError:
            raise
        except Exception:
            logger.exception("[responses] could not read the gate's history; delegating")
            # The failed read left its transaction unusable. Rolling it back
            # loses nothing: creating or relinking above committed already.
            session.rollback()
            return None
        return [m for m in rows if m.role in {"user", "assistant"}][-_MAX_HISTORY_MESSAGES:]

    async def _route_request(
        self,
        *,
        conversation_id: UUID,
        harness_input: list[dict],
        gate_rows: list[Message] | None,
        has_attachments: bool,
        has_disabled_connections: bool,
        trace_metadata: dict[str, str] | None = None,
    ) -> tuple[RouteDecision, dict | None]:
        """Run Cowork's narrow pre-Anton gate with only safe text context.

        `gate_rows` is the history handle()'s first unit read for the gate
        (see _TurnStart).

        The composer's per-conversation model pick (`request.model`) is
        deliberately not passed down: it drives Anton's turn, not the gate
        (see `UserSettings.resolved_gate_model`).

        Returns the decision plus pre-minted turn credentials
        (`{"correlation_id", "llm"}`) for a delegated remote turn to reuse."""
        has_non_text_input = any(block.get("type") != "text" for block in harness_input)
        # Shape checks first: ineligible turns were never given history.
        reason = ineligible_reason(
            has_non_text_input=has_non_text_input,
            has_attachments=has_attachments,
            has_disabled_connections=has_disabled_connections,
        )
        if reason:
            return RouteDecision(route=DELEGATED_AGENTIC, reason=reason), None
        if gate_rows is None:
            # The history read failed. Like any other gate failure, the turn
            # goes to the agent rather than failing.
            return RouteDecision(
                route=DELEGATED_AGENTIC, reason="router_unavailable", fallback=True
            ), None
        try:
            # Scrub credentials: this history bypasses the normal turn's
            # _scrub_user_input/_stamp_message pass. Its DS_* values were
            # registered by handle(), not _build_chat_session. Scrubbed here on
            # the loop, after that registration, never inside the unit.
            history = [scrubbed_openai_dump(m) for m in gate_rows]
            history.append({
                "role": "user",
                "content": scrub_credentials(self._prompt_text(harness_input)),
            })
            # The gate's LLM call is the only one a direct turn makes, and it
            # is made outside any ChatSession — so without a trace context it
            # reaches MindsHub anonymous, and a direct turn leaves no trace to
            # count (ENG-1851 Done-when #1). Installing one attributes the call
            # to the conversation and stamps the build (ENG-1279), which is what
            # makes the direct/delegated share answerable from traces.
            from anton.core.llm.tracing import (
                TraceContext,
                reset_trace_context,
                set_trace_context,
            )

            trace_token = None
            try:
                # The gate resolves the router role + key ambiently; bind the org scope.
                with use_settings_scope(self.scope):
                    # Minted before the context goes in (it is an auth call, not
                    # an LLM call), so the context can carry the turn's
                    # correlation_id: the gate's trace and the Jev shadow probe's
                    # (which inherits this context) share it, making gate
                    # decision <-> Jev answer an exact join. Never
                    # turn_id: the gateway renames harness+turn_id traces to
                    # "{harness}:turn-N", which would count these as user turns.
                    binding, turn_llm = await self._router_binding()
                    correlation_id = (turn_llm or {}).get("correlation_id")
                    trace_token = set_trace_context(TraceContext(
                        session_id=str(conversation_id),
                        harness=self.harness_name,
                        tags=("cowork-gate",),
                        metadata={
                            **(trace_metadata or {}),
                            **({"correlation_id": correlation_id} if correlation_id else {}),
                        },
                    ))
                    turn_queue_settings = TurnQueueSettings()

                    gate_started = time.monotonic()
                    decision = await decide_route(
                        history=history,
                        has_non_text_input=has_non_text_input,
                        has_attachments=has_attachments,
                        has_disabled_connections=has_disabled_connections,
                        binding=binding,
                    )
                    gate_ms = round((time.monotonic() - gate_started) * 1000)
                    # warning, not info: this deployment's LOG_LEVEL defaults to
                    # WARNING (app_settings.py's own default too), so an info-level
                    # line here is silently dropped everywhere it would actually
                    # be read from — found live on staging, zero [gate] lines
                    # across 8 real requests until this was bumped.
                    logger.warning(
                        "[gate] conversation=%s correlation_id=%s route=%s reason=%s "
                        "provider=%s model=%s gate_ms=%d",
                        conversation_id, correlation_id, decision.route, decision.reason,
                        decision.provider, decision.model, gate_ms,
                    )
                    # Detached on purpose: awaiting this (even via asyncio.gather)
                    # would make a ready gate decision wait for Jev, exactly the
                    # thing a *shadow* probe must never do. Logs on its own once
                    # it finishes; never read by anything on the request path.
                    _spawn_jev_shadow_probe(
                        conversation_id=conversation_id,
                        correlation_id=correlation_id,
                        messages=_text_history(history),
                        llm_block=(turn_llm or {}).get("llm"),
                        settings=turn_queue_settings,
                    )
            finally:
                if trace_token is not None:
                    reset_trace_context(trace_token)
            return decision, turn_llm
        except (ProductPermissionDenied, ProductPermissionUnavailable):
            raise
        except PoolTimeoutError:
            # No database connection freed in time. The agent would meet the
            # same full pool, so the request is refused (503), not delegated.
            raise
        except Exception:
            # Non-authorization routing failures may delegate to the agent.
            logger.exception("[responses] routing gate failed — delegating")
            return RouteDecision(
                route=DELEGATED_AGENTIC, reason="router_unavailable", fallback=True
            ), None

    async def _router_binding(self) -> tuple[RouterBinding | None, dict | None]:
        """Hosted orgs keep no stored Minds key (remote turns mint one), so the
        gate mints its own per-turn key here and hands it back for the
        delegated turn to reuse. Everywhere else the stored settings apply."""
        if not TurnQueueSettings().is_remote:
            return None, None
        settings = get_user_settings(self.scope)
        if (settings.resolved_router_provider is not Provider.MINDS_CLOUD
                or provider_api_key(settings, Provider.MINDS_CLOUD) is not None):
            return None, None
        from anton.core.llm.openai import OpenAIProvider
        from cowork.turnqueue.producer import _mint_llm_block, _mint_llm_block_with_turn_key_id

        corr = str(uuid4())
        queue_settings = TurnQueueSettings()
        turn_key_id = None
        workspace_id = getattr(settings, "hub_workspace_id", "") or None
        if queue_settings.datasource_enabled:
            block, turn_key_id = await _mint_llm_block_with_turn_key_id(
                org_id=self.scope.org_id,
                user_id=self.scope.user_id,
                correlation_id=corr,
                settings=queue_settings,
                workspace_id=workspace_id,
            )
        else:
            block = await _mint_llm_block(
                org_id=self.scope.org_id,
                user_id=self.scope.user_id,
                correlation_id=corr,
                settings=queue_settings,
                workspace_id=workspace_id,
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
        turn_context = {"correlation_id": corr, "llm": block}
        if turn_key_id is not None:
            turn_context["turn_key_id"] = turn_key_id
        return binding, turn_context

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
            text = route.text
            events = [{
                "type": "response.output_text.delta",
                "delta": text,
                "response_route": route.route,
                "response_route_reason": route.reason,
            }, {"type": "response.completed"}]

            def save_turn(session: ScopedSession) -> UUID:
                service = ConversationService(session)
                user_message = service.save_user_message(conversation_id, original_content)
                assistant_message = service.save_assistant_turn(
                    conversation_id, text, events, harness="cowork-direct",
                )
                return _turn_anchor_id(user_message, assistant_message)

            async with conversation_writes(conversation_id):
                anchor_id = await run_db(save_turn, scope=self.scope)
            return Response(
                status=ResponseStatus.completed,
                model=route.model,
                output=[self._build_output(str(anchor_id), text)],
            )

        # turn_id comes from handle(): same numbering as the delegated path.
        lifecycle = TurnLifecycle()
        handle = await registry.start(
            conversation_id=str(conversation_id),
            turn_id=turn_id,
            open_buffer=partial(new_buffer, str(conversation_id), turn_id),
            produce=lambda buffer: self._produce_direct(
                lifecycle=lifecycle,
                conv_id=conversation_id,
                original_content=original_content,
                route=route,
                buffer=buffer,
            ),
            org_id=self.scope.org_id,
            user_id=self.scope.user_id,
            lifecycle=lifecycle,
        )
        # A direct answer still uses the shared Redis buffer in a multi-replica
        # deployment. Register it just like a delegated turn so another replica
        # can locate and replay that buffer. start() refuses a duplicate send,
        # so the handle is always this turn's own.
        if get_backend() == "redis":
            await record_turn(
                str(conversation_id),
                turn_id=turn_id,
                correlation_id=f"direct-{uuid4()}",
                org_id=self.scope.org_id,
                user_id=self.scope.user_id,
            )
        return sse_from_buffer(handle.buffer, 0)

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

        The full answer exists up front — the gate does not return until its
        stream ends — so persistence happens before any frame is emitted: the
        client can never see a completed turn the DB does not have, and no
        pending row is needed. Both rows are saved in one unit, under the
        conversation's write lock.

        The save, the frames that report it and the terminal record are one
        step a Stop cannot split: a Stop that lands during the save waits for
        it, so the stream ends with what the database holds."""
        item_id = f"msg-{conv_id.hex[:12]}"
        delta = {
            "type": "response.output_text.delta",
            "sequence_number": 2,
            "item_id": item_id,
            "delta": route.text,
            "response_route": route.route,
            "response_route_reason": route.reason,
        }

        def save_turn(session: ScopedSession) -> _DirectTurn | None:
            if lifecycle.discarded:
                # The turn was deleted while this save waited for a
                # connection. Its rows would land in the history the delete
                # cut.
                return None
            service = ConversationService(session)
            user_message = service.save_user_message(conv_id, original_content)
            assistant_message = service.save_assistant_turn(
                conv_id, route.text, [delta, {"type": "response.completed"}],
                harness="cowork-direct",
            )
            return _DirectTurn(
                user_message_id=user_message.id,
                assistant_message_id=assistant_message.id if assistant_message is not None else None,
            )

        async def answer() -> None:
            async with conversation_writes(conv_id):
                saved = await run_db(save_turn, scope=scope_from_principal(self.principal))
            if saved is None or lifecycle.discarded:
                # Deleted while it saved: its buffer is gone, and writing a
                # terminal record would recreate it for the next turn to tail.
                return
            response = Response(status=ResponseStatus.created, model=route.model)
            # conversation_id/harness sit at the event root, like both
            # delegated paths — the GUI reads them there.
            await buffer.append("sse", {"sse": sse_frame("response.created", {
                "type": "response.created",
                "sequence_number": 1,
                "conversation_id": str(conv_id),
                "harness": "cowork-direct",
                "user_message_id": str(saved.user_message_id),
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
            completed_frame = {
                "type": "response.completed",
                "sequence_number": 3,
                "response": completed_response,
            }
            # The persisted row's real id, at the frame root (not
            # inside `response`) — lets the client delete/rekey this turn
            # without ever having only a positional index for it. Both rows
            # are already persisted above, before this frame is built, so no
            # reordering is needed on this path. Omitted, not null, when
            # nothing was persisted — same convention every other producer
            # uses for this field.
            if saved.assistant_message_id is not None:
                completed_frame["assistant_message_id"] = str(saved.assistant_message_id)
            await buffer.append("sse", {"sse": sse_frame("response.completed", completed_frame)})
            await buffer.close("completed")

        try:
            await run_to_completion(answer())
        except asyncio.CancelledError:
            if lifecycle.discarded:
                # Same reasoning as _produce_remote's discarded branch.
                logger.info("[responses] discarded direct turn %s — not persisting", conv_id)
                return
            await buffer.close("cancelled")
        except Exception as exc:
            if isinstance(exc, PoolTimeoutError):
                # No database connection freed in time to save the answer. The
                # frame says so with the wait, as a refused request's 503 does.
                logger.warning(
                    "[responses] direct turn for conversation %s found no free database connection",
                    conv_id,
                )
            else:
                logger.exception("[responses] direct turn failed for conversation %s", conv_id)
            await buffer.append("sse", {"sse": _failed_frame_for(exc)})
            await buffer.close("error")
        finally:
            await _seal_unterminated_buffer(buffer, lifecycle, conv_id)

    def _select_producer(
        self,
        *,
        conv_id: UUID,
        harness_input: list[dict],
        original_content,
        model: str,
        reasoning_effort: str | None = None,
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
        if TurnQueueSettings().is_remote:
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
                disabled=disabled,
            )
        return self._produce(
            lifecycle=lifecycle,
            conv_id=conv_id,
            harness_input=harness_input,
            original_content=original_content,
            model=model,
            reasoning_effort=reasoning_effort,
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
            from cowork.services.artifact_roots import project_artifacts_base
            from cowork.services.files import stage_project_instructions

            conversation = ConversationService(session).get_conversation(conv_id)
            project_path = conversation.project.path
            # ENG-2056: the pod mounts the PROJECT-level artifacts base (subPath
            # `.anton/artifacts`) at /project-artifacts, and a subPath mount needs
            # the directory to exist before the pod starts. Project creation does
            # not make it, so make it here — this is the one place that runs
            # before every remote turn. First in the block: the pod needs it even
            # when a later staging step degrades.
            project_artifacts_base(project_path).mkdir(parents=True, exist_ok=True)
            FileService(session).stage_conversation_attachments(conv_id, project_path)
            stage_project_instructions(project_path, conv_id)
            SkillService(session.scope).ensure_builtin_skills()
        except Exception:
            logger.exception("[responses] failed to stage workspace files for conversation %s", conv_id)

    @staticmethod
    def _remote_started_at(session: ScopedSession, conv_id: UUID) -> str | None:
        """The conversation's creation time as ISO 8601, for the pod's fixed
        "conversation started" prompt line. Mirrors what the in-process path
        passes as `started_at`. A lookup failure degrades to None (today's
        date in the pod) rather than failing the turn."""
        try:
            created_at = getattr(
                ConversationService(session).get_conversation(conv_id), "created_at", None
            )
            return created_at.isoformat() if created_at is not None else None
        except Exception:
            logger.exception("[responses] failed to resolve started_at for conversation %s", conv_id)
            return None

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
        a channel turn's `index_turn_artifacts` recovers the tenant scope from
        the session the row is bound to; the ids, the project name and (on the
        remote producer, _read_remote_turn_inputs) the creator are read here,
        while it is unambiguously attached, rather than after the turn.

        None on any failure: no artifact card and no autopublish is better than
        a failed turn. Outside org mode the next turn in this project
        reconciles them; in org mode nothing does.
        """
        from cowork.services.artifact_roots import project_artifacts_base

        try:
            conversation = ConversationService(session).get_conversation(conv_id)
            # ENG-2056: project-scoped in BOTH modes. The pod mounts the
            # PROJECT-level base at /project-artifacts and anton writes there
            # (ANTON_CLOUD_ARTIFACTS_ROOT), so that is where the worker's
            # artifacts land — no longer under conversations/<id>/. Resolved
            # through artifact_roots so this and the artifacts list agree on
            # the layout.
            #
            # The base is now shared by every task in the project, so the raw
            # before/after diff below can attribute a concurrent sibling turn's
            # artifact to this turn. Pre-existing caveat, not new machinery: the
            # in-process path bounds the same diff with the session's
            # artifacts_touched set (ENG-1933), but the remote pod reports no
            # equivalent yet, so the diff stands alone here.
            artifacts_base = project_artifacts_base(conversation.project.path)
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
        """This project's shared memory slots for the pod.

        Personal/global memory is already mounted read-only per (org, user).
        The pod's writable mount starts at the conversation workspace, so it
        cannot see the project-level ``.anton/memory`` sibling unless these two
        slots travel on the wire. A read error degrades to a turn without
        project memory rather than failing the turn.
        """
        try:
            conversation = ConversationService(session).get_conversation(conv_id)
            resolved = build_turn_memory(session.scope, conversation.project.path)
            project = resolved.get("project")
            return {"project": project} if project else {}
        except Exception:
            logger.exception("[responses] failed to read memory for conversation %s", conv_id)
            return {}

    @staticmethod
    def _persist_turn_memory(
        session: ScopedSession,
        conv_id: UUID,
        entries: list,
        principal: Principal | None,
    ) -> None:
        """Apply what the pod asked to remember, re-anchoring the conversation
        first (like persist(): one deleted mid-turn must not write memory).

        Never fails the turn — a lost memory is recoverable, a lost reply isn't.
        """
        if not entries:
            return
        try:
            conversation = ConversationService(session).get_conversation(conv_id)
            from cowork.models.project import Project
            from cowork.services.shared_resources import (
                PROJECT,
                SharedResourceAccess,
                project_resource_key,
            )

            access = SharedResourceAccess(session, principal)
            project_id = conversation.project_id
            with access.coordination_lock(
                PROJECT,
                project_resource_key(project_id),
            ):
                # The turn may have waited behind rename/delete. Re-anchor both
                # rows while holding the project lock and use only that current
                # path for the contained memory mutation.
                conversation = ConversationService(session).get_conversation(conv_id)
                session.refresh(conversation)
                if conversation.project_id != project_id:
                    raise RuntimeError("Conversation project changed during memory write")
                project = session.get(Project, project_id)
                if project is None:
                    raise ValueError("Project not found")
                session.refresh(project)
                applied = apply_turn_memory(
                    session.scope,
                    project.path,
                    entries,
                    access=access,
                    project_id=project.id,
                )
            logger.info("[responses] applied %d memory entr(ies) for conversation %s",
                        applied, conv_id)
        except Exception:
            logger.exception("[responses] failed to apply memory for conversation %s", conv_id)

    def _read_remote_turn_inputs(self, session: ScopedSession, *, conv_id: UUID) -> _RemoteTurnInputs:
        """What a remote turn's pod is seeded with, read as one unit once the
        question is saved: the scrubbed history, where the project's artifacts
        live and who created the conversation, the project's memory, its
        workspace, and when the conversation started."""
        history, seed_info = self._remote_seed_history(session, conv_id)
        artifacts = self._remote_artifacts_context(session, conv_id)
        return _RemoteTurnInputs(
            history=history,
            seed_info=seed_info,
            artifacts=None if artifacts is None else _RemoteArtifactsDir(
                base=artifacts[1],
                project_id=artifacts[2],
                project_name=artifacts[3],
                creator=getattr(artifacts[0], "created_by", None),
            ),
            memory=self._remote_memory(session, conv_id),
            workspace=self._remote_workspace(session, conv_id),
            started_at=self._remote_started_at(session, conv_id),
        )

    @staticmethod
    def _remote_seed_history(session, conv_id) -> tuple[list[dict], dict | None]:
        """History to seed the pod with, and what's needed to map its compaction
        result back onto our messages.

        Messages are OpenAI-shaped, scrubbed dicts (mode="json": the payload
        gets json.dumps'd into the Redis job). The pod's harness only scrubs the
        current turn's input, never this replayed history, so it must arrive
        already clean.

        With compaction on, this is `[summary] + [messages after the cutoff]`
        rather than the whole conversation, exactly as the in-process path
        seeds it — the pod compacts either way, and without the saved summary
        every turn resent the full history and paid to summarize it again.
        `seed_info` carries message *ids*, not ORM rows: the pod's reply lands
        after this session may be closed.

        Unlike in-process, messages are not timestamp-stamped here; that
        divergence is tracked separately.
        """
        from cowork.harnesses.anton_harness.harness import AntonHarness

        service = ConversationService(session)
        replayable = [
            m for m in service.get_ordered_messages(conv_id)
            if m.role in {"user", "assistant"}
        ]
        fmt = partial(scrubbed_openai_dump, mode="json")
        if not get_user_settings(session.scope).history_compaction_enabled:
            return [fmt(m) for m in replayable], None

        conversation = service.get_conversation(conv_id)
        history, seed_info = AntonHarness._seed_history(
            replayable,
            conversation.history_summary,
            conversation.history_summary_cutoff_id,
            fmt,
        )
        return history, {
            "message_ids": [m.id for m in seed_info["ordered_messages"]],
            "tail_start": seed_info["tail_start"],
            "synthetic_prefix_len": seed_info["synthetic_prefix_len"],
        }

    @staticmethod
    def _persist_remote_compaction(
        session: ScopedSession, *, conv_id: UUID, data: dict, seed_info: dict | None,
    ) -> None:
        """Save the summary the pod folded this turn's leading history into,
        in the caller's session (a unit's, on the remote producer).

        Everything here is untrusted: the pod reports `covered_through` against
        the history we sent it, so a wrong or malformed count must degrade to
        "no compaction saved" — the next turn then replays in full, which is
        merely the old behaviour — never to a cutoff pointing at the wrong
        message, which would silently drop real turns from every later replay.
        """
        from cowork.harnesses.anton_harness.harness import AntonHarness

        summary = data.get("summary")
        covered_through = data.get("covered_through") or 0
        if not seed_info or not summary:
            return
        # Types, not just presence: the arithmetic below runs outside the
        # try/except, so a string count would raise out of the turn's reply
        # loop and fail the turn it rode in on.
        if (
            not isinstance(summary, str)
            or not isinstance(covered_through, int)
            or isinstance(covered_through, bool)
        ):
            logger.warning(
                "[responses] malformed compaction frame for conversation %s — not saved",
                conv_id,
            )
            return
        # Rejected, not truncated: half a summary is still replayed on every
        # later turn, while dropping the frame costs one full replay.
        summary_bytes = len(summary.encode("utf-8"))
        if summary_bytes > _MAX_COMPACTION_SUMMARY_BYTES:
            logger.warning(
                "[responses] compaction summary for conversation %s is %d bytes, over "
                "the %d-byte cap — not saved",
                conv_id, summary_bytes, _MAX_COMPACTION_SUMMARY_BYTES,
            )
            return
        message_ids = seed_info["message_ids"]
        idx = AntonHarness.compaction_cutoff_index(
            seed_info, covered_through, len(message_ids),
        )
        if idx is None:
            return
        try:
            ConversationService(session).update_history_compaction(
                conv_id, summary, message_ids[idx],
            )
        except Exception:
            # Rolled back so the session stays usable: the unit still commits,
            # and a channel turn goes on in the same session.
            session.rollback()
            logger.exception(
                "[responses] failed to persist history compaction for conversation %s",
                conv_id,
            )

    async def _produce_remote(self, **kwargs) -> None:
        # Detached task: bind the turn's org scope, as _produce does. Every
        # settings read in the remote turn's subtree then resolves this org,
        # and the turn's settings snapshot, which handle() bound around this
        # task's creation, answers it without a connection (use_turn_settings).
        with use_settings_scope(scope_from_principal(self.principal)):
            await self._run_remote_turn(**kwargs)

    async def _run_remote_turn(
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
        disabled: list[dict] | None = None,
    ) -> None:
        """Remote-backend counterpart of _run_turn: pipe the pod's replies
        through the same SSE formatter as the in-process path (full step /
        thinking parity, live and in the persisted events log).

        Its database work runs as units (cowork.db.units), so it holds no
        pooled connection while the pod answers, and a wait for one never
        stalls the event loop. Staging the workspace is the first unit. The
        next saves the question (pending) under the conversation's
        write lock. What the pod is seeded with is read in one unit when the
        reply stream starts (get_ordered_messages leaves the pending row out,
        so the current input isn't replayed). Memory and compaction the pod
        reports are saved as they arrive, the artifacts it created are
        recorded once whichever way the turn ends, and the last unit saves the
        assistant turn and clears the question's pending flag in one commit.
        Never reaches the HTTP response: readers tail the buffer.
        """
        lifecycle = lifecycle if lifecycle is not None else TurnLifecycle()
        # Scoped from the immutable principal captured at handler
        # construction, never from request state.
        scope = scope_from_principal(self.principal)
        collected_text: list[str] = []
        collected_events: list[dict] = []
        # This turn's tool block-rows, for LLM-history persistence only. Kept
        # out of collected_events: the client rebuilds its UI from those, and
        # tool rows are hidden from the UI (mirrors _run_turn's event_sink).
        turn_rows: list[dict] = []
        persisted = False
        question_id: UUID | None = None
        turn_artifacts: _RemoteTurnArtifacts | None = None
        failure: dict = {}
        # Resolved once, up front — not left for stream_remote_replies to mint
        # internally on a None — so every failure branch below (classified or
        # not) can attach the SAME id a support/log lookup would use. Each of
        # those branches logs the id at WARNING or above, which is the floor
        # the deployed environments run at; this INFO line is only the
        # per-turn breadcrumb that ties a successful turn to its id in dev.
        corr = (turn_llm or {}).get("correlation_id") or str(uuid4())
        logger.info(
            "[responses] remote turn conversation=%s correlation_id=%s", conv_id, corr,
        )

        def event_sink(event_type: str, data: dict) -> None:
            # Same event log the in-process path records, so the client
            # rebuilds the thinking block + steps identically on reload.
            # at_ms is stamped at receipt (the pod sends no timestamps), so
            # replayed durations are approximate under consumer lag.
            collected_events.append(data)
            accumulate_answer_text(collected_text, event_type, data)

        async def save_memory(entries: list) -> None:
            """Apply what the pod asked to remember, as it arrives. Never
            fails the turn: a lost memory is recoverable, a lost reply isn't."""
            if not entries:
                return
            try:
                await run_db(
                    partial(
                        self._persist_turn_memory,
                        conv_id=conv_id, entries=entries, principal=self.principal,
                    ),
                    scope=scope,
                )
            except PoolTimeoutError:
                logger.warning(
                    "[responses] no database connection freed in time to apply memory for "
                    "conversation %s", conv_id, extra={"request_id": corr},
                )
            except Exception:
                logger.exception(
                    "[responses] failed to apply memory for conversation %s", conv_id,
                    extra={"request_id": corr},
                )

        async def save_compaction(data: dict, seed_info: dict | None) -> None:
            """Save the summary the pod folded earlier history into, as it
            arrives: the cutoff names a message from an earlier turn, so it
            stays correct even if this turn goes on to fail. Never fails the
            turn: without it the next turn replays in full."""
            try:
                await run_db(
                    partial(
                        self._persist_remote_compaction,
                        conv_id=conv_id, data=data, seed_info=seed_info,
                    ),
                    scope=scope,
                )
            except PoolTimeoutError:
                logger.warning(
                    "[responses] no database connection freed in time to save the compaction for "
                    "conversation %s", conv_id, extra={"request_id": corr},
                )
            except Exception:
                logger.exception(
                    "[responses] failed to persist history compaction for conversation %s", conv_id,
                    extra={"request_id": corr},
                )

        async def record_artifacts(*, completed_cleanly: bool) -> ArtifactChanges | None:
            """Record what the pod did to the project's artifacts, once per
            turn, whichever way the turn ends, as the in-process harness's
            `finally` does: the folders it created are indexed as this
            conversation's (and, in org mode, its creator's), in a unit. The
            changes, for a clean finish to publish and card; None when the
            turn may not keep what the pod wrote (its workspace was not
            persistent), when it recorded them already, or when the diff
            failed. Outside org mode the next turn in the project reconciles
            them; in org mode nothing does."""
            from cowork.services.task_objects import record_new_artifacts, turn_artifact_changes

            if turn_artifacts is None or not turn_artifacts.writes_allowed or turn_artifacts.recorded:
                return None
            turn_artifacts.recorded = True
            directory = turn_artifacts.directory
            try:
                changes = turn_artifact_changes(
                    conversation_id=conv_id,
                    artifacts_base=directory.base,
                    before=turn_artifacts.before_slugs,
                    before_mtimes=turn_artifacts.before_mtimes,
                    # ENG-2961: the project base is shared, so only folders
                    # whose provenance names this conversation are its own.
                    attribute_by_provenance=True,
                    completed_cleanly=completed_cleanly,
                )
            except Exception:
                logger.warning(
                    "[responses] could not diff the artifacts of remote turn %s", conv_id,
                    exc_info=True, extra={"request_id": corr},
                )
                return None
            if changes.created:
                try:
                    # The turn is marked recorded before this unit, so its
                    # cancel branches never record again. A Stop that lands
                    # while the unit waits for a slot or a connection waits
                    # for it, rather than abandoning the rows.
                    await run_to_completion(run_db(
                        partial(
                            record_new_artifacts,
                            conversation_id=conv_id,
                            project_id=UUID(directory.project_id) if directory.project_id else None,
                            slugs=changes.created,
                            creator=directory.creator,
                        ),
                        scope=scope,
                    ))
                except Exception:
                    # A busy pool lands here too. In org mode nothing records
                    # these folders later: reconcile_conversation skips org
                    # roots and the owner backfill runs once, at startup, so
                    # they keep no index row and their owner stays unknown.
                    logger.error(
                        "[responses] could not record the artifacts remote turn %s created: %s",
                        conv_id, changes.created, exc_info=True, extra={"request_id": corr},
                    )
            return changes

        async def persist(*, clean: bool) -> _SavedAnswer:
            """Save the answer as it stands, once per turn, in one unit.

            The flag is set and the collected parts copied here, on the event
            loop, before the unit starts: the loop goes on appending to the
            lists while the unit's thread reads its copies, and a later call
            must not save the turn again. A turn whose question was never
            saved has nothing to answer, so it saves nothing.

            Tool rows only on a clean finish. They arrive before the terminal
            event, so a turn can carry rows and then fail or be cancelled,
            and this runs on those paths too.
            """
            nonlocal persisted
            if persisted or question_id is None:
                return _SavedAnswer()
            persisted = True
            save = partial(
                _save_answer,
                lifecycle=lifecycle,
                conversation_id=conv_id,
                question_id=question_id,
                text="".join(collected_text),
                events=tuple(collected_events),
                tool_rows=tuple(turn_rows) if clean else (),
                harness=harness_id,
            )
            try:
                async with conversation_writes(conv_id):
                    return _SavedAnswer(message_id=await run_db(save, scope=scope))
            except PoolTimeoutError as exc:
                logger.warning(
                    "[responses] no database connection freed in time to save the remote turn "
                    "for conversation %s", conv_id, extra={"request_id": corr},
                )
                return _SavedAnswer(failure=exc)
            except Exception as exc:
                logger.exception(
                    "[responses] failed to persist remote turn for conversation %s", conv_id,
                    extra={"request_id": corr},
                )
                return _SavedAnswer(failure=exc)

        async def finish(completed_frame: str | None) -> None:
            """Save the answer, then write the frame that reports it and the
            terminal record: completed with the saved row's id, or failed when
            the save failed, so the stream never says completed for an answer
            the database does not hold."""
            saved = await persist(clean=True)
            if lifecycle.discarded:
                # Deleted while it saved: its buffer is gone, and writing a
                # terminal record would recreate it for the next turn to tail.
                return
            if saved.failure is not None:
                await buffer.append("sse", {"sse": _failed_frame_for(saved.failure, request_id=corr)})
                await buffer.close("error")
                return
            if completed_frame is not None:
                await buffer.append("sse", {"sse": self._inject_completion_id(completed_frame, saved.message_id)})
            await buffer.close("completed")

        async def end_turn(
            reason: TerminalReason, failed_frame: Callable[[str | None], str] | None = None,
        ) -> None:
            """Record the pod's artifacts and save the answer as it stands,
            then write the failure frame, if any, carrying the saved row's id,
            and the terminal record. The cancel and error branches below run it
            under run_to_completion, so a second Stop or a shutdown that lands
            during a save waits for it, and the stream still ends the way this
            turn did."""
            await record_artifacts(completed_cleanly=False)
            saved = await persist(clean=False)
            if lifecycle.discarded:
                # Deleted while it saved: see finish().
                return
            if failed_frame is not None:
                # Interrupted and error endings keep their own frame and
                # terminal even when the save fails: an error keeps its own
                # code and reset_at, and boot recovery seals an `interrupted`
                # turn.
                await buffer.append("sse", {"sse": failed_frame(saved.assistant_message_id)})
            elif saved.failure is not None:
                # A Stop whose save failed says so, as finish() does.
                await buffer.append("sse", {"sse": _failed_frame_for(saved.failure, request_id=corr)})
                reason = "error"
            await buffer.close(reason)

        async def start() -> None:
            """The question's unit, under the conversation's write lock. Run
            under run_to_completion: a cancel that lands while the unit
            writes waits for it, so the branches below know the question it
            saved and finalize it instead of leaving it pending."""
            nonlocal question_id
            async with conversation_writes(conv_id):
                question_id = await run_db(
                    partial(
                        _save_remote_question,
                        lifecycle=lifecycle, conversation_id=conv_id, content=original_content,
                    ),
                    scope=scope,
                )

        async def replies_as_stream_events():
            nonlocal turn_artifacts
            from anton.core.llm.provider import StreamTaskProgress, StreamTextDelta
            from cowork.harnesses.anton_harness.stream_formatter import ArtifactCreated
            from cowork.services.task_objects import (
                publish_and_card_turn_artifacts,
                snapshot_artifact_state,
            )

            # Read once and held for the whole turn: the pod counts its
            # compaction against exactly the history seeded here, so
            # re-reading it when the reply arrives could map the count onto a
            # different list.
            seed = await run_db(partial(self._read_remote_turn_inputs, conv_id=conv_id), scope=scope)
            if seed.artifacts is not None:
                # The worker writes artifacts into the shared tree while this
                # turn runs, so cowork-server does the same before/after diff
                # it does for an in-process turn. Snapshotting here rather than
                # in the caller is what makes it a genuine "before":
                # stream_remote_replies below only enqueues the job once this
                # generator is first iterated.
                before_slugs, before_mtimes = snapshot_artifact_state(seed.artifacts.base)
                turn_artifacts = _RemoteTurnArtifacts(
                    directory=seed.artifacts, before_slugs=before_slugs, before_mtimes=before_mtimes,
                )
            # Set only on turn_completed: any other exit (Stop, cancel, failure)
            # may have cut anton off between writing an artifact's metadata and
            # appending its provenance, see `turn_created_slugs`.
            completed_cleanly = False

            async for kind, data in stream_remote_replies(
                conversation_id=str(conv_id),
                org_id=self.scope.org_id,
                user_id=self.scope.user_id,
                input_text=input_text,
                model=model,
                turn_id=turn_id,
                history=seed.history,
                # Global memory and skills use read-only mounts. Project
                # memory is outside the conversation workspace and therefore
                # travels as a bounded, sheddable wire block.
                memory=seed.memory,
                **seed.workspace,
                started_at=seed.started_at,
                correlation_id=corr,
                llm=(turn_llm or {}).get("llm"),
                turn_key_id=(turn_llm or {}).get("turn_key_id"),
                disabled=disabled,
                # Questions need someone to answer them: this path serves
                # the web UI, which renders the card and posts /answer.
                interactive=self.interactive and get_app_settings().ask_user_enabled,
                # This handler serves the cowork UI (scheduled turns show
                # there too), which renders a tool's message whether or
                # not it can answer questions. Channel turns build their
                # own request in turnqueue/remote_turn.py and leave it off.
                tool_messages=True,
            ):
                if kind == "progress" and data.get("phase") == "workspace_authorized":
                    if turn_artifacts is not None:
                        turn_artifacts.writes_allowed = data.get("workspace_mode") == "persistent"
                elif kind == "turn_delta":
                    yield StreamTextDelta(text=data.get("text", ""))
                elif kind == "turn_step":
                    for event in step_stream_events(data):
                        yield event
                elif kind == "turn_memory":
                    await save_memory(data.get("entries") or [])
                elif kind == "turn_history":
                    # Slice-assign, not extend: the pod emits one frame per
                    # turn, so a repeated frame must replace the rows rather
                    # than double them. Sanitized at the boundary — the pod
                    # is semi-trusted and these rows reach both the DB and
                    # every later turn's LLM context.
                    turn_rows[:] = sanitize_turn_history_rows(data.get("rows"))
                elif kind == "turn_compaction":
                    await save_compaction(data, seed.seed_info)
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
                    # `break`, not `return`: the record/publish/card block
                    # below must still run on a clean finish.
                    completed_cleanly = True
                    break
                elif kind == "turn_failed":
                    if await _remote_cancel_confirmed(data.get("error"), corr):
                        # A /cancel that reached a replica which doesn't
                        # own the producer sets the Redis flag only, so
                        # the controller is what discards the pod and it
                        # reports the stop as a turn_failed. Neither of
                        # its cancel literals reaches remote_turn_error,
                        # which would collapse them to anton_error.
                        # Route them through the same path a locally
                        # aborted turn takes: partial text persists, no
                        # response.failed frame, no error bubble on
                        # reload.
                        raise asyncio.CancelledError()
                    failure.update(data)
                    raise _RemoteTurnFailed()

            # The reply stream ended without a raise. Every other exit records
            # the artifacts in end_turn, with no publish and no cards, matching
            # the in-process path, where Stop/error produce no cards. Outside
            # org mode the next turn in the project reconciles the publish.
            changes = await record_artifacts(completed_cleanly=completed_cleanly)
            if changes is not None:
                for card in await publish_and_card_turn_artifacts(
                    turn_artifacts.directory.base,
                    new_slugs=changes.created,
                    touched_slugs=changes.touched,
                    scope=scope,
                    project_id=turn_artifacts.directory.project_id,
                    project_name=turn_artifacts.directory.project_name,
                ):
                    yield ArtifactCreated(card)

        try:
            # Stage attachments + project instructions before the pod runs.
            try:
                await run_db(partial(self._stage_remote_workspace_files, conv_id=conv_id), scope=scope)
            except PoolTimeoutError:
                # A full pool is not a missing file: the turn is refused
                # (server_busy) rather than run without what the user sent.
                raise
            except Exception:
                # Staging never fails the turn (_stage_remote_workspace_files
                # degrades on its own); this is its unit failing to connect or
                # to commit.
                logger.exception(
                    "[responses] failed to stage workspace files for conversation %s", conv_id,
                )
            # Save the question (pending) as this producer's first write
            # (ENG-1231); see the note in handle(). Committed before
            # streaming, so a refresh/reconnect mid-turn shows the question
            # via /items.
            await run_to_completion(start())
            first = True
            completed_frame: str | None = None
            async for sse in format_responses_stream(
                replies_as_stream_events(), model or "", event_sink,
            ):
                if first:
                    # The formatter's created frame lacks conversation_id +
                    # harness; inject them like the in-process path does.
                    sse = self._inject_created(sse, conv_id, harness_id, question_id)
                    first = False
                if self._sse_event_type(sse) == "response.completed":
                    # The formatter yields its terminal frame last, once its
                    # source is exhausted, so event_sink has already seen
                    # everything this turn produced. finish() sends it once
                    # the answer is saved, carrying the saved row's id.
                    completed_frame = sse
                    continue
                await buffer.append("sse", {"sse": sse})
            # A Stop that lands here waits for the save and its frame, so the
            # stream's terminal record always matches what the database holds.
            await run_to_completion(finish(completed_frame))
        except _RemoteTurnFailed:
            message = failure.get("message") or GENERIC_TURN_ERROR_MESSAGE
            code = failure.get("code") or GENERIC_TURN_ERROR_CODE
            # At WARNING because that is the floor the deployed environments
            # run at: the id below is what the client shows the user as a
            # Reference, so a line carrying it has to survive prod's log level
            # or support has nothing to search for.
            logger.warning(
                "[responses] remote turn reported a failure for conversation %s "
                "correlation_id=%s code=%s", conv_id, corr, code,
                extra={"request_id": corr},
            )
            # The producer keeps `reset_at` only for RESET_AT_CODES and only as
            # an offset-aware instant (`_remote_reset_at` in producer.py), so
            # it rides both the frame and the persisted event as sent.
            reset_at = failure.get("reset_at")
            # The pod never got to retire a question it was blocked on.
            collected_events.extend(cancelled_ask_user_retirements(collected_events))
            collected_events.append(response_failed_payload(
                message, code, reset_at=reset_at, request_id=corr))

            async def fail() -> None:
                if code in CONTENT_REPAIR_CODES:
                    # ENG-1992: the remote/org path's twin of the streaming
                    # handler's repair. producer.py already classified this
                    # via remote_turn_error from the pod's scrubbed error
                    # string, so `code` alone is enough to act on here.
                    # Repaired before the terminal record, so the next
                    # question into this conversation reads the repaired
                    # history. Never lets a repair failure mask the turn's
                    # real outcome.
                    try:
                        async with conversation_writes(conv_id):
                            repaired = await run_db(
                                lambda session: ConversationService(session).repair_image_content(conv_id),
                                scope=scope,
                            )
                        logger.warning(
                            "[responses] content validation error on remote conversation %s — "
                            "repaired %d message(s) with image content: %s",
                            conv_id, len(repaired), failure.get("error"),
                        )
                    except Exception:
                        logger.exception(
                            "[responses] failed to repair conversation %s after remote content "
                            "validation error", conv_id,
                        )
                await end_turn("error", lambda message_id: response_failed_sse(
                    message, code, reset_at=reset_at, request_id=corr,
                    assistant_message_id=message_id,
                ))

            await run_to_completion(fail())
        except asyncio.CancelledError:
            if lifecycle.discarded:
                # Same reasoning as _run_turn's discarded branch — see there.
                # The artifacts the pod wrote are still recorded, as the
                # in-process harness's `finally` records them.
                logger.info("[responses] discarded remote turn %s — not persisting", conv_id)
                await record_artifacts(completed_cleanly=False)
                return
            if buffer.is_closed:
                # The cancel arrived while finish() saved the answer and wrote
                # the terminal record, and waited for both. The turn is over.
                return
            if lifecycle.shutting_down or lifecycle.timed_out:
                collected_events.extend(cancelled_ask_user_retirements(collected_events))
                collected_events.append(response_failed_payload(
                    INTERRUPTED_TURN_MESSAGE, GENERIC_TURN_ERROR_CODE, request_id=corr,
                ))
                await run_to_completion(end_turn("interrupted", lambda message_id: response_failed_sse(
                    INTERRUPTED_TURN_MESSAGE, GENERIC_TURN_ERROR_CODE, request_id=corr,
                    assistant_message_id=message_id,
                )))
                return
            # Partial text generated before cancellation is persisted, with
            # every question the Stop left open retired.
            collected_events.extend(cancelled_ask_user_retirements(collected_events))
            await run_to_completion(end_turn("cancelled"))
        except Exception as exc:
            if isinstance(exc, PoolTimeoutError):
                # No database connection freed in time. The frame says so with
                # the wait, as a refused request's 503 does.
                code, message = friendly_turn_error(exc)
                retry_after = busy_retry_seconds()
                extra: dict = {"retry_after": retry_after, "retry_at": retry_at_instant(retry_after)}
                logger.warning(
                    "[responses] remote turn for conversation %s correlation_id=%s found no free "
                    "database connection", conv_id, corr, extra={"request_id": corr},
                )
            else:
                code, message, extra = GENERIC_TURN_ERROR_CODE, GENERIC_TURN_ERROR_MESSAGE, {}
                logger.exception(
                    "[responses] remote turn failed for conversation %s correlation_id=%s",
                    conv_id, corr, extra={"request_id": corr},
                )
            collected_events.extend(cancelled_ask_user_retirements(collected_events))
            collected_events.append(response_failed_payload(message, code, request_id=corr, **extra))
            # Saved before the frame is built: a client's SSE reader stops
            # at response.failed, so any id has to ride this frame.
            await run_to_completion(end_turn("error", lambda message_id: response_failed_sse(
                message, code, request_id=corr, assistant_message_id=message_id, **extra,
            )))
        finally:
            await _seal_unterminated_buffer(
                buffer, lifecycle, conv_id, request_id=corr
            )

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
        reasoning_effort: str | None = None,
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

        Its database work runs as units (cowork.db.units), so it holds no
        pooled connection while the model answers. The first unit saves the
        question (pending) so a mid-turn refresh shows it, and loads the
        conversation for the harness, which reads history (get_ordered_messages
        leaves the pending row out, so the current input isn't double-fed) in
        units of its own. The last unit saves the assistant turn and clears
        the question's pending flag in one commit (ENG-1231). Both hold the
        conversation's write lock.
        Never reaches the HTTP response: readers tail the buffer.
        """
        lifecycle = lifecycle if lifecycle is not None else TurnLifecycle()
        # Scoped from the immutable principal captured at handler
        # construction, never from request state.
        scope = scope_from_principal(self.principal)
        collected_text: list[str] = []
        collected_events: list[dict] = []
        turn_rows: list[dict] = []
        persisted = False
        # Send time captured before the turn
        sent_at = datetime.now(timezone.utc)
        question_id: UUID | None = None
        # The in-process twin of _produce_remote's correlation id. Minted up
        # front so the failure branch and the seal below attach the SAME id the
        # log line carries. This path has no pod and so no correlation id of its
        # own to reuse — a desktop turn's only trace is the local log, which is
        # exactly the report this id exists to make possible.
        corr = str(uuid4())

        def event_sink(event_type: str, data: dict) -> None:
            # Tool block-rows are for LLM-history persistence, not UI replay —
            # keep them out of the events log the client rebuilds from.
            if event_type == "response.turn_history":
                # Id-checked even though we produced these ourselves: an
                # unreplayable id here is permanent for the conversation, and
                # the installed anton can be older than this server (ENG-2420).
                turn_rows[:] = reject_unreplayable_tool_rows(data.get("rows") or [])
                return
            collected_events.append(data)
            accumulate_answer_text(collected_text, event_type, data)

        async def persist() -> _SavedAnswer:
            """Save the answer as it stands, once per turn, in one unit.

            The flag is set and the collected parts copied here, on the event
            loop, before the unit starts: the loop goes on appending to the
            lists while the unit's thread reads its copies, and a later call
            must not save the turn again. A turn whose question was never
            saved has nothing to answer, so it saves nothing.
            """
            nonlocal persisted
            if persisted or question_id is None:
                return _SavedAnswer()
            persisted = True
            save = partial(
                _save_answer,
                lifecycle=lifecycle,
                conversation_id=conv_id,
                question_id=question_id,
                text="".join(collected_text),
                events=tuple(collected_events),
                tool_rows=tuple(turn_rows),
                harness=harness_id,
            )
            try:
                async with conversation_writes(conv_id):
                    return _SavedAnswer(message_id=await run_db(save, scope=scope))
            except PoolTimeoutError as exc:
                logger.warning(
                    "[responses] no database connection freed in time to save the turn for conversation %s",
                    conv_id, extra={"request_id": corr},
                )
                return _SavedAnswer(failure=exc)
            except Exception as exc:
                logger.exception(
                    "[responses] failed to persist turn for conversation %s", conv_id,
                    extra={"request_id": corr},
                )
                return _SavedAnswer(failure=exc)

        async def finish(completed_frame: str | None) -> None:
            """Save the answer, then write the frame that reports it and the
            terminal record: completed with the saved row's id, or failed when
            the save failed, so the stream never says completed for an answer
            the database does not hold."""
            saved = await persist()
            if lifecycle.discarded:
                # Deleted while it saved: its buffer is gone, and writing a
                # terminal record would recreate it for the next turn to tail.
                return
            if saved.failure is not None:
                await buffer.append("sse", {"sse": _failed_frame_for(saved.failure, request_id=corr)})
                await buffer.close("error")
                return
            if completed_frame is not None:
                await buffer.append("sse", {"sse": self._inject_completion_id(completed_frame, saved.message_id)})
            await buffer.close("completed")

        async def end_turn(
            reason: TerminalReason, failed_frame: Callable[[str | None], str] | None = None,
        ) -> None:
            """Save the answer as it stands, then write the failure frame, if
            any, carrying the saved row's id, and the terminal record. The
            cancel and error branches below run it under run_to_completion, so
            a second Stop or a shutdown that lands during the save waits for
            it, and the stream still ends the way this turn did."""
            saved = await persist()
            if lifecycle.discarded:
                # Deleted while it saved: see finish().
                return
            if failed_frame is not None:
                # Interrupted and error endings keep their own frame and
                # terminal even when the save fails: boot recovery seals an
                # `interrupted` turn, and an error keeps its own code.
                await buffer.append("sse", {"sse": failed_frame(saved.assistant_message_id)})
            elif saved.failure is not None:
                # A Stop whose save failed says so, as finish() does.
                await buffer.append("sse", {"sse": _failed_frame_for(saved.failure, request_id=corr)})
                reason = "error"
            await buffer.close(reason)

        async def start() -> Conversation | None:
            """The turn's first unit, under the conversation's write lock.
            Run under run_to_completion: a cancel that lands while the unit
            writes waits for it, so the branches below know the question it
            saved and finalize it instead of leaving it pending."""
            nonlocal question_id
            async with conversation_writes(conv_id):
                started = await run_db(
                    partial(
                        _start_turn, lifecycle=lifecycle, conversation_id=conv_id,
                        content=original_content, sent_at=sent_at,
                    ),
                    scope=scope,
                )
            if started is None:
                return None
            question_id = started.question_id
            return started.conversation

        try:
            conversation = await run_to_completion(start())
            if conversation is None:
                return
            harness = get_harness(harness_name)
            stream = harness.stream_response(
                conversation=conversation, input=harness_input, model=model,
                reasoning_effort=reasoning_effort, disabled_connections=disabled,
                trace_tags=trace_tags, trace_metadata=trace_metadata,
                # The cowork UI (scheduled turns show there too) renders a
                # tool's message to the user, as on the remote path.
                tool_messages=True,
            )
            event_count = 0
            completed_frame: str | None = None
            async for sse_string in harness.formatter(stream, model, event_sink):
                event_count += 1
                sse_string = self._inject_created(sse_string, conv_id, harness_id, question_id)
                if self._sse_event_type(sse_string) == "response.completed":
                    # The formatter yields its terminal frame last, once its
                    # source is exhausted, so event_sink has already seen
                    # everything this turn produced. finish() sends it once
                    # the answer is saved, carrying the saved row's id.
                    completed_frame = sse_string
                    continue
                await buffer.append("sse", {"sse": sse_string})
            logger.info("[responses] turn %s finished — %d events", conv_id, event_count)
            # A Stop that lands here waits for the save and its frame, so the
            # stream's terminal record always matches what the database holds.
            await run_to_completion(finish(completed_frame))
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
            if buffer.is_closed:
                # The cancel arrived while finish() saved the answer and wrote
                # the terminal record, and waited for both. The turn is over.
                return
            if lifecycle.shutting_down or lifecycle.timed_out:
                collected_events.extend(cancelled_ask_user_retirements(collected_events))
                collected_events.append(response_failed_payload(
                    INTERRUPTED_TURN_MESSAGE, GENERIC_TURN_ERROR_CODE, request_id=corr,
                ))
                await run_to_completion(end_turn("interrupted", lambda message_id: response_failed_sse(
                    INTERRUPTED_TURN_MESSAGE, GENERIC_TURN_ERROR_CODE, request_id=corr,
                    assistant_message_id=message_id,
                )))
                return
            # Nothing special is emitted on cancellation.
            # The partial text and events generated before cancellation are persisted.
            # A question that was on screen when Stop was pressed never got its
            # `response.ask_user_answered` (see cancelled_ask_user_retirements),
            # so retire it here — otherwise the persisted log holds a published
            # question that nothing in it ever closes.
            collected_events.extend(cancelled_ask_user_retirements(collected_events))
            await run_to_completion(end_turn("cancelled"))
            return
        except Exception as exc:
            # Resolve the model-403 info once and hand it to friendly_turn_error
            # so it isn't computed twice on this path (reused by the extras below).
            model_info = model_unavailable_info(exc)
            friendly = friendly_turn_error(exc, model_info=model_info)
            if friendly is not None:
                code, message = friendly
                logger.info(
                    "[responses] user-facing turn error: %s", exc,
                    extra={"request_id": corr},
                )
            else:
                code, message = GENERIC_TURN_ERROR_CODE, GENERIC_TURN_ERROR_MESSAGE
                # WARNING or above is the floor the deployed environments run
                # at, and this is the branch whose message tells the user
                # nothing — so the id has to survive that level or the
                # Reference they quote resolves to no line.
                logger.exception(
                    "[responses] turn failed for conversation %s correlation_id=%s",
                    conv_id, corr, extra={"request_id": corr},
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
                    provider = _auth_failure_provider(
                        get_user_settings(), getattr(exc, "role", None)
                    )
                    # An unattributable failure keeps the generic auth copy with
                    # no provider fields, rather than naming a provider that may
                    # not be the one that failed.
                    if provider is not None:
                        reconnectable = provider == Provider.MINDS_CLOUD
                        message = auth_error_detail(provider.label, reconnectable)
                        extra = {"reconnectable": reconnectable, "provider_label": provider.label}
                except Exception:
                    logger.exception(
                        "[responses] could not resolve provider for auth error",
                        extra={"request_id": corr},
                    )
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
            elif code in RESET_AT_CODES:
                # When the free way forward comes back: the spent allowance
                # refills, or the free-Air fuse resets at the end of the UTC
                # day. The gate sends it on these denials and not on a
                # velocity one, so the card can offer waiting as a real
                # alternative to paying instead of only asking for money.
                _reset = gate_reset_at(exc)
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
                    logger.exception(
                        "[responses] could not resolve the retry hint",
                        extra={"request_id": corr},
                    )
            elif code == SERVER_BUSY_CODE:
                # No database connection freed in time. The same time-gated
                # Retry as rate_limited, with the pool's wait as the interval.
                retry_after = busy_retry_seconds()
                extra = {"retry_after": retry_after, "retry_at": retry_at_instant(retry_after)}
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
                    logger.exception(
                        "[responses] could not resolve provider for overload error",
                        extra={"request_id": corr},
                    )
            # Set after the branches above, each of which REPLACES `extra`
            # rather than adding to it — seeding it earlier would survive only
            # the unmapped path. Carried on every failure so the payload shape
            # stays uniform, but only the unmapped branch tags its log line
            # with the id, so a curated failure can reach the log with nothing
            # to match a quoted reference against. The client renders it on
            # the generic card alone, so there is nothing to quote for one.
            extra["request_id"] = corr
            failed = response_failed_payload(message, code, **extra)
            # Append to collected_events BEFORE persisting — persist()
            # reads collected_events to build the assistant row, and this failure
            # event must land in it exactly as it did before this frame carried
            # an id; only the append/persist order relative to the SSE frame
            # below actually changed.
            collected_events.append(failed)

            async def fail() -> None:
                """Repair the stored history when the turn's error asks for
                it, then save the answer and write its frame. One step under
                run_to_completion, so a Stop that lands during the repair
                waits for both and the turn still ends with its own error."""
                if code in CONTENT_REPAIR_CODES:
                    # ENG-1992: the provider permanently rejected an image block in
                    # this conversation's stored history — repair the DATA once,
                    # here, rather than special-case every future replay. Never
                    # lets a repair failure mask the turn's real outcome; the
                    # terminal frame below goes out either way.
                    try:
                        async with conversation_writes(conv_id):
                            repaired = await run_db(
                                lambda session: ConversationService(session).repair_image_content(conv_id),
                                scope=scope,
                            )
                        logger.warning(
                            "[responses] content validation error on conversation %s — "
                            "repaired %d message(s) with image content: %s",
                            conv_id, len(repaired), exc, extra={"request_id": corr},
                        )
                    except Exception:
                        logger.exception(
                            "[responses] failed to repair conversation %s after content validation error",
                            conv_id, extra={"request_id": corr},
                        )
                await end_turn("error", lambda message_id: response_failed_sse(
                    message, code, **extra, assistant_message_id=message_id,
                ))

            await run_to_completion(fail())
        finally:
            await _seal_unterminated_buffer(buffer, lifecycle, conv_id, request_id=corr)

    @staticmethod
    def _sse_event_type(sse_string: str) -> str | None:
        """The frame's own `event:` line, e.g. "response.completed" — every
        frame in this codebase is built by `sse_frame`/this same f-string
        shape, always `event: {type}\\n` as the first line (streaming/sse.py).

        Never substring-search the full SSE text for a frame type: the JSON
        payload can carry untrusted model output (a delta chunk, an error
        message) that happens to contain a frame-type-looking string, which
        would otherwise make a plain delta frame match "response.completed"
        and get rewritten into one — persisting a partial turn early and
        mislabeling the real event to the client."""
        first_line = sse_string.strip().split("\n", 1)[0]
        prefix = "event: "
        return first_line[len(prefix):].strip() if first_line.startswith(prefix) else None

    @classmethod
    def _inject_created(
        cls,
        sse_string: str,
        conversation_id: UUID,
        harness_id: str | None,
        user_message_id: UUID | None = None,
    ) -> str:
        """Inject conversation_id + harness into the response.created event so
        the client learns the canonical id and which agent generated this.

        `user_message_id` is the persisted user Message's id, carried on the
        same frame. The client appends the user's row optimistically on send,
        so without this it holds a row with no id until a later refetch — and
        every consumer keyed on message id (turn delete, the step sidecar, the
        usage-notice anchor) silently degrades for the live turn. Omitted, not
        null, on a producer that persists no user row (the probe path).
        """
        if cls._sse_event_type(sse_string) != "response.created":
            return sse_string
        try:
            lines = sse_string.strip().split("\n")
            data_line = next(line for line in lines if line.startswith("data:"))
            payload = json.loads(data_line[5:])
            if "conversation_id" in payload:
                return sse_string
            payload["conversation_id"] = str(conversation_id)
            if harness_id:
                payload["harness"] = harness_id
            if user_message_id is not None:
                payload["user_message_id"] = str(user_message_id)
            return f"event: response.created\ndata: {json.dumps(payload)}\n\n"
        except Exception:
            return sse_string

    @classmethod
    def _inject_completion_id(cls, sse_string: str, assistant_message_id: UUID | None) -> str:
        """Inject the persisted assistant message's id into a formatter-built
        response.completed frame, at the frame root (sibling to
        `type`/`response`, not nested inside `response.output`) — the client
        reads it from there the same way `_inject_created` places
        conversation_id/harness. Persistence for this turn must already have
        happened by the time this is called; a formatter never yields
        response.completed before its source is exhausted, so `event_sink`
        has already seen every delta and event this turn produced.

        A turn that persisted nothing (an early-return in save_assistant_turn)
        passes assistant_message_id=None and the field is simply omitted,
        same convention as response_failed_payload's optional fields."""
        if cls._sse_event_type(sse_string) != "response.completed":
            return sse_string
        try:
            lines = sse_string.strip().split("\n")
            data_line = next(line for line in lines if line.startswith("data:"))
            payload = json.loads(data_line[5:])
            if assistant_message_id is not None:
                payload["assistant_message_id"] = str(assistant_message_id)
            return f"event: response.completed\ndata: {json.dumps(payload)}\n\n"
        except Exception:
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
                # Id-checked even though we produced these ourselves: an
                # unreplayable id here is permanent for the conversation, and
                # the installed anton can be older than this server (ENG-2420).
                turn_rows[:] = reject_unreplayable_tool_rows(data.get("rows") or [])
                return
            collected_events.append(data)
            accumulate_answer_text(collected_text, event_type, data)

        try:
            async for _ in self._get_harness().formatter(stream, model, event_sink):
                pass
        except PoolTimeoutError:
            # No database connection freed in time: the app answers 503 with
            # the wait, not a turn failure.
            raise
        except Exception as exc:
            # Mirror the streaming path: a recognised failure (e.g. an
            # unsupported image) surfaces its curated message with a 400;
            # anything else stays a generic 500 so provider internals never
            # leak. (cowork PR #156.)
            # Minted here rather than up front like the streaming twin: this
            # path has no seal to feed, so a successful turn needs no id.
            corr = str(uuid4())
            friendly = friendly_turn_error(exc)
            if friendly is not None:
                code, message = friendly
                logger.info(
                    "[responses] user-facing turn error: %s", exc,
                    extra={"request_id": corr},
                )
                if code in CONTENT_REPAIR_CODES:
                    # ENG-1992: see the streaming path's twin for the full
                    # rationale — repair the conversation's stored history
                    # once here rather than special-case every future replay.
                    try:
                        async with conversation_writes(conversation_id):
                            repaired = await run_db(
                                lambda session: ConversationService(session).repair_image_content(conversation_id),
                                scope=self.scope,
                            )
                        logger.warning(
                            "[responses] content validation error on conversation %s — "
                            "repaired %d message(s) with image content: %s",
                            conversation_id, len(repaired), exc,
                            extra={"request_id": corr},
                        )
                    except Exception:
                        logger.exception(
                            "[responses] failed to repair conversation %s after content validation error",
                            conversation_id, extra={"request_id": corr},
                        )
                # The ladder already produced a code; carry it instead of dropping
                # it here. Same wire shape the streaming twin emits.
                raise HTTPException(
                    status_code=400,
                    detail=response_failed_payload(message, code, request_id=corr),
                )
            logger.exception(
                "[responses] turn failed for conversation %s correlation_id=%s",
                conversation_id, corr, extra={"request_id": corr},
            )
            # Same shape as the 400 above: one body for every turn failure, so a
            # caller never has to branch on status to know how to read `detail`.
            raise HTTPException(
                status_code=500,
                detail=response_failed_payload(
                    GENERIC_TURN_ERROR_MESSAGE, GENERIC_TURN_ERROR_CODE, request_id=corr,
                ),
            )

        assistant_text = "".join(collected_text)
        harness_id = getattr(self._get_harness(), "id", None)

        # Persist the user message now — after the harness has read history for
        # this turn — so it isn't replayed into the turn as duplicate context.
        def save_turn(session: ScopedSession) -> UUID:
            service = ConversationService(session)
            user_message = service.save_user_message(
                conversation_id, original_content, created_at=sent_at,
            )
            assistant_message = service.save_assistant_turn(
                conversation_id, assistant_text, collected_events,
                harness=harness_id, tool_rows=turn_rows,
            )
            return _turn_anchor_id(user_message, assistant_message)

        async with conversation_writes(conversation_id):
            anchor_id = await run_db(save_turn, scope=self.scope)

        return Response(
            status=ResponseStatus.completed,
            model=model,
            output=[self._build_output(str(anchor_id), assistant_text)],
        )

    def _build_harness_input(self, request: ResponsesRequest, *, session: ScopedSession) -> list[dict]:
        blocks: list[dict] = []

        # Resolve attachment_ids to image/file blocks
        if request.attachment_ids:
            file_svc = FileService(session)
            for aid in request.attachment_ids:
                try:
                    content_type, filename, filepath = file_svc.get_file_content(UUID(aid))
                except PoolTimeoutError:
                    # A full pool is not a missing attachment: dropping it would
                    # run the turn without the file the user sent.
                    raise
                except Exception:
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
                                        content_type, filename, filepath = FileService(session).get_file_content(UUID(item.file_id))
                                    except ValueError:
                                        raise HTTPException(status_code=404, detail=f"File {item.file_id!r} not found")
                                    if content_type and content_type.startswith("image/"):
                                        blocks.append(self._image_block(filepath, content_type))
                                    else:
                                        blocks.append({"type": "file", "path": str(filepath), "filename": filename})
                    break

        return blocks or [{"type": "text", "text": ""}]

    def _relink_attachments(
        self, client_session_id: str, conversation, *, session: ScopedSession,
    ) -> None:
        """Repoint attachments uploaded against a client-side session id to
        the conversation that actually got created, so the Task Uploads
        rail (which queries by the live conversation id) still finds them."""
        from cowork.services.files import attachment_purpose

        moved = FileService(session).relink_purpose(
            attachment_purpose(client_session_id),
            attachment_purpose(str(conversation.id)),
        )
        if moved:
            logger.info(
                "[responses] relinked %d attachment(s) from client session %r to conversation %s",
                moved, client_session_id, conversation.id,
            )

    def _resolve_project_id(self, request: ResponsesRequest, *, session: ScopedSession) -> UUID:
        """Project for a conversation being CREATED this turn.

        Only called on the creation paths: an existing conversation already
        pins its project via conversation.project_id, and the client-held
        name it echoes can be stale after a project rename — resolving it
        eagerly used to 404 every later turn of the task (ENG-1028).
        """
        service = ProjectService(session)
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
    ends the stream — the harness's own response.completed/failed frame was
    already written as a normal record. User Stop has no such frame, so its
    terminal is turned into ``response.cancelled`` here.

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
                if rec.data.get("reason") == "cancelled":
                    yield sse_frame("response.cancelled", {"type": "response.cancelled"})
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
