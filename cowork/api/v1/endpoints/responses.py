"""Responses API endpoints (OpenAI-compatible Responses API).

Streaming turns run **detached**: POST /responses starts a background
producer (see handlers.responses + cowork.streaming) that writes to a
per-turn buffer; the response tails that buffer. By default, closing the
connection does NOT stop the run — the client reconnects via GET
/responses/tail with a `from_seq` cursor and resumes from where it left off.
An originating POST with cancel_on_disconnect=true owns its run and cancels
that instance on transport loss. Tail readers never own the run. Explicit
POST /responses/cancel remains available with an optional run-instance guard.
"""
import time
from typing import Annotated, NamedTuple
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, StringConstraints
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse, Response as HTTPResponse

from cowork.api.v1.permissions import AuthenticatedInOrgMode, OpenByDesign, require
from cowork.common.logger import setup_logging
from cowork.common.settings.app_settings import TurnQueueSettings
from cowork.db.personal_resources import PersonalResource
from cowork.db.scoped import (
    MissingTenantScopeError,
    TenantScope,
    get_tenant_scope,
    scope_from_principal,
)
from cowork.handlers.responses import ResponsesHandler, sse_from_buffer
from cowork.principal import Principal, get_principal
from cowork.schemas.responses import ResponsesRequest
from cowork.streaming import RunHandle, TurnInProgress, registry
from cowork.streaming.answers import SubmitResult, broker
from cowork.streaming.backend import get_backend
from cowork.streaming.cancellation import cancel_disconnected_run, cancel_run, request_cancel
from cowork.streaming.disconnect import DisconnectStreamingResponse, DisconnectWatch
from cowork.streaming.buffer import RedisStreamBuffer
from cowork.streaming.turn_index import TurnOwner, get_turn, list_turns
from cowork.turnqueue.answers import RemoteAnswerResult, submit_remote_answer
from cowork.turnqueue.redis_client import (
    STALLED_CANCEL_REASON,
    get_redis,
)


logger = setup_logging()

# Declared per route, not on the router: options_handler below has to be
# genuinely open, and a route-level dependency is ADDED to its router's rather
# than substituted for it — a router-level AuthenticatedInOrgMode would still
# run on the preflight and 401 it, because TrustedHeaderMiddleware builds no
# Principal for an OPTIONS request. Every route that carries
# AuthenticatedInOrgMode below already fails closed on its own too
# (_require_streaming_scope in this module raises MissingTenantScopeError ->
# 401 exactly like ScopedSession, and POST / scopes its own sessions
# inside ResponsesHandler); declaring it makes that visible to the route
# walker instead of only to someone reading those checks.
router = APIRouter()
TenantScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


def _require_streaming_scope(scope: TenantScope) -> None:
    """Fail closed BEFORE any registry access.

    Where conversations are partitioned (org mode, or a personal scope that
    keeps them per person) a caller with no org in scope gets 401 — exactly
    like every ScopedSession-backed endpoint. Checked up front, independent of
    whether the target turn exists, so an empty list or an unknown
    conversation_id can't mask the missing-identity case behind a 200/404.
    """
    if scope.partitioned_for(PersonalResource.CONVERSATIONS) and scope.org_id is None:
        raise MissingTenantScopeError("streaming access requires an org in scope")


def _authorized_handle(
    handle: RunHandle | None, scope: TenantScope
) -> RunHandle | None:
    """The run handle the caller is allowed to touch, or None.

    A conversation_id is not an authorization token — the registry is keyed by
    it and would otherwise hand any authenticated caller another org's live
    turn. Where conversations are shared (local mode, the operator view, or a
    personal scope that keeps them shared) it never filters. Where they are
    partitioned, a handle owned by a different org/partition or user reads as
    absent, never 403 (no existence leak). Assumes the scope was already
    validated by _require_streaming_scope.
    """
    if handle is None:
        return None
    owner = TurnOwner(org_id=handle.org_id, user_id=handle.user_id)
    return handle if _owner_matches(owner, scope) else None


def _owner_matches(owner: TurnOwner, scope: TenantScope) -> bool:
    """Whether the caller may touch a turn ``owner`` started.

    Turns are personal — a conversation belongs to its creator — so wherever
    conversations are partitioned BOTH the org (or person's partition) and the
    owning user must match. Matching the org alone let any member tail/cancel/
    answer a teammate's live turn and read its prompts and output (staging
    audit P0). Local mode never filters (single user), and neither does a
    personal scope that keeps conversations shared, nor the operator view of a
    trusted-identity deployment (LOCAL_SCOPE). Shared by the local-handle check
    and the Redis turn index so the two can't disagree.
    """
    if not scope.partitioned_for(PersonalResource.CONVERSATIONS):
        return True   # today's single-user behavior
    return owner.org_id == scope.org_id and owner.user_id == scope.user_id


class SharedTurn(NamedTuple):
    """A turn read from Redis, and whether it is still going.

    ``in_flight`` is not simply ``not buffer.is_closed``: an empty stream reads
    as closed, which is right for a truncated conversation and wrong for a turn
    enqueued moments ago.
    """

    index: dict
    buffer: RedisStreamBuffer
    in_flight: bool


async def _shared_turn(
    conversation_id: str | None, scope: TenantScope
) -> SharedTurn | None:
    """The conversation's current turn read from Redis, or None.

    Used when this replica has no local handle, which on the Redis backend
    means only that some other replica started the turn. There is nothing to
    route: the buffer is readable from here. Returns None for an unknown
    conversation and for one owned by another org, so a foreign id is
    indistinguishable from a missing one.
    """
    if not conversation_id or get_backend() != "redis":
        return None
    turn = await get_turn(conversation_id)
    if turn is None or not _owner_matches(TurnOwner.from_index(turn), scope):
        return None
    buf = RedisStreamBuffer(conversation_id=conversation_id, turn_id=int(turn["turn_id"]))
    await buf.refresh()
    # Admission publishes the index before starting the producer. Until the
    # grace period is up, an empty stream means "not started yet" rather than
    # "over" (a remote worker may still be starting).
    starting = buf.latest_seq == 0 and _just_started(turn)
    return SharedTurn(index=turn, buffer=buf, in_flight=(not buf.is_closed) or starting)


def _just_started(turn: dict) -> bool:
    """Whether the turn was recorded too recently for an empty stream to mean
    anything. Enqueue to first record covers pod startup, so this is generous."""
    try:
        return (time.time() - float(turn.get("started_at") or 0)) < TURN_START_GRACE_SECONDS
    except (TypeError, ValueError):
        return False


# Enqueue to first record: the controller has to lock the conversation, find or
# create a pod (up to two minutes), and get the first event out of it.
TURN_START_GRACE_SECONDS = 180


async def _request_cancel(correlation_id: str, *, stalled: bool = False) -> None:
    """Ask whoever is running this turn to stop.

    The controller checks this key while it streams. Writing it rather than
    messaging a replica is what makes cancel work regardless of which replica
    the request landed on.

    The cause goes in the same transaction as the flag: a stall sets it, any
    other cancel clears it. scratchpad-controller's ``_clear_cancel`` deletes
    the flag right after it publishes the cancelled reply, so the replica that
    owns the turn reads the cause, not the flag, when that reply arrives. One
    transaction means the latest cancel decides: a failed write leaves no
    stray cause behind, and a Stop after a stall saves as a Stop.
    """
    await request_cancel(
        correlation_id, reason=STALLED_CANCEL_REASON if stalled else None, client=get_redis(),
    )


# no-store (not just no-cache): a chat stream can carry secrets the model
# echoed (e.g. a raw API key embedded in generated scratchpad code), so it
# must never be written to a client's on-disk HTTP cache. See ENG-462.
_SSE_HEADERS = {
    "Cache-Control": "no-store",
    "Connection": "keep-alive",
    "Access-Control-Allow-Origin": "*",
}


# OpenByDesign, standalone reason: hardcoded CORS-preflight response, no
# identity or data involved.
@router.options("/", dependencies=[Depends(require(OpenByDesign))])
async def options_handler():
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )


def _refuse_while_answering(*, conversation: str | None, scope: TenantScope) -> None:
    """Refuse a question into a conversation whose turn is still answering.

    Runs before any database read and before the routing gate, so the refusal
    costs nothing. RunRegistry.start repeats the check under its lock for a
    request that gets past this one while the running turn is still in its
    own gate. Where conversations are partitioned only the caller's own turn
    counts, so the refusal never reveals someone else's.
    """
    if not conversation:
        return
    try:
        conversation_id = str(UUID(conversation))
    except ValueError:
        return  # not a conversation id: handle() starts a new conversation
    _require_streaming_scope(scope)
    handle = _authorized_handle(registry.get(conversation_id), scope)
    if handle is not None and handle.is_answering:
        raise TurnInProgress(conversation_id=conversation_id, turn_id=handle.turn_id)


@router.post("/", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def responses(
    responses_request: ResponsesRequest,
    request: Request,
    principal: Principal | None = Depends(get_principal),
):
    # No request session: the handler runs its reads as short units, so no
    # connection stays checked out for the stream this returns.
    _refuse_while_answering(
        conversation=responses_request.conversation, scope=scope_from_principal(principal)
    )
    handler = ResponsesHandler(principal=principal)
    if responses_request.cancel_on_disconnect:
        watch = DisconnectWatch(request.receive)
        try:
            result = await watch.prepare(handler.handle(responses_request))
        except ClientDisconnect:
            await watch.close()
            await cancel_disconnected_run(handler.last_run_handle)
            return HTTPResponse(status_code=204)
        except BaseException:
            await watch.close()
            raise
        handle = handler.last_run_handle
        return DisconnectStreamingResponse(
            result, watch=watch,
            on_disconnect=lambda: cancel_disconnected_run(handle),
            media_type="text/event-stream", headers=_SSE_HEADERS,
        )
    result = await handler.handle(responses_request)
    if responses_request.stream:
        # `result` is sse_from_buffer(buffer, 0); the producer is already
        # running detached in the registry.
        return StreamingResponse(result, media_type="text/event-stream", headers=_SSE_HEADERS)
    return result


@router.get("/in-flight-list", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def in_flight_list(scope: TenantScopeDep):
    """Conversations with a turn running. The renderer uses it to sync stream
    state across clients/boots. Scoped to the caller's org so it can't
    enumerate another org's live conversation ids.

    Local handles first, then the Redis index for turns other replicas
    started. On the file backend the index is empty and this is the old
    in-memory lookup.
    """
    _require_streaming_scope(scope)
    out = [
        {"conversation_id": h.conversation_id, "turn_id": h.turn_id, "latest_seq": h.buffer.latest_seq}
        for h in registry.in_flight()
        if _authorized_handle(h, scope) is not None
    ]
    if get_backend() == "redis":
        seen = {row["conversation_id"] for row in out}
        for turn in await list_turns():
            cid = turn["conversation_id"]
            if cid in seen or not _owner_matches(TurnOwner.from_index(turn), scope):
                continue
            buf = RedisStreamBuffer(conversation_id=cid, turn_id=int(turn["turn_id"]))
            await buf.refresh()
            if buf.is_closed:
                continue   # finished; only the replay buffer is left
            out.append({
                "conversation_id": cid,
                "turn_id": int(turn["turn_id"]),
                "latest_seq": buf.latest_seq,
            })
    return {"in_flight": out}


@router.get("/in-flight", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def in_flight(scope: TenantScopeDep, conversation_id: str | None = None):
    """Probe so the renderer can decide whether to open a /tail on mount.

    `latest_seq` is the count of records so far; pass `from_seq=0` to
    replay the whole turn on first reconnect, or the last-rendered seq to
    resume without re-rendering.
    """
    _require_streaming_scope(scope)
    handle = registry.get(conversation_id) if conversation_id else None
    handle = _authorized_handle(handle, scope)
    if handle is None:
        found = await _shared_turn(conversation_id, scope)
        if found is None:
            return {"in_flight": False, "has_buffer": False, "latest_seq": 0, "turn_id": None}
        # Liveness comes from the buffer's terminal record, not from a process:
        # the replica running this turn may not be the one answering.
        return {
            "in_flight": found.in_flight,
            "has_buffer": True,
            "latest_seq": found.buffer.latest_seq,
            "turn_id": int(found.index["turn_id"]),
        }
    return {
        "in_flight": handle.is_running,
        "has_buffer": True,
        "latest_seq": handle.buffer.latest_seq,
        "turn_id": handle.turn_id,
    }


class CancelRequest(BaseModel):
    conversation_id: str
    expected_run_id: UUID | None = None
    # "stalled" saves the turn as a stall rather than a Stop. "client_gone"
    # retains the disconnect cause alongside a remote cancellation request.
    # Free text keeps newer clients compatible: unknown reasons act as Stop.
    reason: str | None = None


@router.post("/cancel", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def cancel_response(req: CancelRequest, scope: TenantScopeDep):
    """Halt the in-flight producer (Stop button), optionally matching one run.

    Fetch-abort / tab-close otherwise cancels only an originating streaming
    POST that opted in with cancel_on_disconnect=true.

    404 when no turn the caller may touch is registered — same shape as /tail,
    so a foreign-org id is indistinguishable from an unknown one (no existence
    leak) and can never cancel another org's run. The client treats 404 as
    "already done."

    On the remote backend the work is in a scratchpad pod, so cancelling the
    local producer only stops our tail: the pod would run to completion,
    spending tokens, with nobody listening. The flag is what reaches it, and
    because it is a Redis key rather than a message, it works whichever replica
    this request landed on.
    """
    _require_streaming_scope(scope)
    stalled = req.reason == STALLED_CANCEL_REASON
    handle = _authorized_handle(registry.get(req.conversation_id), scope)
    if req.expected_run_id is not None:
        expected = str(req.expected_run_id)
        if handle is not None and handle.run_id == expected:
            cancelled = await cancel_run(handle, reason=req.reason)
            return {"cancelled": cancelled, "conversation_id": req.conversation_id}
        found = await _shared_turn(req.conversation_id, scope)
        if found is None or found.index.get("run_id") != expected:
            return JSONResponse(status_code=404, content={"status": "not_found"})
        if not found.in_flight or found.buffer.is_finishing:
            return {"cancelled": False, "conversation_id": req.conversation_id}
        await request_cancel(found.index["correlation_id"], reason=req.reason, client=get_redis())
        return {"cancelled": True, "conversation_id": req.conversation_id}
    if handle is not None and get_backend() == "redis":
        turn = await get_turn(req.conversation_id)
        if turn is not None and turn["correlation_id"] != handle.lifecycle.identity.correlation_id:
            # RunRegistry retains finished handles for replay. A successor may
            # run on another replica; _shared_turn below authorizes its owner.
            handle = None
    if handle is not None:
        if get_backend() == "redis" or handle.lifecycle.identity.remote:
            await _request_cancel(handle.lifecycle.identity.correlation_id, stalled=stalled)
        # Before the cancel, so the producer's CancelledError handler cannot
        # miss it (see TurnLifecycle). Assigned, not only set: the latest
        # cancel decides, so a Stop after a stall cancel that the turn
        # swallowed (a running cell) saves as a Stop.
        handle.lifecycle.stalled = stalled
        cancelled = await handle.cancel()
        return {"cancelled": cancelled, "conversation_id": req.conversation_id}

    found = await _shared_turn(req.conversation_id, scope)
    if found is None or not found.in_flight:
        return JSONResponse(status_code=404, content={"status": "not_found"})
    await _request_cancel(found.index["correlation_id"], stalled=stalled)
    return {"cancelled": True, "conversation_id": req.conversation_id}


# Bounds on the answer body. Not a security boundary — one server process per
# user, and the payload is the user answering their own agent — but the values
# subset check is set-based, so an unbounded list would be built and hashed
# before anything rejected it. Sized so no legitimate GUI answer can hit them:
# anton caps a choice question at MAX_OPTIONS=10 options and the broker now
# rejects duplicates, so a well-formed `values` is at most 10 entries;
# _MAX_VALUE_LENGTH clears any plausible option value (they are model-authored
# short tokens); _MAX_TEXT_LENGTH is well past what someone types into a
# free-form answer box. Exceeding these is a 422 from Pydantic rather than one
# of the spec's 400 statuses: a malformed request, not a rejected answer.
_MAX_VALUES = 64
_MAX_VALUE_LENGTH = 512
_MAX_TEXT_LENGTH = 8192


class AnswerRequest(BaseModel):
    conversation_id: str
    question_id: str
    values: list[Annotated[str, StringConstraints(max_length=_MAX_VALUE_LENGTH)]] | None = (
        Field(default=None, max_length=_MAX_VALUES)
    )
    text: str | None = Field(default=None, max_length=_MAX_TEXT_LENGTH)
    skipped: bool | None = None


# Response per answer outcome, shared by the in-process broker (SubmitResult)
# and the remote pod path (RemoteAnswerResult); both enums use these values.
_ANSWER_RESPONSES: dict[str, tuple[int, dict]] = {
    "not_found": (404, {"status": "not_found"}),
    "already_answered": (409, {"accepted": False, "status": "already_answered"}),
    "invalid_option": (400, {"status": "invalid_option"}),
    "too_large": (400, {"status": "answer_too_large"}),
}


def _answer_response(result: SubmitResult | RemoteAnswerResult):
    # Exhaustive on purpose, with a raising default: callers settle
    # authorization before this runs, so the cost of a fall-through is not an
    # access-control hole but a desynchronised UI — a new result member
    # answered as 200 {"accepted": true} while the
    # future stayed unresolved, so the card would render as delivered and the
    # turn would hang to its 300 s timeout. A 500 is the right direction for
    # "the server does not understand its own state".
    value = getattr(result, "value", None)
    if value == "accepted":
        return {"accepted": True}
    if value not in _ANSWER_RESPONSES:
        raise AssertionError(f"unhandled answer result: {result}")
    status_code, content = _ANSWER_RESPONSES[value]
    return JSONResponse(status_code=status_code, content=content)


@router.post("/answer", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def answer_question(req: AnswerRequest, scope: TenantScopeDep):
    """Deliver the user's answer to a question a turn is blocked on.

    404 mirrors /cancel: no run the caller may touch, or no such pending
    question. 409 means somebody already answered — a second tab, or a
    double click. The client's source of truth for what was chosen is the
    response.ask_user_answered event, not its own click.

    On the remote backend the question lives in a pod: the answer is queued
    in Redis for the turn's controller and the status is the pod's own
    verdict (turnqueue/answers.py).
    """
    _require_streaming_scope(scope)
    # Keyed off the backend setting rather than local-registry presence (the
    # way /cancel and /tail check first): on the replica that owns this turn a
    # registry handle exists, but the broker it would submit to never holds a
    # pod's question — the pod is the source of truth on the remote backend —
    # so a registry-first check here would 404 on the very replica that could
    # otherwise serve the request. Remote mode also requires
    # COWORK_STREAM_BACKEND=redis: _shared_turn (used below) returns None on
    # any other stream backend, so a misconfigured deployment 404s instead of
    # silently falling through to the in-process broker.
    remote = TurnQueueSettings().is_remote
    correlation_id = None
    if remote:
        found = await _shared_turn(req.conversation_id, scope)
        if found is None or not found.in_flight:
            return JSONResponse(status_code=404, content={"status": "not_found"})
        correlation_id = found.index["correlation_id"]
    else:
        handle = _authorized_handle(registry.get(req.conversation_id), scope)
        if handle is None:
            return JSONResponse(status_code=404, content={"status": "not_found"})

    values = [v for v in (req.values or []) if v]
    text = (req.text or "").strip()
    skipped = bool(req.skipped)
    if skipped and (values or text):
        return JSONResponse(status_code=400, content={"status": "ambiguous_answer"})
    if not (values or text or skipped):
        return JSONResponse(status_code=400, content={"status": "empty_answer"})

    payload: dict = {}
    if values:
        payload["values"] = values
    if text:
        payload["text"] = text
    if skipped:
        payload["skipped"] = True

    if remote:
        remote_result = await submit_remote_answer(
            conversation_id=req.conversation_id,
            correlation_id=correlation_id,
            question_id=req.question_id,
            payload=payload,
        )
        return _answer_response(remote_result)

    return _answer_response(broker.submit(req.conversation_id, req.question_id, payload))


@router.get("/tail", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def tail_response(
    scope: TenantScopeDep,
    conversation_id: str = Query(..., description="Conversation to tail."),
    from_seq: int = Query(0, ge=0, description="Resume from this seq; records with seq >= from_seq are replayed, then live-tail."),
):
    """Reconnect to an in-flight (or just-finished) turn: replay from
    `from_seq` then live-tail to the terminal record. 404 when no buffer is
    registered — the client should fall back to GET /conversations/{id}/items
    for the persisted history."""
    _require_streaming_scope(scope)
    handle = _authorized_handle(registry.get(conversation_id), scope)
    if handle is None:
        # No local handle only means another replica started this turn; its
        # buffer is readable from here.
        found = await _shared_turn(conversation_id, scope)
        if found is None:
            return JSONResponse(status_code=404, content={"status": "not_found"})
        return StreamingResponse(
            sse_from_buffer(found.buffer, from_seq),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )
    return StreamingResponse(
        sse_from_buffer(handle.buffer, from_seq),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
