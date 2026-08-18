"""Responses API endpoints (OpenAI-compatible Responses API).

Streaming turns run **detached**: POST /responses starts a background
producer (see handlers.responses + cowork.streaming) that writes to a
per-turn buffer; the response tails that buffer. Closing the connection
does NOT stop the run — the client reconnects via GET /responses/tail
with a `from_seq` cursor and resumes from where it left off. Only an
explicit POST /responses/cancel halts the producer.
"""
import time
from typing import Annotated, NamedTuple

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, StringConstraints
from sqlmodel import Session
from starlette.responses import JSONResponse

from cowork.common.logger import setup_logging
from cowork.db.scoped import (
    MissingTenantScopeError,
    TenantScope,
    get_tenant_scope,
)
from cowork.db.session import get_session
from cowork.handlers.responses import ResponsesHandler, sse_from_buffer
from cowork.principal import Principal, get_principal
from cowork.schemas.responses import ResponsesRequest
from cowork.streaming import RunHandle, registry
from cowork.streaming.answers import SubmitResult, broker
from cowork.streaming.backend import get_backend
from cowork.streaming.buffer import RedisStreamBuffer
from cowork.streaming.turn_index import get_turn, list_turns
from cowork.turnqueue.redis_client import get_redis


logger = setup_logging()

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]
TenantScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


def _require_streaming_scope(scope: TenantScope) -> None:
    """Fail closed BEFORE any registry access.

    In org mode a caller with no org in scope (e.g. audit mode with no
    identity headers) gets 401 — exactly like every ScopedSession-backed
    endpoint. Checked up front, independent of whether the target turn
    exists, so an empty list or an unknown conversation_id can't mask the
    missing-identity case behind a 200/404.
    """
    if scope.org_mode and scope.org_id is None:
        raise MissingTenantScopeError("streaming access requires an org in scope")


def _authorized_handle(
    handle: RunHandle | None, scope: TenantScope
) -> RunHandle | None:
    """The run handle the caller is allowed to touch, or None.

    A conversation_id is not an authorization token — the registry is keyed by
    it and would otherwise hand any authenticated caller another org's live
    turn. Local mode never filters (today's single-user behavior); in org mode
    a handle owned by a different org reads as absent, never 403 (no existence
    leak). Assumes the scope was already validated by _require_streaming_scope.
    """
    if handle is None:
        return None
    return handle if _org_matches(handle.org_id, scope) else None


def _org_matches(org_id: str | None, scope: TenantScope) -> bool:
    """Whether the caller's scope may touch a turn owned by ``org_id``.

    Shared by the local-handle check above and the Redis turn index, so the two
    cannot drift into disagreeing about who may cancel whose turn.
    """
    if not scope.org_mode:
        return True   # today's single-user behavior
    return org_id == scope.org_id


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
    if turn is None or not _org_matches(turn.get("org_id") or None, scope):
        return None
    buf = RedisStreamBuffer(conversation_id=conversation_id, turn_id=int(turn["turn_id"]))
    await buf.refresh()
    # The index entry is written when the job is enqueued; the first record only
    # lands once the pod answers. Until the grace period is up, an empty stream
    # means "not started yet" rather than "over".
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


async def _request_cancel(correlation_id: str) -> None:
    """Ask whoever is running this turn to stop.

    The controller checks this key while it streams. Writing it rather than
    messaging a replica is what makes cancel work regardless of which replica
    the request landed on.
    """
    await get_redis().set(
        f"cowork:cancel:{correlation_id}", "1", ex=CANCEL_FLAG_TTL_SECONDS
    )


# Long enough for a controller mid-turn to notice, short enough that a stale
# flag cannot cancel a later turn that reuses the id.
CANCEL_FLAG_TTL_SECONDS = 300

# no-store (not just no-cache): a chat stream can carry secrets the model
# echoed (e.g. a raw API key embedded in generated scratchpad code), so it
# must never be written to a client's on-disk HTTP cache. See ENG-462.
_SSE_HEADERS = {
    "Cache-Control": "no-store",
    "Connection": "keep-alive",
    "Access-Control-Allow-Origin": "*",
}


@router.options("/")
async def options_handler():
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )


def _bearer_from_request(request: Request) -> str | None:
    """The caller's raw bearer, or None. In org mode this is the Keycloak token
    the ingress forwards; the turn-path cold-start warm (ENG-748) reaches the
    org catalog with it, exactly as /settings/recommended-models does."""
    header = request.headers.get("Authorization", "")
    return header[7:].strip() or None if header.lower().startswith("bearer ") else None


@router.post("/")
async def responses(
    responses_request: ResponsesRequest,
    request: Request,
    session: SessionDep,
    principal: Principal | None = Depends(get_principal),
):
    handler = ResponsesHandler(
        session, principal=principal, bearer_token=_bearer_from_request(request)
    )
    result = await handler.handle(responses_request)
    if responses_request.stream:
        # `result` is sse_from_buffer(buffer, 0); the producer is already
        # running detached in the registry.
        return StreamingResponse(result, media_type="text/event-stream", headers=_SSE_HEADERS)
    return result


@router.get("/in-flight-list")
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
            if cid in seen or not _org_matches(turn.get("org_id") or None, scope):
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


@router.get("/in-flight")
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


@router.post("/cancel")
async def cancel_response(req: CancelRequest, scope: TenantScopeDep):
    """Halt the in-flight producer (Stop button). Fetch-abort / tab-close
    does NOT cancel — only this does.

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
    handle = _authorized_handle(registry.get(req.conversation_id), scope)
    if handle is not None:
        turn = await get_turn(req.conversation_id) if get_backend() == "redis" else None
        if turn is not None:
            await _request_cancel(turn["correlation_id"])
        cancelled = await handle.cancel()
        return {"cancelled": cancelled, "conversation_id": req.conversation_id}

    found = await _shared_turn(req.conversation_id, scope)
    if found is None or not found.in_flight:
        return JSONResponse(status_code=404, content={"status": "not_found"})
    await _request_cancel(found.index["correlation_id"])
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


@router.post("/answer")
async def answer_question(req: AnswerRequest, scope: TenantScopeDep):
    """Deliver the user's answer to a question a turn is blocked on.

    404 mirrors /cancel: no run the caller may touch, or no such pending
    question. 409 means somebody already answered — a second tab, or a
    double click. The client's source of truth for what was chosen is the
    response.ask_user_answered event, not its own click.
    """
    _require_streaming_scope(scope)
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

    # Exhaustive on purpose, with a raising default: authorization is already
    # settled by here (:_authorized_handle above), so the cost of a fall-through
    # is not an access-control hole but a desynchronised UI — a new
    # SubmitResult member would answer 200 {"accepted": true} while the
    # future stayed unresolved, so the card would render as delivered and the
    # turn would hang to its 300 s timeout. A 500 is the right direction for
    # "the server does not understand its own state".
    result = broker.submit(req.conversation_id, req.question_id, payload)
    match result:
        case SubmitResult.ACCEPTED:
            return {"accepted": True}
        case SubmitResult.NOT_FOUND:
            return JSONResponse(status_code=404, content={"status": "not_found"})
        case SubmitResult.ALREADY_ANSWERED:
            return JSONResponse(
                status_code=409, content={"accepted": False, "status": "already_answered"}
            )
        case SubmitResult.INVALID_OPTION:
            return JSONResponse(status_code=400, content={"status": "invalid_option"})
        case _:
            raise AssertionError(f"unhandled SubmitResult: {result}")


@router.get("/tail")
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
