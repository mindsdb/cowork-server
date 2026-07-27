"""Responses API endpoints (OpenAI-compatible Responses API).

Streaming turns run **detached**: POST /responses starts a background
producer (see handlers.responses + cowork.streaming) that writes to a
per-turn buffer; the response tails that buffer. Closing the connection
does NOT stop the run — the client reconnects via GET /responses/tail
with a `from_seq` cursor and resumes from where it left off. Only an
explicit POST /responses/cancel halts the producer.
"""
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
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
    if not scope.org_mode:
        return handle
    return handle if handle.org_id == scope.org_id else None

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


@router.post("/")
async def responses(
    responses_request: ResponsesRequest,
    session: SessionDep,
    principal: Principal | None = Depends(get_principal),
):
    handler = ResponsesHandler(session, principal=principal)
    result = await handler.handle(responses_request)
    if responses_request.stream:
        # `result` is sse_from_buffer(buffer, 0); the producer is already
        # running detached in the registry.
        return StreamingResponse(result, media_type="text/event-stream", headers=_SSE_HEADERS)
    return result


@router.get("/in-flight-list")
async def in_flight_list(scope: TenantScopeDep):
    """Conversations whose producer task is currently running. Cheap
    in-memory lookup — the renderer uses it to sync stream state across
    clients/boots. Scoped to the caller's org so it can't enumerate another
    org's live conversation ids."""
    _require_streaming_scope(scope)
    return {
        "in_flight": [
            {"conversation_id": h.conversation_id, "turn_id": h.turn_id, "latest_seq": h.buffer.latest_seq}
            for h in registry.in_flight()
            if _authorized_handle(h, scope) is not None
        ],
    }


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
        return {"in_flight": False, "has_buffer": False, "latest_seq": 0, "turn_id": None}
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
    """
    _require_streaming_scope(scope)
    handle = _authorized_handle(registry.get(req.conversation_id), scope)
    if handle is None:
        return JSONResponse(status_code=404, content={"status": "not_found"})
    cancelled = await handle.cancel()
    return {"cancelled": cancelled, "conversation_id": req.conversation_id}


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
        return JSONResponse(status_code=404, content={"status": "not_found"})
    return StreamingResponse(
        sse_from_buffer(handle.buffer, from_seq),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
