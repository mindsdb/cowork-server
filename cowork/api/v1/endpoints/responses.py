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
from cowork.db.session import get_session
from cowork.handlers.responses import ResponsesHandler, sse_from_buffer
from cowork.schemas.responses import ResponsesRequest
from cowork.streaming import registry

logger = setup_logging()

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]

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
async def responses(responses_request: ResponsesRequest, session: SessionDep):
    handler = ResponsesHandler(session)
    result = await handler.handle(responses_request)
    if responses_request.stream:
        # `result` is sse_from_buffer(buffer, 0); the producer is already
        # running detached in the registry.
        return StreamingResponse(result, media_type="text/event-stream", headers=_SSE_HEADERS)
    return result


@router.get("/in-flight-list")
async def in_flight_list():
    """Conversations whose producer task is currently running. Cheap
    in-memory lookup — the renderer uses it to sync stream state across
    clients/boots."""
    return {
        "in_flight": [
            {"conversation_id": h.conversation_id, "turn_id": h.turn_id, "latest_seq": h.buffer.latest_seq}
            for h in registry.in_flight()
        ],
    }


@router.get("/in-flight")
async def in_flight(conversation_id: str | None = None):
    """Probe so the renderer can decide whether to open a /tail on mount.

    `latest_seq` is the count of records so far; pass `from_seq=0` to
    replay the whole turn on first reconnect, or the last-rendered seq to
    resume without re-rendering.
    """
    handle = registry.get(conversation_id) if conversation_id else None
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
async def cancel_response(req: CancelRequest):
    """Halt the in-flight producer (Stop button). Fetch-abort / tab-close
    does NOT cancel — only this does."""
    cancelled = await registry.cancel(req.conversation_id)
    return {"cancelled": cancelled, "conversation_id": req.conversation_id}


@router.get("/tail")
async def tail_response(
    conversation_id: str = Query(..., description="Conversation to tail."),
    from_seq: int = Query(0, ge=0, description="Resume from this seq; records with seq >= from_seq are replayed, then live-tail."),
):
    """Reconnect to an in-flight (or just-finished) turn: replay from
    `from_seq` then live-tail to the terminal record. 404 when no buffer is
    registered — the client should fall back to GET /conversations/{id}/items
    for the persisted history."""
    handle = registry.get(conversation_id)
    if handle is None:
        return JSONResponse(status_code=404, content={"status": "not_found"})
    return StreamingResponse(
        sse_from_buffer(handle.buffer, from_seq),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
