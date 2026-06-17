"""
Responses API endpoints for API v1.

This module contains endpoints for handling OpenAI-compatible Responses API requests,
including both streaming and non-streaming responses.
"""

import asyncio
import json
import threading
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session
from starlette.responses import JSONResponse

from cowork.common.logger import setup_logging
from cowork.db.session import get_session
from cowork.handlers.responses import ResponsesHandler
from cowork.schemas.responses import ResponsesRequest


logger = setup_logging()

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]

# In-memory tracking of conversations with active streams.
_active_streams: dict[str, asyncio.Event] = {}
_active_streams_lock = threading.Lock()


def mark_stream_active(conversation_id: str) -> asyncio.Event:
    cancel_event = asyncio.Event()
    with _active_streams_lock:
        _active_streams[conversation_id] = cancel_event
    return cancel_event


def mark_stream_finished(conversation_id: str) -> None:
    with _active_streams_lock:
        _active_streams.pop(conversation_id, None)


def get_active_stream_ids() -> list[str]:
    with _active_streams_lock:
        return list(_active_streams.keys())


def request_cancel(conversation_id: str) -> bool:
    with _active_streams_lock:
        event = _active_streams.get(conversation_id)
    if event is None:
        return False
    event.set()
    return True


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
):
    handler = ResponsesHandler(session)
    result = await handler.handle(responses_request)

    if responses_request.stream:
        conversation_id = handler.last_conversation_id
        cancel_event = None
        if conversation_id:
            cancel_event = mark_stream_active(conversation_id)

        async def tracked_stream():
            try:
                async for chunk in result:
                    if cancel_event and cancel_event.is_set():
                        break
                    yield chunk
            finally:
                if conversation_id:
                    mark_stream_finished(conversation_id)

        return StreamingResponse(tracked_stream(), media_type="text/event-stream")

    return result


@router.get("/in-flight-list")
async def in_flight_list():
    """List conversations with active streams."""
    from cowork.services import stream_buffer

    entries = []
    for cid in get_active_stream_ids():
        buffer = stream_buffer.get_buffer(cid)
        entries.append({
            "conversation_id": cid,
            "latest_seq": buffer.latest_seq if buffer else 0,
        })
    return {"in_flight": entries}


@router.get("/in-flight")
async def in_flight(conversation_id: str | None = None):
    """Cheap probe so the client can decide whether to open a /tail SSE
    on conversation mount.

    ``in_flight``  — a producer (stream or scheduled run) is running.
    ``has_buffer`` — a turn buffer exists (running or just finished;
                     a just-finished buffer still has events to replay).
    ``latest_seq`` — events written so far; pass as ``from_seq`` to
                     /tail to skip what the client already rendered.
    """
    from cowork.services import stream_buffer

    if not conversation_id:
        return {"in_flight": False, "has_buffer": False, "latest_seq": 0, "conversation_id": conversation_id}
    active = conversation_id in get_active_stream_ids()
    buffer = stream_buffer.get_buffer(conversation_id)
    return {
        "in_flight": active or bool(buffer and not buffer.done),
        "has_buffer": buffer is not None,
        "latest_seq": buffer.latest_seq if buffer else 0,
        "conversation_id": conversation_id,
    }


class CancelRequest(BaseModel):
    conversation_id: str


@router.post("/cancel")
async def cancel_response(req: CancelRequest):
    """Cancel an active stream for the given conversation."""
    cancelled = request_cancel(req.conversation_id)
    return {"cancelled": cancelled, "conversation_id": req.conversation_id}


@router.get("/tail")
async def tail_response(conversation_id: str, from_seq: int = 0, model: str = "anton"):
    """Reconnect to an in-flight (or just-finished) turn.

    Replays the turn buffer from ``from_seq`` and then follows the live
    producer; the SSE frames are the same shape POST /responses emits,
    so the client reuses its existing event parser. Returns 404 when no
    buffer exists — the client falls back to persisted history. This is
    what lets the Task view attach to a scheduled "Run now" run and show
    progress + the final answer without a refresh.
    """
    from cowork.services import stream_buffer

    buffer = stream_buffer.get_buffer(conversation_id)
    if buffer is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No stream buffer for this conversation")

    async def sse():
        async for data in buffer.follow(max(0, from_seq)):
            event_type = data.get("type", "message")
            yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")
