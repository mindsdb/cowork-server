"""
Responses API endpoints for API v1.

This module contains endpoints for handling OpenAI-compatible Responses API requests,
including both streaming and non-streaming responses.
"""

import asyncio
import threading
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session
from starlette.responses import JSONResponse

from cowork.common.logger import setup_logging
from cowork.db.session import get_session
from cowork.handlers.responses import ResponsesHandler
from cowork.schemas.responses import ResponsesRequest
from cowork.services.conversations import ConversationService


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
    return {"in_flight": [{"conversation_id": cid} for cid in get_active_stream_ids()]}


@router.get("/in-flight")
async def in_flight(conversation_id: str | None = None):
    """Check if a conversation has an active stream."""
    active = conversation_id in get_active_stream_ids() if conversation_id else False
    return {"in_flight": active, "conversation_id": conversation_id}


class CancelRequest(BaseModel):
    conversation_id: str


@router.post("/cancel")
async def cancel_response(req: CancelRequest):
    """Cancel an active stream for the given conversation."""
    cancelled = request_cancel(req.conversation_id)
    return {"cancelled": cancelled, "conversation_id": req.conversation_id}


@router.get("/tail")
async def tail_response(
    session: SessionDep,
    conversation_id: str | None = None,
    from_seq: int = 0,
):
    """Tail/reconnect to a conversation's latest turn with event replay.

    Replays the durably persisted events of the newest assistant turn,
    starting at `from_seq` (the 0-based message_events sequence number).
    Events are committed BEFORE they are yielded to the live SSE stream
    (write-ahead — see ResponsesHandler._stream), so this replay is
    complete even when the client disconnected mid-turn. While the turn
    is still streaming `status` is "active"; poll again with the returned
    `next_seq` for a gapless continuation. Full history remains at
    GET /conversations/{id}/items.
    """
    if not conversation_id:
        return {"status": "not_found"}
    try:
        conv_uuid = UUID(conversation_id)
    except ValueError:
        return {"status": "not_found"}

    active = conversation_id in get_active_stream_ids()
    service = ConversationService(session)
    message = service.latest_assistant_message(conv_uuid)
    if message is None:
        if not active:
            return {"status": "not_found"}
        # Stream registered but no event persisted yet — nothing to replay.
        return {
            "status": "active",
            "conversation_id": conversation_id,
            "events": [],
            "next_seq": from_seq,
        }

    events = service.get_turn_events(message.id, from_seq)
    next_seq = (events[-1].sequence_number + 1) if events else from_seq
    return {
        "status": "active" if active else "completed",
        "conversation_id": conversation_id,
        "message_id": str(message.id),
        "events": [e.event_data for e in events],
        "next_seq": next_seq,
    }
