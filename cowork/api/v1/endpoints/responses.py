"""
Responses API endpoints for API v1.

This module contains endpoints for handling OpenAI-compatible Responses API requests,
including both streaming and non-streaming responses.
"""

import threading
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
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
_active_streams: set[str] = set()
_active_streams_lock = threading.Lock()


def mark_stream_active(conversation_id: str) -> None:
    with _active_streams_lock:
        _active_streams.add(conversation_id)


def mark_stream_finished(conversation_id: str) -> None:
    with _active_streams_lock:
        _active_streams.discard(conversation_id)


def get_active_stream_ids() -> list[str]:
    with _active_streams_lock:
        return list(_active_streams)


@router.options("/")
async def options_handler():
    """
    Handle CORS preflight requests for Responses API endpoints.

    Returns:
        JSONResponse: CORS headers for preflight requests
    """
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
    """
    Handle Responses API requests (API v1).

    This endpoint provides OpenAI-compatible Responses API with support for
    both streaming and non-streaming responses.

    Args:
        responses_request (ResponsesRequest): The request containing chat messages and other parameters.

    Returns:
        StreamingResponse | JSONResponse: A streaming response if stream=True,
            otherwise a JSON response containing Responses API messages.

    Raises:
        HTTPException: 429 if usage limit exceeded.
        HTTPException: 500 if there's an error processing the request.
    """
    handler = ResponsesHandler(session)
    result = await handler.handle(responses_request)

    if responses_request.stream:
        conversation_id = handler.last_conversation_id
        if conversation_id:
            mark_stream_active(conversation_id)

        async def tracked_stream():
            try:
                async for chunk in result:
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


@router.post("/cancel")
async def cancel_response():
    """Cancel an active stream. Stub — returns success regardless."""
    return {"ok": True}


@router.get("/tail")
async def tail_response(conversation_id: str | None = None):
    """Tail/reconnect to an active stream. Stub — returns empty."""
    return {"status": "not_found"}
