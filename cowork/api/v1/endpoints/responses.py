"""
Responses API endpoints for API v1.

This module contains endpoints for handling OpenAI-compatible Responses API requests,
including both streaming and non-streaming responses.
"""

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
        return StreamingResponse(result, media_type="text/event-stream")

    return result
