"""The caller's MindsHub account deletion request, read and made as them.

MindsHub's ``/v1/me/deletion-request/`` records that the caller wants their
account deleted; MindsHub's support deletes it from there. ``GET`` answers
the open request or 404, ``POST`` opens one (or answers the one already
open) with 202.

Same rules as every MindsHub call (cowork.services.mindshub_client): the
bearer is the caller's own and the host is the operator's. Nothing is cached,
since the app shows this right after the caller asks, and the POST is never
retried: MindsHub's answer, or the lack of one, is passed back as is.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import status
from pydantic import ValidationError

from cowork.schemas.mindshub_account_deletion import MindsHubDeletionRequestView
from cowork.services.mindshub_client import (
    MindsHubAnswer,
    MindsHubReply,
    get_mindshub_reply,
    send_mindshub,
)

logger = logging.getLogger(__name__)

DELETION_REQUEST_PATH = "/me/deletion-request/"

DELETION_UNREACHABLE = "Couldn't reach MindsHub. Try again in a moment."
DELETION_REFUSED = "MindsHub refused this request for your account."
DELETION_THROTTLED = "MindsHub is limiting how often this can be asked. Try again in a minute."


class DeletionRequestError(Exception):
    """What the route answers when MindsHub didn't give a request:
    ``status_code`` and a short ``detail``."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _parse(payload: Any) -> MindsHubDeletionRequestView | None:
    try:
        return MindsHubDeletionRequestView.model_validate(payload)
    except ValidationError:
        return None


def _error(reply: MindsHubReply, *, method: str) -> DeletionRequestError:
    """The route's answer for a reply that isn't a request."""
    code = reply.status_code
    if reply.answer is MindsHubAnswer.REFUSED and code in (
        status.HTTP_401_UNAUTHORIZED,
        status.HTTP_403_FORBIDDEN,
    ):
        # Never a 401: the app reads a 401 from this server as its own
        # sign-out, and the gateway already took this credential.
        return DeletionRequestError(status.HTTP_403_FORBIDDEN, DELETION_REFUSED)
    if reply.answer is MindsHubAnswer.THROTTLED:
        return DeletionRequestError(status.HTTP_429_TOO_MANY_REQUESTS, DELETION_THROTTLED)
    logger.warning(
        "MindsHub gave no deletion request",
        extra={
            "event": "mindshub_deletion_request_failed",
            "method": method,
            "answer": str(reply.answer),
            "status_code": code,
        },
    )
    return DeletionRequestError(status.HTTP_502_BAD_GATEWAY, DELETION_UNREACHABLE)


async def read_deletion_request(*, bearer_token: str) -> MindsHubDeletionRequestView | None:
    """The caller's open deletion request, or None when they have none.
    DeletionRequestError for any other answer."""
    reply = await get_mindshub_reply(DELETION_REQUEST_PATH, bearer_token)
    if reply.answer is MindsHubAnswer.REFUSED and reply.status_code == status.HTTP_404_NOT_FOUND:
        return None
    view = _parse(reply.payload) if reply.answer is MindsHubAnswer.OK else None
    if view is None:
        raise _error(reply, method="GET")
    return view


async def request_account_deletion(*, bearer_token: str) -> MindsHubDeletionRequestView:
    """Open a deletion request for the caller, or answer the one already
    open. DeletionRequestError for any other answer."""
    reply = await send_mindshub("POST", DELETION_REQUEST_PATH, bearer_token)
    view = _parse(reply.payload) if reply.answer is MindsHubAnswer.OK else None
    if view is None:
        raise _error(reply, method="POST")
    return view
