"""One final HTTP diagnostic, with context that lasts through streamed bodies."""

import logging
import re
from http import HTTPMethod, HTTPStatus
from time import monotonic
from uuid import uuid4

from fastapi import FastAPI
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from structlog.contextvars import bind_contextvars, reset_contextvars

logger = logging.getLogger(__name__)
_REQUEST_ID = re.compile(
    r"[0-9a-fA-F]{32}|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
)
_REFUSAL_REASONS = frozenset(
    {
        "expected_organization_missing",
        "expected_organization_invalid",
        "organization_mismatch",
    }
)


class RequestLoggingApp(FastAPI):
    def build_middleware_stack(self) -> ASGIApp:
        # Include ServerErrorMiddleware so generated 500 responses carry the
        # same correlation header and access-log context as handled responses.
        return RequestLoggingMiddleware(super().build_middleware_stack())


class RequestLoggingMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        supplied = headers.get(b"x-request-id", b"").decode("latin-1")
        valid = _REQUEST_ID.fullmatch(supplied) is not None
        request_id = supplied if valid else str(uuid4())
        tokens = bind_contextvars(
            request_id=request_id,
            request_id_source="header" if valid else "generated",
            method=scope["method"] if scope["method"] in HTTPMethod else "OTHER",
        )
        started = monotonic()
        status = 0
        escaped = False

        def fields():
            route = getattr(scope.get("route"), "path", None)
            return {
                "status_code": status or 500,
                "route": route,
                "duration_ms": round((monotonic() - started) * 1000, 2),
            }

        async def send_logged(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                response_headers = message.get("headers", [])
                message = {
                    **message,
                    "headers": [
                        (k, v)
                        for k, v in response_headers
                        if k.lower() != b"x-request-id"
                    ]
                    + [(b"x-request-id", request_id.encode("ascii"))],
                }
            await send(message)

        try:
            await self.app(scope, receive, send_logged)
        except Exception as exc:
            escaped = True
            logger.exception(
                "Request execution failed",
                extra={
                    **fields(),
                    "event": "request_failed",
                    "reason": "unhandled_exception",
                    "error_type": type(exc).__name__,
                },
            )
            exc._cowork_diagnostic_logged = True
            raise
        finally:
            if not escaped and status >= 400:
                candidate = scope.get("state", {}).get("_request_log_reason")
                reason = candidate if candidate in _REFUSAL_REASONS else None
                fallback = (
                    HTTPStatus(status).name.lower()
                    if status in HTTPStatus
                    else "http_error"
                )
                logger.log(
                    logging.ERROR if status >= 500 else logging.WARNING,
                    "Request refused" if status < 500 else "Request failed",
                    extra={
                        **fields(),
                        "event": "request_refused"
                        if status < 500
                        else "request_failed",
                        "reason": reason or fallback,
                    },
                )
            reset_contextvars(**tokens)
