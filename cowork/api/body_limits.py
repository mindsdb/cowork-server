"""Per-route caps on the size of a request's body, for the hosted server.

A hosted pod serves every person in it from one worker under a fixed memory
limit, and FastAPI reads a route's whole body before the route's own code runs
(``fastapi.routing.get_request_handler``): a JSON body into memory, and a
multipart form into temporary files that stay in memory up to 1 MiB each. A
route therefore can't refuse an oversize body itself. ``BodyLimitMiddleware``
refuses it first, with a 413, and ``cowork.server.create_app`` installs it in
org tenancy only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from starlette import status
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from cowork.services.comment_images import MAX_IMAGE_BYTES

#: The file upload routes. Matches the chart's ingress annotation
#: ``proxy-body-size: 500m`` (deployment/cowork-server/values.yaml); the two
#: change together.
UPLOAD_MAX_BYTES = 500 * 1024 * 1024
#: The skill upload: the skill zip's 200 MB expansion bound
#: (``SkillService._ZIP_MAX_UNCOMPRESSED`` in cowork/services/skills.py) plus
#: room for the zip's headers and the form's framing, so every zip within that
#: bound still uploads.
SKILL_UPLOAD_MAX_BYTES = 208 * 1024 * 1024
#: The comment image upload: the largest image (``MAX_IMAGE_BYTES`` in
#: cowork/services/comment_images.py) plus 64 KiB for the form's framing.
COMMENT_IMAGE_UPLOAD_MAX_BYTES = MAX_IMAGE_BYTES + 64 * 1024
#: Every other route. The largest real body found is a quick edit's four 5 MB
#: images in base64, about 27 MB (cowork/api/v1/endpoints/artifact_quick_edit.py).
DEFAULT_BODY_MAX_BYTES = 32 * 1024 * 1024

#: A Content-Length this layer reads: ASCII digits only, at most 19 of them,
#: so ``int`` never parses an unbounded string.
_CONTENT_LENGTH = re.compile(r"[0-9]{1,19}")


@dataclass(frozen=True)
class UploadRoute:
    """An upload route and the largest body it takes, in place of the default."""

    method: str
    #: Matched against the request's whole path with ``re.fullmatch``.
    path: re.Pattern[str]
    max_bytes: int


# Each segment a route fills in matches ``[^/]+``, as a Starlette path
# parameter does. The trailing slash is optional: Starlette redirects one form
# to the other without reading the body, so both forms take the route's cap.
UPLOAD_ROUTES: tuple[UploadRoute, ...] = (
    # upload_file, cowork/api/v1/endpoints/files.py
    UploadRoute(
        method="POST",
        path=re.compile(r"/api/v1/files/?"),
        max_bytes=UPLOAD_MAX_BYTES,
    ),
    # upload_project_files, cowork/api/v1/endpoints/project_files.py
    UploadRoute(
        method="POST",
        path=re.compile(r"/api/v1/projects/[^/]+/files/upload/?"),
        max_bytes=UPLOAD_MAX_BYTES,
    ),
    # upload_attachment, cowork/api/v1/endpoints/compat/stubs.py
    UploadRoute(
        method="POST",
        path=re.compile(r"/api/v1/attachments/[^/]+/[^/]+/upload/?"),
        max_bytes=UPLOAD_MAX_BYTES,
    ),
    # upload_skill, cowork/api/v1/endpoints/skills.py
    UploadRoute(
        method="POST",
        path=re.compile(r"/api/v1/skills/upload/?"),
        max_bytes=SKILL_UPLOAD_MAX_BYTES,
    ),
    # upload_comment_image, cowork/api/v1/endpoints/artifact_comment_images.py
    UploadRoute(
        method="POST",
        path=re.compile(r"/api/v1/artifacts/workspace/[^/]+/[^/]+/comment-images/?"),
        max_bytes=COMMENT_IMAGE_UPLOAD_MAX_BYTES,
    ),
)


@dataclass(frozen=True)
class BodyLimit:
    """The largest body one request may send, and what its refusal calls it."""

    max_bytes: int
    noun: Literal["upload", "request"]

    def refusal(self) -> str:
        """The 413's detail: one sentence the app shows as it is."""
        return f"This {self.noun} is larger than the {self.max_bytes // (1024 * 1024)} MB limit."


def body_limit_for(*, method: str, path: str) -> BodyLimit:
    """The cap on a request's body: its upload route's, or the default."""
    for route in UPLOAD_ROUTES:
        if route.method == method and route.path.fullmatch(path):
            return BodyLimit(max_bytes=route.max_bytes, noun="upload")
    return BodyLimit(max_bytes=DEFAULT_BODY_MAX_BYTES, noun="request")


class BodyLimitMiddleware:
    """Refuse a request body past its route's cap (body_limit_for) with 413.

    A ``Content-Length`` past the cap is answered at once: the app is never
    called and the body is never read. A ``Content-Length`` that isn't a byte
    count answers 400. Any other body, a chunked one with no
    ``Content-Length`` included, is counted as the app reads it, and the read
    that passes the cap raises ``HTTPException(413)``. FastAPI re-raises an
    HTTPException raised while it reads the body (``get_request_handler``),
    and the route's exception handling (Starlette's
    ``wrap_app_handling_exceptions``) answers it as the 413.

    This layer must sit inside every BaseHTTPMiddleware. That base class reads
    the body for the app inside an anyio task group (``receive_or_disconnect``
    in ``starlette.middleware.base``), and the group's exit wraps the 413
    raised here in an ExceptionGroup. FastAPI answers any exception other than
    an HTTPException raised while it reads the body as a 400, "There was an
    error parsing the body". Pure ASGI (not BaseHTTPMiddleware) itself, so it
    never buffers a body or an SSE stream.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = body_limit_for(method=scope["method"], path=scope["path"])
        declared = Headers(scope=scope).get("content-length")
        if declared is not None:
            if not _CONTENT_LENGTH.fullmatch(declared):
                response = JSONResponse(
                    {"detail": "The request's Content-Length header is not a byte count."},
                    status_code=status.HTTP_400_BAD_REQUEST,
                )
                await response(scope, receive, send)
                return
            if int(declared) > limit.max_bytes:
                response = JSONResponse(
                    {"detail": limit.refusal()},
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                )
                await response(scope, receive, send)
                return

        received = 0

        async def counted_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit.max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=limit.refusal()
                    )
            return message

        await self.app(scope, counted_receive, send)
