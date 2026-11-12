"""POST /api/v1/artifacts/checks/{check_id}: the app's answer to a preview check.

The turn asked with a ``response.artifact_check`` frame (see
streaming/preview_checks.py). Authorization copies POST /responses/answer: the
check belongs to a conversation, and a caller who may not touch that
conversation's turn gets the same 404 as for an id that does not exist, so
nothing leaks.

On the remote turn backend no check is ever opened in this process (the pod
runs anton), so every id is a 404 there. That is the intended degradation
until the pod side exists: anton reads no answer as "unverified".
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Path
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from starlette.responses import JSONResponse

from cowork.api.v1.endpoints.responses import TenantScopeDep, _authorized_handle, _require_streaming_scope
from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.streaming.preview_checks import CHECK_ID_PREFIX, CheckSubmitResult, preview_checks
from cowork.streaming.registry import registry

router = APIRouter()

# Bounds on the answer. The text comes from the artifact's own JavaScript, so
# every string is cut to a size before it reaches anton (which caps again for
# the prompt), rather than failing the whole answer: one long minified error
# line must not turn a real "failed" into a 422 and leave anton to time out.
# The list itself is bounded outright: the shim sends at most 20 reports.
MAX_ERRORS = 50
_MAX_MESSAGE = 1000
_MAX_FILE = 500
_MAX_URL = 1000
_MAX_TYPE = 32
# Ten minutes, well past any timeout the checker accepts (120 s).
_MAX_MS = 600_000
# "acheck:" plus 32 hex characters.
_MAX_CHECK_ID = 64


def _cut(limit: int):
    def cut(value: object) -> object:
        if value is None:
            return ""
        if isinstance(value, str):
            return value[:limit]
        return value

    return BeforeValidator(cut)


class PreviewDiagnostic(BaseModel):
    """One report the preview frame sent, as the app's usePreviewDiagnostics
    normalizes it. Types today: error, resource, csp, rejection, console,
    http. Not an enum: a newer shim must not make an older server reject the
    whole answer."""

    model_config = ConfigDict(extra="ignore")

    type: Annotated[str, _cut(_MAX_TYPE)] = "error"
    message: Annotated[str, _cut(_MAX_MESSAGE)] = ""
    file: Annotated[str, _cut(_MAX_FILE)] = ""
    line: int = Field(default=0, ge=0)
    url: Annotated[str, _cut(_MAX_URL)] = ""


class PreviewCheckAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: Literal["passed", "failed", "unavailable"]
    errors: list[PreviewDiagnostic] = Field(default_factory=list, max_length=MAX_ERRORS)
    ms: int = Field(default=0, ge=0, le=_MAX_MS)


_NOT_FOUND = (404, {"status": "not_found"})
_RESPONSES: dict[CheckSubmitResult, tuple[int, dict]] = {
    CheckSubmitResult.NOT_FOUND: _NOT_FOUND,
    CheckSubmitResult.ALREADY_ANSWERED: (409, {"accepted": False, "status": "already_answered"}),
}


@router.post("/checks/{check_id}", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def answer_preview_check(
    body: PreviewCheckAnswer,
    scope: TenantScopeDep,
    check_id: Annotated[str, Path(max_length=_MAX_CHECK_ID)],
):
    """Deliver the preview's verdict to the turn waiting on it.

    404: no such check, it timed out, its turn ended, or it belongs to a turn
    the caller may not touch. 409: already answered (a second tab).
    """
    _require_streaming_scope(scope)
    conversation_id = preview_checks.conversation_of(check_id) if check_id.startswith(CHECK_ID_PREFIX) else None
    if conversation_id is None or _authorized_handle(registry.get(conversation_id), scope) is None:
        status_code, content = _NOT_FOUND
        return JSONResponse(status_code=status_code, content=content)

    result = preview_checks.submit(check_id, body.model_dump())
    if result is CheckSubmitResult.ACCEPTED:
        return {"accepted": True}
    status_code, content = _RESPONSES[result]
    return JSONResponse(status_code=status_code, content=content)
