"""/voice: voice calls with the agent (ENG-4138, prototype).

GET  /voice/status   whether this deployment's MindsHub Inference serves live voice
POST /voice/session  the browser's SDP offer in, GPT-Live's answer out (cowork.services.voice)
POST /voice/operate  a request from the call in, the app actions to take out (cowork.services.voice_operator)

All three act as the caller, with their own MindsHub credential. Status
answers ``available: false`` for any refusal or failure. Session keeps
MindsHub's 401, 402, 403 or 429, and answers 409 where live voice is off;
operate keeps 401, 402, 403 or 429. Either puts a code and a sentence for the
person in ``detail`` (inference's own sentence for a 402 or 429); any other
failure is 424.

Under zero data retention voice is off: its models (GPT-Live, the operator's)
aren't ZDR, so status answers unavailable and session and operate answer 403
``zdr_restricted`` without sending MindsHub anything.
"""

from __future__ import annotations

import logging
from functools import partial

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.db.units import FilesystemBusy, run_efs
from cowork.harnesses.memory.policy import resolve_memory_policy
from cowork.principal import mindshub_credential
from cowork.services import voice, voice_operator
from cowork.services.memory import build_turn_memory
from cowork.services.mindshub_zdr import zdr_on

logger = logging.getLogger(__name__)

router = APIRouter()


_ZDR_MESSAGE = "Voice calls aren't available while zero data retention is on."


async def _zdr_on(request: Request, scope: TenantScope) -> bool:
    """Whether the caller's org has ZDR on, read with their own credential; unknown counts as on."""
    try:
        return await zdr_on(
            bearer_token=mindshub_credential(request), org_id=scope.org_id or "", user_id=scope.user_id or "",
        )
    except Exception:
        logger.warning("voice: couldn't read ZDR, so voice is off", exc_info=True)
        return True


def _refuse_under_zdr() -> HTTPException:
    return HTTPException(status_code=403, detail={"code": "zdr_restricted", "message": _ZDR_MESSAGE})


class VoiceSessionRequest(BaseModel):
    sdp: str = Field(min_length=1, max_length=voice.MAX_SDP_CHARS)
    voice: str | None = Field(default=None, max_length=32)
    # The person's IANA time zone, from their browser, so the call knows their date and time.
    timezone: str | None = Field(default=None, max_length=64)


def _person_memory(scope: TenantScope) -> dict[str, str]:
    """The person's own memory slots (profile, rules), unless memory is off.
    A read error leaves the call without memory rather than without a call."""
    try:
        if resolve_memory_policy(scope=scope).off:
            return {}
        return build_turn_memory(scope).get("global", {})
    except Exception:
        logger.exception("voice: couldn't read memory for the call")
        return {}


class VoiceOperateRequest(BaseModel):
    messages: list[voice_operator.OperatorMessage] = Field(min_length=1, max_length=200)
    snapshot: str = Field(default="", max_length=voice_operator.MAX_SNAPSHOT_CHARS * 2)
    # The person's IANA time zone, from their browser, so dates are counted from their today.
    timezone: str | None = Field(default=None, max_length=64)


@router.get("/status", response_model=voice.VoiceStatus, dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def voice_status(request: Request, scope: TenantScope = Depends(get_tenant_scope)) -> voice.VoiceStatus:
    if await _zdr_on(request, scope):
        return voice.VoiceStatus(available=False)
    return voice.VoiceStatus(available=await voice.available(mindshub_credential(request)))


@router.post("/session", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def voice_session(
    body: VoiceSessionRequest, request: Request, scope: TenantScope = Depends(get_tenant_scope)
) -> dict:
    if await _zdr_on(request, scope):
        raise _refuse_under_zdr()
    # Memory lives on the shared file store: read it in run_efs's slots and the
    # person's share of them. A busy store starts the call without memory.
    try:
        memory = await run_efs(partial(_person_memory, scope), scope=scope)
    except FilesystemBusy:
        logger.warning("voice: the file store is busy, so the call starts without memory")
        memory = {}
    context = voice.voice_context(tz_name=body.timezone, memory=memory)
    try:
        return await voice.start_session(mindshub_credential(request), body.sdp, body.voice, context)
    except voice.VoiceError as exc:
        raise HTTPException(status_code=exc.status, detail={"code": exc.code, "message": exc.message}) from exc


@router.post("/operate", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def voice_operate(
    body: VoiceOperateRequest, request: Request, scope: TenantScope = Depends(get_tenant_scope)
) -> dict:
    if await _zdr_on(request, scope):
        raise _refuse_under_zdr()
    try:
        return await voice_operator.operate(mindshub_credential(request), body.messages, body.snapshot, body.timezone)
    except voice_operator.OperatorError as exc:
        raise HTTPException(status_code=exc.status, detail={"code": exc.code, "message": exc.message}) from exc
