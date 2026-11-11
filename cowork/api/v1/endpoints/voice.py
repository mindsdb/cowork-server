"""/voice: voice calls with the agent (ENG-4138, prototype).

GET  /voice/status   whether this deployment's MindsHub Inference serves live voice
POST /voice/session  the browser's SDP offer in, GPT-Live's answer out (cowork.services.voice)
POST /voice/operate  a request from the call in, the app actions to take out (cowork.services.voice_operator)

Both act as the caller, with their own MindsHub credential. A refusal keeps
MindsHub's 401, 403, 409 or 429, with a code and a sentence for the person in
``detail``; any other failure is 424.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

import asyncio
import logging

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.harnesses.memory.policy import resolve_memory_policy
from cowork.principal import mindshub_credential
from cowork.services import voice, voice_operator
from cowork.services.memory import build_turn_memory

logger = logging.getLogger(__name__)

router = APIRouter()


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
    messages: list[dict[str, Any]] = Field(min_length=1, max_length=200)
    snapshot: str = Field(default="", max_length=voice_operator.MAX_SNAPSHOT_CHARS * 2)
    # The person's IANA time zone, from their browser, so dates are counted from their today.
    timezone: str | None = Field(default=None, max_length=64)


@router.get("/status", response_model=voice.VoiceStatus, dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def voice_status(request: Request) -> voice.VoiceStatus:
    return voice.VoiceStatus(available=await voice.available(mindshub_credential(request)))


@router.post("/session", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def voice_session(
    body: VoiceSessionRequest, request: Request, scope: TenantScope = Depends(get_tenant_scope)
) -> dict:
    memory = await asyncio.to_thread(_person_memory, scope)
    context = voice.voice_context(tz_name=body.timezone, memory=memory)
    try:
        return await voice.start_session(mindshub_credential(request), body.sdp, body.voice, context)
    except voice.VoiceError as exc:
        raise HTTPException(status_code=exc.status, detail={"code": exc.code, "message": exc.message}) from exc


@router.post("/operate", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def voice_operate(body: VoiceOperateRequest, request: Request) -> dict:
    try:
        return await voice_operator.operate(mindshub_credential(request), body.messages, body.snapshot, body.timezone)
    except voice_operator.OperatorError as exc:
        raise HTTPException(status_code=exc.status, detail={"code": exc.code, "message": exc.message}) from exc
