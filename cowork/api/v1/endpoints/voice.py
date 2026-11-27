"""/voice: voice calls with the agent (ENG-4138, prototype).

GET  /voice/status   whether this deployment's MindsHub Inference serves live voice
POST /voice/session  the browser's SDP offer in, GPT-Live's answer out (cowork.services.voice); mode "dictation" for the composer mic
POST /voice/operate  a request from the call in, the app actions to take out (cowork.services.voice_operator)

All three act as the caller, with their own MindsHub credential. Status
answers ``available: false`` for any refusal or failure. Session keeps
MindsHub's 401, 402, 403 or 429, and answers 409 where live voice is off;
operate keeps 401, 402, 403 or 429. Either puts a code and a sentence for the
person in ``detail`` (inference's own sentence for a 402 or 429); any other
failure is 424.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.common.settings.user_settings import get_user_settings
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.db.units import FilesystemBusy, run_efs
from cowork.harnesses.memory.policy import memory_reads_off
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
    # "dictation": the composer's mic, which only listens (ENG-4630).
    mode: Literal["call", "dictation"] = "call"


def _person_memory(scope: TenantScope) -> dict[str, str]:
    """The person's own memory slots (profile, rules), unless memory is off.
    A read error leaves the call without memory rather than without a call."""
    try:
        settings = get_user_settings(scope)
        if memory_reads_off(memory_enabled=settings.memory_enabled, memory_mode=settings.memory_mode):
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
    # The widgets and inline pieces in the open chat, latest first (ENG-4684):
    # the model sees the first MAX_WIDGETS, each field cut to its bound.
    widgets: list[voice_operator.ScreenWidget] = Field(
        default_factory=list, max_length=voice_operator.MAX_WIDGETS * 2
    )


@router.get("/status", response_model=voice.VoiceStatus, dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def voice_status(request: Request) -> voice.VoiceStatus:
    return voice.VoiceStatus(available=await voice.available(mindshub_credential(request)))


@router.post("/session", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def voice_session(
    body: VoiceSessionRequest, request: Request, scope: TenantScope = Depends(get_tenant_scope)
) -> dict:
    context = ""
    if body.mode == "call":
        # Memory lives on the shared file store: read it in run_efs's slots and the
        # person's share of them. A busy store starts the call without memory.
        # Dictation only transcribes, so it reads none.
        try:
            memory = await run_efs(partial(_person_memory, scope), scope=scope)
        except FilesystemBusy:
            logger.warning("voice: the file store is busy, so the call starts without memory")
            memory = {}
        context = voice.voice_context(tz_name=body.timezone, memory=memory)
    try:
        return await voice.start_session(mindshub_credential(request), body.sdp, body.voice, context, body.mode)
    except voice.VoiceError as exc:
        raise HTTPException(status_code=exc.status, detail={"code": exc.code, "message": exc.message}) from exc


@router.post("/operate", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def voice_operate(body: VoiceOperateRequest, request: Request) -> dict:
    try:
        return await voice_operator.operate(
            mindshub_credential(request), body.messages, body.snapshot, body.timezone, body.widgets
        )
    except voice_operator.OperatorError as exc:
        raise HTTPException(status_code=exc.status, detail={"code": exc.code, "message": exc.message}) from exc
