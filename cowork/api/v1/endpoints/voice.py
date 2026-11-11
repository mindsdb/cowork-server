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

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.principal import mindshub_credential
from cowork.services import voice, voice_operator

router = APIRouter()


class VoiceSessionRequest(BaseModel):
    sdp: str = Field(min_length=1, max_length=voice.MAX_SDP_CHARS)
    voice: str | None = Field(default=None, max_length=32)


class VoiceOperateRequest(BaseModel):
    messages: list[dict[str, Any]] = Field(min_length=1, max_length=200)
    snapshot: str = Field(default="", max_length=voice_operator.MAX_SNAPSHOT_CHARS * 2)


@router.get("/status", response_model=voice.VoiceStatus, dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def voice_status(request: Request) -> voice.VoiceStatus:
    return voice.VoiceStatus(available=await voice.available(mindshub_credential(request)))


@router.post("/session", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def voice_session(body: VoiceSessionRequest, request: Request) -> dict:
    try:
        return await voice.start_session(mindshub_credential(request), body.sdp, body.voice)
    except voice.VoiceError as exc:
        raise HTTPException(status_code=exc.status, detail={"code": exc.code, "message": exc.message}) from exc


@router.post("/operate", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def voice_operate(body: VoiceOperateRequest, request: Request) -> dict:
    try:
        return await voice_operator.operate(mindshub_credential(request), body.messages, body.snapshot)
    except voice_operator.OperatorError as exc:
        raise HTTPException(status_code=exc.status, detail={"code": exc.code, "message": exc.message}) from exc
