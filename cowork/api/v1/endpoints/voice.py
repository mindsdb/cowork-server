"""/voice: voice calls with the agent (ENG-4138, prototype).

GET  /voice/status   whether this deployment's MindsHub Inference serves live voice
POST /voice/session  the browser's SDP offer in, GPT-Live's answer out (cowork.services.voice)

Both act as the caller, with their own MindsHub credential. A refusal keeps
MindsHub's 401, 403, 409 or 429, with a code and a sentence for the person in
``detail``; any other failure is 424.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.principal import mindshub_credential
from cowork.services import voice

router = APIRouter()


class VoiceSessionRequest(BaseModel):
    sdp: str = Field(min_length=1, max_length=voice.MAX_SDP_CHARS)


@router.get("/status", response_model=voice.VoiceStatus, dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def voice_status(request: Request) -> voice.VoiceStatus:
    return voice.VoiceStatus(available=await voice.available(mindshub_credential(request)))


@router.post("/session", dependencies=[Depends(require(AuthenticatedInOrgMode))])
async def voice_session(body: VoiceSessionRequest, request: Request) -> dict:
    try:
        return await voice.start_session(mindshub_credential(request), body.sdp)
    except voice.VoiceError as exc:
        raise HTTPException(status_code=exc.status, detail={"code": exc.code, "message": exc.message}) from exc
