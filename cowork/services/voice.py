"""Voice calls with the agent: OpenAI GPT-Live through MindsHub Inference (ENG-4138, prototype).

The voice-only sibling of the video call (``cowork.services.facetime``). The
person talks; GPT-Live listens and speaks; every request for work is handed
back to the web app (client delegation), which sends it to Anton in the task
the way a typed message goes, and returns Anton's answer to be said.

This server only starts the call. ``start_session`` sends the browser's SDP
offer to MindsHub Inference's ``POST /v1/live/sessions`` as the caller, with
their own MindsHub credential: inference holds the OpenAI key, checks who they
are, and watches the session's usage. The answer comes back unchanged, and the
browser then talks to OpenAI directly over WebRTC.

The session is fixed here: GPT-Live's prompt (``VOICE_PROMPT``) and client
delegation. GPT-Live's client delegation carries no task text, so the web app
takes the request from the person's transcript.

Which inference: this deployment's own (``default_turn_minds_api_host``), the
one its turns use, so a PR environment calls its own and not dev's.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from pydantic import BaseModel

from cowork.common.settings.app_settings import default_turn_minds_api_host
from cowork.services.providers import minds_chat_base_url

logger = logging.getLogger(__name__)

_TIMEOUT_S = 30.0
MODEL = "gpt-live-1"
# An SDP offer is a few KB.
MAX_SDP_CHARS = 64_000

VOICE_PROMPT = (
    "You are Anton, the user's coworker in MindsHub Cowork, on a voice call with them. Cowork is where they get "
    "work done with you: you analyze their data, research, write documents, browse websites and work with the "
    "apps they connected.\n\n"
    "Delegate every request for work: anything about their data, files, projects or connected apps, anything "
    "to look up, write, change, send or build. Delegate requests about the app too: creating a project, "
    "moving a task, starting, opening or stopping a task, what a task's steps are, what is in the browser and "
    "doing something in it. Say briefly what you are doing, such as \"Let me look into that.\" Never say "
    "something is done before the result arrives. When it does, tell them the gist in two or three plain "
    "sentences.\n\n"
    "Answer small talk and questions about this call yourself. Keep every reply short and spoken: no lists, "
    "links or code. When you don't know something, say so."
)


class VoiceStatus(BaseModel):
    available: bool


class VoiceError(Exception):
    """A call MindsHub refused or couldn't start, with a sentence for the person."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def live_sessions_url() -> str:
    return f"{minds_chat_base_url(default_turn_minds_api_host())}/live/sessions"


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=httpx.Timeout(_TIMEOUT_S))


# Whether this deployment's inference serves live voice, per process. Inference
# has no capability route: an empty request answers 404 ``live_not_enabled``
# where it is off and 400 (no transport) where it is on, without starting
# anything. Cached, so the + menu doesn't ask on every page load.
_STATUS_TTL_S = 600.0
_status_cache: dict[str, tuple[float, bool]] = {}


async def available(bearer: str) -> bool:
    if not bearer:
        return False
    url = live_sessions_url()
    cached = _status_cache.get(url)
    if cached and time.monotonic() - cached[0] < _STATUS_TTL_S:
        return cached[1]
    try:
        async with _client() as client:
            response = await asyncio.wait_for(
                client.post(url, json={}, headers={"Authorization": f"Bearer {bearer}"}), _TIMEOUT_S
            )
    except Exception:
        return False
    if response.status_code in (401, 403):
        # Says nothing about the deployment: don't cache this caller's answer.
        return False
    on = response.status_code == 400
    _status_cache[url] = (time.monotonic(), on)
    return on


def session_body(sdp: str) -> dict[str, Any]:
    return {
        "session": {"model": MODEL, "instructions": VOICE_PROMPT, "delegation": {"type": "client"}},
        "transport": {"type": "webrtc", "sdp": sdp},
    }


# What the person sees for a refusal; anything else is "couldn't start".
_REFUSALS = {
    "live_not_enabled": "Voice calls aren't available on this deployment.",
    "rate_limit_exceeded": "Too many voice calls right now. Try again in a minute.",
}


async def start_session(bearer: str, sdp: str) -> dict:
    """GPT-Live's answer for the browser's SDP offer: ``{session: {id}, transport: {type, sdp}}``."""
    if not bearer:
        raise VoiceError(401, "unauthorized", "Sign in to MindsHub to start a voice call.")
    if not isinstance(sdp, str) or not sdp.strip() or len(sdp) > MAX_SDP_CHARS:
        raise VoiceError(400, "invalid_sdp", "The browser's call offer is missing.")
    try:
        async with _client() as client:
            response = await asyncio.wait_for(
                client.post(live_sessions_url(), json=session_body(sdp), headers={"Authorization": f"Bearer {bearer}"}),
                _TIMEOUT_S,
            )
    except Exception as exc:
        logger.warning("voice session start failed", extra={"event": "voice_unreachable", "error_type": type(exc).__name__})
        raise VoiceError(424, "unreachable", "Couldn't reach MindsHub to start the call. Try again.") from exc
    if response.status_code >= 400:
        try:
            error = response.json().get("error") or {}
        except (ValueError, AttributeError):
            error = {}
        code = str(error.get("code") or "")
        logger.warning(
            "MindsHub refused a voice session",
            extra={"event": "voice_refused", "status_code": response.status_code, "reason": code},
        )
        status = response.status_code if response.status_code in (401, 403, 409, 429) else 424
        if code == "live_not_enabled":
            status = 409
        raise VoiceError(status, code or "refused", _REFUSALS.get(code, "Couldn't start the voice call. Try again."))
    created = response.json()
    if not isinstance(created, dict) or not (created.get("transport") or {}).get("sdp"):
        raise VoiceError(424, "bad_answer", "MindsHub answered without a call.")
    return created
