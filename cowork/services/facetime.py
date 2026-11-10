"""Video calls with the agent: an avatar the person talks to, which hands
their requests to Anton in their task (prototype).

The avatar is Nuance Labs' Nuance Live (docs.nuancelabs.ai). This server holds
the organization's Nuance API key (``COWORK_NUANCE_API_KEY``) and mints one
short-lived session token per call (``mint``). The web app starts the call
with that token and joins its LiveKit room; nothing else here takes part.

The session is fixed when the token is minted:

- The avatar's prompt (``PERSONA_PROMPT``) says it is Anton on a call, and
  that every request for work goes to its backend.
- The backend, Nuance's own reasoner, has one tool, ``ask_cowork``, which the
  web app runs: it sends the request to Anton in the person's task, as if they
  had typed it, and answers with Anton's reply. The avatar says the gist.

Nuance reviews every new prompt and tool definition against its acceptable
use policy, with a daily limit on reviews, so they are constants: what one
task is about reaches the avatar after the call starts, as background lines
the web app sends, never in the prompt.

Each token names the person by an opaque id (``end_user_id``), so repeated
policy ends pause that person's calls rather than the whole organization's.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

import httpx
from pydantic import BaseModel

from cowork.common.settings.app_settings import get_app_settings

logger = logging.getLogger(__name__)

_TIMEOUT_S = 30.0  # the first mint of a prompt waits on Nuance's review

PERSONA_PROMPT = (
    "You are Anton, the user's coworker in MindsHub Cowork, on a video call with them. Cowork is where they "
    "get work done with you: you analyze their data, research, write documents, browse websites and work with "
    "the apps they connected.\n\n"
    "Hand every request for work to your backend: anything about their data, files, projects or connected "
    "apps, anything to look up, write, change, send or build. Hand it off at once and say briefly what you are "
    'doing, such as "Let me look into that." Never say something is done before your backend says so. When '
    "the result comes, tell them the gist in two or three plain sentences, and say the details are in the "
    "task on their screen.\n\n"
    "Answer small talk and questions about this call yourself. Keep every reply short and spoken: no lists, "
    "links or code. When you don't know something, say so."
)

BACKEND_INSTRUCTIONS = (
    "You run requests for Anton, the user's agent in MindsHub Cowork. For every request, call ask_cowork once "
    "with the user's request in their own words and every detail from the conversation it needs, as if the "
    "user typed it in the chat. Never answer from your own knowledge. When ask_cowork returns, answer from its "
    "result in two or three short spoken sentences: the outcome first, then anything the user needs to know "
    "or decide. If it failed, say so plainly."
)

# The web app runs it (cowork's lib/facetime). destructiveHint: Anton may send,
# book or change things, so one request runs at most once across the
# session's tasks.
ASK_COWORK: dict[str, Any] = {
    "type": "function",
    "name": "ask_cowork",
    "description": (
        "Sends a request to Anton, the user's agent in Cowork, in the task on the user's screen, and returns "
        "Anton's answer. Anton analyzes data, searches and browses the web, reads and writes files, and uses "
        "the user's connected apps. It can take a few minutes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": "The request as the user would type it in the chat, with every detail Anton needs.",
            },
        },
        "required": ["request"],
    },
    "annotations": {"destructiveHint": True},
}


class FacetimeStatus(BaseModel):
    available: bool


class FacetimeToken(BaseModel):
    token: str
    expires_at: str


class FacetimeError(Exception):
    """A mint Nuance refused or couldn't serve, with what to tell the person."""

    def __init__(self, status: int, code: str, message: str, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.retry_after = retry_after


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=httpx.Timeout(_TIMEOUT_S))


def available() -> bool:
    return bool(get_app_settings().nuance_api_key.get_secret_value())


def end_user_id(org_id: str | None, user_id: str | None) -> str:
    """Nuance's id for the person: opaque, so it carries no name or email."""
    digest = hashlib.sha256(f"cowork\n{org_id or ''}\n{user_id or 'local'}".encode()).hexdigest()
    return f"cw_{digest[:40]}"


def mint_body(end_user: str) -> dict[str, Any]:
    return {
        "interface": "nuance_live",
        "personality": {"persona_id": get_app_settings().nuance_persona, "system_prompt": PERSONA_PROMPT},
        "interface_setup": {
            "delegation": {
                "type": "responses",
                "responses": {
                    "model": "default",
                    "instructions": BACKEND_INSTRUCTIONS,
                    "tools": [ASK_COWORK],
                    # One Anton turn at a time: the task runs them in order anyway.
                    "parallel_tool_calls": False,
                    "reasoning": {"effort": "low"},
                },
            },
            # The avatar may hang up after its goodbye.
            "runtime_tools": {"end_call": True},
        },
        "end_user_id": end_user,
    }


# What the person sees for a refusal Nuance gives a reason for; anything else
# is "couldn't start". Never Nuance's own detail: it is for developers.
_REFUSALS = {
    "credits_spent": "Video calls are used up for today. Try again tomorrow.",
    "start_limit": "Too many video calls were started in the last hour. Try again later.",
    "concurrency_limit": "Another video call is running. Try again when it ends.",
    "policy_curfew": "Video calls are paused for you for a while. Try again later.",
    "organization_paused": "Video calls are paused for this organization.",
    "organization_suspended": "Video calls are turned off for this organization.",
}


async def mint(end_user: str) -> FacetimeToken:
    """One session token for one call, which the web app must start within 60
    seconds. Never logs the API key or the token."""
    settings = get_app_settings()
    key = settings.nuance_api_key.get_secret_value()
    if not key:
        raise FacetimeError(409, "not_configured", "Video calls aren't set up on this server.")
    url = f"{settings.nuance_api_url.rstrip('/')}/session-tokens"

    async def _send() -> httpx.Response:
        async with _client() as client:
            return await client.post(url, json=mint_body(end_user), headers={"Authorization": f"Bearer {key}"})

    try:
        response = await asyncio.wait_for(_send(), _TIMEOUT_S)
    except Exception as exc:
        logger.warning("nuance mint failed", extra={"event": "facetime_unreachable", "error_type": type(exc).__name__})
        raise FacetimeError(424, "unreachable", "Couldn't reach the video call service. Try again.") from exc
    try:
        body = response.json()
    except ValueError:
        body = {}
    if response.status_code >= 400:
        body = body if isinstance(body, dict) else {}
        error = body.get("error") if isinstance(body.get("error"), dict) else {}
        code = str(body.get("code") or error.get("code") or "")
        retry = body.get("retry_after_seconds")
        logger.warning(
            "nuance refused a mint",
            extra={"event": "facetime_refused", "status_code": response.status_code, "reason": code},
        )
        message = _REFUSALS.get(code, "Couldn't start the video call. Try again in a moment.")
        status = response.status_code if response.status_code in (403, 409, 429) else 424
        raise FacetimeError(status, code or "refused", message, retry if isinstance(retry, int) else None)
    for warning in body.get("warnings") or []:
        logger.warning("nuance mint warning", extra={"event": "facetime_mint_warning", "warning": str(warning)[:300]})
    return FacetimeToken(token=str(body["token"]), expires_at=str(body.get("expires_at") or ""))
