"""The Teams connector's internal routes, which only the recording proxy calls:

- ``POST /api/v1/connectors/teams/deliver`` posts a reply in a Teams thread.
- ``GET /api/v1/connectors/teams/member`` looks up a person in a conversation.

Each call is signed by the proxy (cowork/services/connectors/teams/bot.py, ``TeamsProxySignature``),
and the routes skip the bearer and identity layers. With any Teams setting missing, or without the
Teams packages, they aren't mounted and Cowork starts as usual.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field

from cowork.api.v1.permissions import require
from cowork.services.connectors.teams.bot import (
    MIN_INTERNAL_KEY_LENGTH,
    TeamsBot,
    TeamsProxySignature,
    TeamsSettings,
)
from cowork.services.connectors.teams.client import TeamsAuthError, teams_available

logger = logging.getLogger(__name__)

ROUTE_PREFIX = "/api/v1/connectors/teams"

# The proxy sends an answer in parts of at most 38,000 UTF-16 units.
MAX_TEXT_CHARS = 40_000
# One part, JSON-escaped at worst six bytes per character, plus the other fields.
MAX_BODY_BYTES = 256 * 1024

# Microsoft's client puts these IDs into the URL path unquoted, so one with "/", "?", "#" or "%"
# could send the bot's token to another path on the allowed host. Real Teams IDs use letters,
# digits and ":;@=-._". At least one character isn't a dot, so ".." can't be a path segment.
TEAMS_ID = r"^[^/?#%\\\s]*[^/?#%\\\s.][^/?#%\\\s]*$"


class DeliverRequest(BaseModel):
    service_url: str
    conversation_id: str = Field(min_length=1, pattern=TEAMS_ID)
    reply_to_id: str = Field(min_length=1, pattern=TEAMS_ID)
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


class DeliverResponse(BaseModel):
    id: str


class MemberResponse(BaseModel):
    id: str
    name: str | None
    email: str | None
    user_principal_name: str | None
    aad_object_id: str | None
    tenant_id: str | None


class _CappedBody(APIRoute):
    """Refuses a body over MAX_BODY_BYTES before reading the rest of it.

    FastAPI reads the body before any dependency runs, and the agent's code can reach this port
    without the key, so the signature check alone can't stop a large body from being held.
    """

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def capped(request: Request):
            declared = request.headers.get("content-length")
            if declared is not None and (not declared.isdigit() or int(declared) > MAX_BODY_BYTES):
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "body too large")
            body = bytearray()
            async for chunk in request.stream():
                body += chunk
                if len(body) > MAX_BODY_BYTES:
                    raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "body too large")
            request._body = bytes(body)
            return await handler(request)

        return capped


router = APIRouter(route_class=_CappedBody)


def _bot(request: Request) -> TeamsBot:
    return request.app.state.teams_bot


async def _call_teams(bot: TeamsBot, what: str, call):
    try:
        return await call(await bot.client())
    except TeamsAuthError as error:
        # The log keeps fields, never message values; the code (e.g.
        # invalid_client) says what to fix without Microsoft's full text.
        logger.warning(
            "Microsoft refused the Teams bot's credentials",
            extra={"event": "teams_call_failed", "operation": what, "reason": str(error).split(":")[0]},
        )
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Microsoft refused the bot's credentials") from None
    except Exception as error:
        # Only the type: an error from the Teams service can echo the request.
        logger.warning(
            "Teams call failed", extra={"event": "teams_call_failed", "operation": what, "error_type": type(error).__name__}
        )
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Teams {what} failed") from None


@router.post("/deliver", response_model=DeliverResponse, dependencies=[Depends(require(TeamsProxySignature))])
async def deliver(body: DeliverRequest, bot: Annotated[TeamsBot, Depends(_bot)]) -> DeliverResponse:
    bot.check_service_url(body.service_url)
    message_id = await _call_teams(
        bot,
        "reply",
        lambda client: client.reply(body.service_url, body.conversation_id, body.reply_to_id, body.text),
    )
    return DeliverResponse(id=message_id)


@router.get("/member", response_model=MemberResponse, dependencies=[Depends(require(TeamsProxySignature))])
async def member(
    bot: Annotated[TeamsBot, Depends(_bot)],
    service_url: Annotated[str, Query()],
    conversation_id: Annotated[str, Query(min_length=1, pattern=TEAMS_ID)],
    member_id: Annotated[str, Query(min_length=1, pattern=TEAMS_ID)],
) -> MemberResponse:
    bot.check_service_url(service_url)
    account = await _call_teams(
        bot, "member lookup", lambda client: client.get_member(service_url, conversation_id, member_id)
    )
    return MemberResponse(
        id=account.id,
        name=account.name,
        email=account.email,
        user_principal_name=account.user_principal_name,
        aad_object_id=account.aad_object_id,
        tenant_id=account.tenant_id,
    )


def install_teams_routes(app: FastAPI, exempt_paths: set[str], settings: TeamsSettings | None = None) -> bool:
    """Mount the routes when the bot is configured; returns whether it did.

    Their paths join ``exempt_paths``, which the bearer and identity layers
    skip: the proxy's signature is these routes' credential.
    """
    settings = settings or TeamsSettings()
    missing = settings.missing()
    if not (
        settings.bot_app_id
        or settings.bot_app_secret.get_secret_value()
        or settings.tenant_id
        or settings.internal_key.get_secret_value()
    ):
        return False
    if missing:
        logger.warning("Teams is off: settings missing", extra={"event": "teams_off", "reason": ", ".join(missing) + " not set"})
        return False
    if not teams_available():
        logger.warning("Teams is off: the Teams packages are installed only on Linux")
        return False
    if len(settings.internal_key.get_secret_value()) < MIN_INTERNAL_KEY_LENGTH:
        logger.warning(
            "COWORK_TEAMS_INTERNAL_KEY is shorter than %d characters; generate a random one",
            MIN_INTERNAL_KEY_LENGTH,
            extra={"event": "teams_weak_key"},
        )
    app.state.teams_bot = TeamsBot(settings)
    app.include_router(router, prefix=ROUTE_PREFIX, tags=["connectors"])
    exempt_paths.update(f"{ROUTE_PREFIX}{route.path}" for route in router.routes)
    logger.info("Teams bot routes on", extra={"event": "teams_routes_on"})
    return True
