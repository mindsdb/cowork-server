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
from pydantic import BaseModel, Field

from cowork.api.v1.permissions import require
from cowork.services.connectors.teams.bot import (
    TeamsBot,
    TeamsProxySignature,
    TeamsSettings,
)
from cowork.services.connectors.teams.client import TeamsAuthError, teams_available

logger = logging.getLogger(__name__)

ROUTE_PREFIX = "/api/v1/connectors/teams"


class DeliverRequest(BaseModel):
    service_url: str
    conversation_id: str = Field(min_length=1)
    reply_to_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class DeliverResponse(BaseModel):
    id: str


class MemberResponse(BaseModel):
    id: str
    name: str | None
    email: str | None
    user_principal_name: str | None
    aad_object_id: str | None
    tenant_id: str | None


router = APIRouter()


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
    conversation_id: Annotated[str, Query(min_length=1)],
    member_id: Annotated[str, Query(min_length=1)],
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
    app.state.teams_bot = TeamsBot(settings)
    app.include_router(router, prefix=ROUTE_PREFIX, tags=["connectors"])
    exempt_paths.update(f"{ROUTE_PREFIX}{route.path}" for route in router.routes)
    logger.info("Teams bot routes on", extra={"event": "teams_routes_on"})
    return True
