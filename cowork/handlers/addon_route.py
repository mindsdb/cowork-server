"""Where a message sent while a turn runs goes: into that turn, or the queue
(ENG-4194).

POST /responses/addon settles authorization and the turn, then hands the
message here. Jev decides with ``jev_route.ADDON_ROUTE_QUESTION``; the person
can also pick "Add to current task" themselves (``force``), which skips Jev.
Every failure queues the message, which is what happens without this feature,
so nothing here raises.
"""
from __future__ import annotations

import logging
from functools import partial
from typing import Literal

from cowork.common.history_scrub import scrub_credentials
from cowork.common.settings.app_settings import TurnQueueSettings
from cowork.db.scoped import TenantScope
from cowork.handlers import jev_route
from cowork.streaming.addons import AddonChannel

logger = logging.getLogger(__name__)

QueuedReason = Literal[
    "disabled", "no_turn", "not_supported", "jev_off", "jev_timeout", "jev_error",
    "low_confidence", "next_task", "empty",
]


def queued(reason: QueuedReason) -> dict:
    return {"status": "queued", "reason": reason}


def added(addon_id: str, decided_by: str) -> dict:
    return {"status": "added", "addon_id": addon_id, "decided_by": decided_by}


def _jev_target(scope: TenantScope) -> tuple[jev_route.JevTarget, str] | None:
    """Who to ask and with which model, from the caller's settings: the
    organization's own decision model, else Jev on MindsHub on the stored key.
    None when neither applies (the LLM gate's providers, Jev turned off, no
    MindsHub key). Sync: it reads settings through a session of its own."""
    from cowork.common.settings.user_settings import Provider, get_user_settings, provider_api_key_str
    from cowork.services.providers import minds_chat_base_url

    settings = get_user_settings(scope)
    queue_settings = TurnQueueSettings()
    mode = jev_route.gate_mode(settings)
    if mode == "own":
        target = jev_route.own_decision_target(settings)
        return (target, settings.decision_model or queue_settings.jev_route_model) if target else None
    if mode != "jev":
        return None
    key = provider_api_key_str(settings, Provider.MINDS_CLOUD)
    if not key:
        return None
    return jev_route.JevTarget.mindshub(minds_chat_base_url(settings.minds_url), key), queue_settings.jev_route_model


async def route_addon(*, channel: AddonChannel, text: str, force: bool, scope: TenantScope) -> dict:
    """Add *text* to the turn *channel* belongs to, or say why it is queued.
    Never raises."""
    try:
        async with channel.deciding:
            if force:
                addon_id = channel.add(text, decided_by="user")
                return added(addon_id, "user") if addon_id else queued("no_turn")
            return await _ask_jev(channel=channel, text=text, scope=scope)
    except Exception:
        logger.exception("[addon] routing failed; queueing", extra={"conversation_id": channel.conversation_id})
        return queued("jev_error")


async def _ask_jev(*, channel: AddonChannel, text: str, scope: TenantScope) -> dict:
    from cowork.db.units import run_on_unit_threads

    try:
        found = await run_on_unit_threads(partial(_jev_target, scope))
    except Exception:
        logger.warning("[addon] could not resolve the decision target; queueing", exc_info=True)
        return queued("jev_error")
    if found is None:
        return queued("jev_off")
    target, model = found
    queue_settings = TurnQueueSettings()
    # Scrubbed like the route gate's history: the state leaves this server.
    state = jev_route.addon_state(
        task=scrub_credentials(channel.task),
        progress=channel.progress(),
        added=[scrub_credentials(t) for t in channel.added_texts()],
        message=scrub_credentials(text),
    )
    from anton.core.llm.tracing import TraceContext, reset_trace_context, set_trace_context

    # Attributed to the conversation like the route gate's decisions, so the
    # gateway doesn't stamp it as a direct API call on the user's key.
    token = set_trace_context(TraceContext(
        session_id=channel.conversation_id, harness="anton", tags=(jev_route.JEV_ROUTE_TAG,),
    ))
    try:
        verdict = await jev_route.decide_addon(
            target=target,
            state=state,
            model=model,
            timeout_seconds=queue_settings.jev_route_timeout_seconds,
            min_p_add=queue_settings.addon_min_p_add,
        )
    finally:
        reset_trace_context(token)
    logger.info(
        "[addon] jev verdict add=%s p_add=%s ms=%d error=%s",
        verdict.add, verdict.p_add, verdict.ms, verdict.error or "-",
        extra={"conversation_id": channel.conversation_id},
    )
    if verdict.error:
        return queued("jev_timeout" if verdict.error == "timeout" else "jev_error")
    if not verdict.add:
        return queued("low_confidence" if verdict.chose_add else "next_task")
    addon_id = channel.add(text, decided_by="jev")
    return added(addon_id, "jev") if addon_id else queued("no_turn")
