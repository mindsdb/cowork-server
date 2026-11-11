"""Where a message sent while a turn runs goes: into that turn, or the queue
(ENG-4194).

POST /responses/addon settles authorization and the turn, then hands the
message here. Jev decides with ``jev_route.ADDON_ROUTE_QUESTION``; the person
can also pick "Add to current task" themselves (``force``), which skips Jev.
Every failure queues the message, which is what happens without this feature,
so nothing here raises.

A turn that runs in a pod (``COWORK_TURN_BACKEND=remote``) has no in-process
channel: ``route_remote_addon`` reads what Jev needs from the database and
Redis, then hands the message to the pod (``turnqueue/addons.py``).
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, Literal
from uuid import UUID

from cowork.common.history_scrub import scrub_credentials
from cowork.common.settings.app_settings import TurnQueueSettings
from cowork.db.scoped import TenantScope
from cowork.handlers import jev_route
from cowork.streaming.addons import AddonChannel
from cowork.turnqueue.addons import RemoteAddonResult, remote_added_texts, remote_progress, submit_remote_addon

logger = logging.getLogger(__name__)

QueuedReason = Literal[
    "disabled", "no_turn", "not_supported", "jev_off", "jev_timeout", "jev_error",
    "low_confidence", "next_task", "empty",
    # Remote (pod) turns only: the pod did not answer in time, and the
    # message did not fit the pod's stdin line.
    "pod_timeout", "too_large",
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
    refusal = await _jev_refusal(
        conversation_id=channel.conversation_id,
        task=channel.task,
        progress=channel.progress(),
        added=channel.added_texts(),
        text=text,
        scope=scope,
    )
    if refusal is not None:
        return refusal
    addon_id = channel.add(text, decided_by="jev")
    return added(addon_id, "jev") if addon_id else queued("no_turn")


async def _jev_refusal(
    *, conversation_id: str, task: str, progress: list[str], added: list[str], text: str, scope: TenantScope,
) -> dict | None:
    """Ask Jev whether *text* belongs to the running task. None when it does,
    else the queued response saying why not."""
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
        task=scrub_credentials(task),
        # Step descriptions are the model's own words, so scrubbed too.
        progress=[scrub_credentials(line) for line in progress],
        added=[scrub_credentials(t) for t in added],
        message=scrub_credentials(text),
    )
    from anton.core.llm.tracing import TraceContext, reset_trace_context, set_trace_context

    # Attributed to the conversation like the route gate's decisions, so the
    # gateway doesn't stamp it as a direct API call on the user's key.
    token = set_trace_context(TraceContext(
        session_id=conversation_id, harness="anton", tags=(jev_route.JEV_ROUTE_TAG,),
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
        extra={"conversation_id": conversation_id},
    )
    if verdict.error:
        return queued("jev_timeout" if verdict.error == "timeout" else "jev_error")
    if not verdict.add:
        return queued("low_confidence" if verdict.chose_add else "next_task")
    return None


# ── Remote (pod) turns ─────────────────────────────────────────────────────

# One decision at a time per turn, as AddonChannel.deciding does in-process,
# so a follow-up is judged with the one before it already added. Per process:
# two replicas deciding for one turn at once is a person sending from two
# tabs within a Jev call, and the worst case is a follow-up judged without
# its predecessor.
_remote_deciding: dict[str, asyncio.Lock] = {}


@asynccontextmanager
async def _deciding(correlation_id: str):
    lock = _remote_deciding.setdefault(correlation_id, asyncio.Lock())
    try:
        async with lock:
            yield
    finally:
        if not lock.locked() and _remote_deciding.get(correlation_id) is lock:
            del _remote_deciding[correlation_id]


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if isinstance(content, list):
        return " ".join(
            block["text"] for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return ""


def _running_task(session, *, conversation_id: UUID) -> str:
    """The text of the question the in-flight turn runs: the latest pending
    user message (the producer saves it pending at turn start)."""
    from cowork.models.message import Message

    message = session.exec(
        session.select(Message)
        .where(Message.conversation_id == conversation_id)
        .where(Message.role == "user")
        .where(Message.pending == True)  # noqa: E712 - SQL boolean column
        .order_by(Message.seq.desc())
        .limit(1)
    ).first()
    return _content_text(message.content) if message is not None else ""


async def _remote_jev_inputs(
    *, conversation_id: str, correlation_id: str, scope: TenantScope,
) -> tuple[str, list[str], list[str]]:
    """What Jev reads about a pod's turn without an in-process channel: its
    question from the database, its recent tool calls from the replies the
    pod relayed, and the messages it already took. A part that can't be read
    is left empty rather than failing the decision; the already-added
    messages are the exception, since without them a follow-up is judged
    as if it were the first."""
    from cowork.db.units import run_db

    task = ""
    try:
        task = await run_db(partial(_running_task, conversation_id=UUID(conversation_id)), scope=scope)
    except Exception:
        logger.warning("[addon] could not read the running task", exc_info=True,
                       extra={"conversation_id": conversation_id})
    progress: list[str] = []
    try:
        progress = await remote_progress(conversation_id=conversation_id, correlation_id=correlation_id)
    except Exception:
        logger.warning("[addon] could not read the running turn's progress", exc_info=True,
                       extra={"conversation_id": conversation_id})
    already = await remote_added_texts(correlation_id)
    return task, progress, already


_REMOTE_ADD_FAILURES: dict[RemoteAddonResult, QueuedReason] = {
    RemoteAddonResult.REFUSED: "no_turn",
    RemoteAddonResult.NOT_FOUND: "no_turn",
    RemoteAddonResult.TIMEOUT: "pod_timeout",
    RemoteAddonResult.TOO_LARGE: "too_large",
}


async def route_remote_addon(
    *, conversation_id: str, correlation_id: str, text: str, force: bool, scope: TenantScope,
) -> dict:
    """``route_addon`` for a turn that runs in a pod: decide with Jev here
    (``force`` skips it), then hand the message to the pod and return its
    verdict (``turnqueue/addons.py``). Never raises."""
    try:
        async with _deciding(correlation_id):
            if force:
                decided_by = "user"
            else:
                task, progress, already = await _remote_jev_inputs(
                    conversation_id=conversation_id, correlation_id=correlation_id, scope=scope,
                )
                refusal = await _jev_refusal(
                    conversation_id=conversation_id, task=task, progress=progress,
                    added=already, text=text, scope=scope,
                )
                if refusal is not None:
                    return refusal
                decided_by = "jev"
            result, addon_id = await submit_remote_addon(
                conversation_id=conversation_id, correlation_id=correlation_id,
                text=text, decided_by=decided_by,
            )
    except Exception:
        logger.exception("[addon] remote routing failed; queueing", extra={"conversation_id": conversation_id})
        return queued("jev_error")
    if result is RemoteAddonResult.ADDED:
        return added(addon_id, decided_by)
    return queued(_REMOTE_ADD_FAILURES[result])
