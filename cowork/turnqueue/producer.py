"""Remote turn producer: enqueue a whole-turn job, tail the reply stream,
and emit SSE frames for the turn.

Emits `response.created`, one `response.output_text.delta` per `turn_delta`
reply, and closes the buffer only on a terminal reply (`turn_completed` or
`turn_failed`) with a final `response.completed` frame. No S3, no
heartbeat. Wired into the responses handler in a later task.
"""
from __future__ import annotations

import json
import logging
import time
import uuid

from cowork.handlers.turn_errors import remote_turn_error
from cowork.services.providers import minds_chat_base_url
from cowork.turnqueue.auth_keys import mint_turn_key
from cowork.turnqueue.models import TurnJob, TurnReply
from cowork.streaming.turn_index import record_turn
from cowork.turnqueue.redis_client import get_redis
from cowork.common.settings.app_settings import TurnQueueSettings, default_turn_minds_api_host

logger = logging.getLogger(__name__)

# The pod reads the request as one line and truncates at 10 MiB (anton's
# MAX_REQUEST_BYTES); a truncated line won't parse. Margin covers the newline +
# controller-added fields.
_MAX_REQUEST_BYTES = 10 * 1024 * 1024
_REQUEST_BYTES_MARGIN = 64 * 1024


def _request_wire_size(params: dict) -> int:
    """Bytes the controller's request line will occupy for these params."""
    return len(json.dumps(params, separators=(",", ":")).encode("utf-8"))


def _fit_request(params: dict, conversation_id: str) -> dict:
    """Shed optional add-ons until the request fits the pod's stdin cap.

    skills then memory: both are re-sent every turn and degrade gracefully
    (pod falls back to builtins / no memory). History is the floor we can't
    shed here.
    """
    budget = _MAX_REQUEST_BYTES - _REQUEST_BYTES_MARGIN
    for field in ("skills", "memory"):
        if _request_wire_size(params) <= budget:
            break
        if params.pop(field, None) is not None:
            logger.warning(
                "[producer] dropped %s from turn %s: request line over %d-byte cap",
                field, conversation_id, budget,
            )
    return params


def _new_correlation_id() -> str:
    return str(uuid.uuid4())


async def _mint_llm_block(*, org_id: str | None, user_id: str | None,
                          correlation_id: str, settings: TurnQueueSettings) -> dict:
    """Mint a short-TTL MindsHub turn key and build the job's `llm` block.

    The mint call is authenticated with the internal shared secret
    (`X-Internal-Auth`) only - there is no per-tenant credential to look up or
    send. `org_id`/`user_id` (the request principal's identity) tell auth
    which tenant the key is scoped to; auth resolves them itself, so no
    per-user provider key is needed or read here.
    """
    api_key = await mint_turn_key(
        user_id=user_id, org_id=org_id, correlation_id=correlation_id,
        ttl_seconds=settings.turn_key_ttl_seconds, settings=settings,
    )
    base_url = settings.minds_base_url or minds_chat_base_url(default_turn_minds_api_host())
    block = {"provider": "minds-cloud", "api_key": api_key, "base_url": base_url}
    # Must be a MINDS alias (the pod runs on minds-cloud). Default to the
    # free-bucket model: a fresh org has no wallet balance, so premium 402s.
    from cowork.common.settings.app_settings import MINDS_FREE_MODEL
    coding_model = settings.minds_coding_model or MINDS_FREE_MODEL
    if coding_model:
        block["coding_model"] = coding_model
    return block


# What the reply loop reports when the worker stops answering. Shaped like the
# pod's own scrubbed "ExceptionType: message" errors so remote_turn_error can
# classify it (today: the generic redacted message and the generic error card).
UNRESPONSIVE_WORKER_ERROR = "TurnWorkerUnresponsive: the turn worker stopped responding"


def step_stream_events(data: dict) -> list:
    """Reconstruct anton Stream* events from one `turn_step` reply.

    The output feeds the same `format_responses_stream` the in-process path
    uses, so remote turns render steps/thinking identically to desktop."""
    from anton.core.llm.provider import (
        LLMResponse,
        StreamComplete,
        StreamContextCompacted,
        StreamTaskProgress,
        StreamToolResult,
        StreamToolUseDelta,
        StreamToolUseEnd,
        StreamToolUseStart,
        ToolCall,
    )

    step = data.get("step")
    if step == "tool_start":
        return [StreamToolUseStart(id=data.get("id") or "", name=data.get("name") or "")]
    if step == "tool_end":
        tool_id = data.get("id") or ""
        args = data.get("args") or ""
        events: list = []
        if args:
            # The pod accumulated the args deltas; replay them as one delta so
            # the formatter's join produces the same payload as in-process.
            events.append(StreamToolUseDelta(id=tool_id, json_delta=args))
        events.append(StreamToolUseEnd(id=tool_id))
        return events
    if step == "progress":
        return [StreamTaskProgress(
            phase=data.get("phase") or "",
            message=data.get("message") or "",
            eta_seconds=data.get("eta_seconds"),
            id=data.get("id"),
            ok=data.get("ok"),
        )]
    if step == "tool_result":
        return [StreamToolResult(
            name=data.get("name") or "",
            content=data.get("content") or "",
            action=data.get("action"),
            id=data.get("id"),
        )]
    if step == "compacted":
        return [StreamContextCompacted(message=data.get("message") or "")]
    if step == "round_end":
        # Only stop_reason and tool_calls truthiness drive the formatter's
        # round-break decision; a placeholder call carries the truthiness.
        calls = [ToolCall(id="", name="", input={})] if data.get("had_tool_calls") else []
        return [StreamComplete(response=LLMResponse(
            content="", tool_calls=calls, stop_reason=data.get("stop_reason"),
        ))]
    return []


async def stream_remote_replies(*, conversation_id: str, org_id: str | None,
                                user_id: str | None, input_text: str,
                                model: str | None,
                                turn_id: int = 0,
                                history: list | None = None,
                                memory: dict | None = None,
                                skills: dict | None = None,
                                correlation_id: str | None = None,
                                llm: dict | None = None):
    """Mint, enqueue, then yield this turn's replies as (kind, data) tuples.

    Yields turn_delta / turn_step / turn_memory in arrival order and ends with
    exactly one terminal — turn_completed, or turn_failed (classified with the
    same (code, message) the caller streams and persists; synthesized locally
    when the worker goes quiet past the idle timeout).
    `correlation_id`/`llm` reuse a turn key the routing gate already minted."""
    settings = TurnQueueSettings()
    r = get_redis()
    corr = correlation_id or _new_correlation_id()
    # A flag left by an earlier turn would cancel this one on its first line.
    await r.delete(f"cowork:cancel:{corr}")
    reply_stream = f"scratchpad:reply:{conversation_id}"

    # No client-picked model → the deployment's resolved default (org mode: the
    # free-bucket model). Resolved here so the model reaching the pod is always
    # a valid minds alias, independent of any harness's built-in default.
    if not model:
        from cowork.common.settings.user_settings import get_user_settings
        from cowork.db.scoped import TenantScope
        scope = TenantScope(org_mode=bool(org_id), org_id=org_id, user_id=user_id)
        model = get_user_settings(scope).resolved_planning_model

    llm_block = llm or await _mint_llm_block(
        org_id=org_id, user_id=user_id, correlation_id=corr, settings=settings,
    )

    params = {"input": input_text, "workspace_path": "/workspace",
              "model": model, "history": history or [], "llm": llm_block}
    # Omitted when empty: a memory-less turn keeps the pre-existing payload shape.
    if memory:
        params["memory"] = memory
    if skills:
        params["skills"] = skills
    params = _fit_request(params, conversation_id)

    job = TurnJob(
        op="anton_turn",
        conversation_id=conversation_id,
        correlation_id=corr,
        reply_stream=reply_stream,
        organization_id=org_id,
        user_id=user_id,
        params=params,
    )
    # Registry first: a conversation whose stream exists but isn't registered would
    # be invisible to the controller. The reverse is harmless, it prunes empty queues.
    await r.sadd(f"{settings.jobs_stream}:queues", conversation_id)
    await r.xadd(f"{settings.jobs_stream}:{conversation_id}", {"payload": job.model_dump_json()})
    # Any replica can now find this turn: its turn_id to open the buffer, its
    # correlation_id to cancel it, its org to authorize the caller. Recorded
    # here rather than after the first buffer record, because this generator
    # does not own the buffer; _shared_turn covers the gap between the two.
    await record_turn(
        conversation_id, turn_id=turn_id, correlation_id=corr,
        org_id=org_id, user_id=user_id, client=r,
    )

    last_id = "0-0"
    idle_timeout = settings.reply_idle_timeout_seconds
    last_reply_at = time.monotonic()
    while True:
        resp = await r.xread({reply_stream: last_id}, count=10, block=5000)
        if not resp:
            # Unbounded, this loop spun forever whenever the worker was down:
            # only a terminal reply ends the turn, so the SSE response never
            # ended. Bound the wait and fail the turn.
            if idle_timeout > 0 and time.monotonic() - last_reply_at > idle_timeout:
                code, message = remote_turn_error(UNRESPONSIVE_WORKER_ERROR)
                logger.warning(
                    "Remote turn abandoned: no reply for %.0fs conversation=%s correlation_id=%s",
                    idle_timeout, conversation_id, corr,
                )
                yield "turn_failed", {"error": UNRESPONSIVE_WORKER_ERROR,
                                      "code": code, "message": message}
                return
            continue
        for _stream, entries in resp:
            for entry_id, fields in entries:
                last_id = entry_id
                reply = TurnReply.model_validate_json(fields["payload"])
                if reply.correlation_id != corr:
                    # Deliberately does NOT refresh the idle clock: liveness
                    # means "this turn is progressing", and another turn's
                    # replies say nothing about ours.
                    continue
                last_reply_at = time.monotonic()
                kind = reply.kind
                data = reply.data or {}
                if kind == "turn_failed":
                    # Classify once; the SSE frame and the persisted events
                    # log must carry the same (code, message).
                    code, message = remote_turn_error(data.get("error"))
                    data = {**data, "code": code, "message": message}
                    logger.warning(
                        "Remote turn failed conversation=%s correlation_id=%s error=%s",
                        conversation_id, corr, data.get("error"),
                    )
                if kind in ("turn_delta", "turn_step", "turn_memory",
                            "turn_completed", "turn_failed"):
                    yield kind, data
                if kind in ("turn_completed", "turn_failed"):
                    return
