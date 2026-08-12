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

from cowork.handlers.turn_errors import remote_turn_error, response_failed_sse
from cowork.services.providers import minds_chat_base_url
from cowork.turnqueue.auth_keys import mint_turn_key
from cowork.turnqueue.models import TurnJob, TurnReply
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


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


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
    # The pod always runs on minds-cloud, so its coding calls (the completion
    # verifier + nested scratchpad calls) must name a MINDS model alias. Do NOT
    # use the tenant's resolved_coding_model: a hosted user with no configured
    # provider resolves to the Anthropic default (claude-haiku-4-5-...), which
    # minds inference 404s. Use the minds-cloud coding default, overridable
    # per-env when a PR/env serves a different alias.
    from cowork.common.settings.app_settings import CODING_MODEL_DEFAULTS
    coding_model = settings.minds_coding_model or CODING_MODEL_DEFAULTS.get("minds_cloud", "")
    if coding_model:
        block["coding_model"] = coding_model
    return block


# What the reply loop reports when the worker stops answering. Shaped like the
# pod's own scrubbed "ExceptionType: message" errors so remote_turn_error can
# classify it (today: the generic redacted message and the generic error card).
UNRESPONSIVE_WORKER_ERROR = "TurnWorkerUnresponsive: the turn worker stopped responding"


async def _fail_unresponsive_worker(*, buffer, on_event, idle_seconds: float,
                                    conversation_id: str, corr: str) -> None:
    """End a turn whose worker went silent, exactly like a `turn_failed` reply.

    Same `response.failed` frame + `buffer.close("error")` as the reply-driven
    failure path, and the same `on_event("turn_failed", ...)` so the caller
    persists the failure — a bare close would render as nothing at all.
    """
    code, message = remote_turn_error(UNRESPONSIVE_WORKER_ERROR)
    logger.warning(
        "Remote turn abandoned: no reply for %.0fs conversation=%s correlation_id=%s",
        idle_seconds, conversation_id, corr,
    )
    if on_event is not None:
        on_event("turn_failed", {"error": UNRESPONSIVE_WORKER_ERROR,
                                 "code": code, "message": message})
    await buffer.append("sse", {"sse": response_failed_sse(message, code)})
    await buffer.close("error")


async def produce_remote_turn(*, conversation_id: str, org_id: str | None,
                              user_id: str | None, input_text: str,
                              model: str | None, buffer,
                              history: list | None = None,
                              harness_id: str | None = None,
                              memory: dict | None = None,
                              skills: dict | None = None,
                              on_event=None) -> None:
    """`on_event(kind, data)` is called per reply (turn_delta/turn_completed/
    turn_failed) so the caller can collect the turn for persistence."""
    settings = TurnQueueSettings()
    r = get_redis()
    corr = _new_correlation_id()
    reply_stream = f"scratchpad:reply:{conversation_id}"

    llm_block = await _mint_llm_block(
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
    await r.xadd(settings.jobs_stream, {"payload": job.model_dump_json()})
    # conversation_id + harness mirror _inject_created on the in-process path:
    # the client learns the canonical id (it may have sent a non-UUID one).
    created = {"type": "response.created", "conversation_id": conversation_id}
    if harness_id:
        created["harness"] = harness_id
    await buffer.append("sse", {"sse": _sse("response.created", created)})

    last_id = "0-0"
    idle_timeout = settings.reply_idle_timeout_seconds
    last_reply_at = time.monotonic()
    while True:
        resp = await r.xread({reply_stream: last_id}, count=10, block=5000)
        if not resp:
            # Unbounded, this loop spun forever whenever the worker was down:
            # only a terminal reply closes the buffer, so FileStreamBuffer.tail
            # never returned and the SSE response never ended. That used to
            # self-heal by accident — an intermediary dropped the quiet
            # connection, the renderer's read completed and the composer came
            # back — but the keepalive comment now holds it open indefinitely,
            # leaking the request task and the buffer's file handle for as long
            # as the tab lives. So bound the wait and fail the turn.
            if idle_timeout > 0 and time.monotonic() - last_reply_at > idle_timeout:
                await _fail_unresponsive_worker(
                    buffer=buffer, on_event=on_event, idle_seconds=idle_timeout,
                    conversation_id=conversation_id, corr=corr,
                )
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
                    # Classify once; the SSE frame and the caller's persisted
                    # events log must carry the same (code, message).
                    code, message = remote_turn_error(data.get("error"))
                    data = {**data, "code": code, "message": message}
                if on_event is not None and kind in (
                    "turn_delta", "turn_memory", "turn_completed", "turn_failed"
                ):
                    on_event(kind, data)
                if kind == "turn_delta":
                    await buffer.append("sse", {"sse": _sse(
                        "response.output_text.delta",
                        {"type": "response.output_text.delta", "delta": data.get("text", "")})})
                    continue
                if kind == "turn_completed":
                    await buffer.append("sse", {"sse": _sse("response.completed",
                                                            {"type": "response.completed"})})
                    await buffer.close("completed")
                    return
                if kind == "turn_failed":
                    logger.warning(
                        "Remote turn failed conversation=%s correlation_id=%s error=%s",
                        conversation_id, corr, data.get("error"),
                    )
                    # Same response.failed frame the in-process path emits, so
                    # the client renders the existing error card (a bare
                    # response.completed+error renders as nothing).
                    await buffer.append("sse", {"sse": response_failed_sse(
                        data["message"], data["code"])})
                    await buffer.close("error")
                    return
