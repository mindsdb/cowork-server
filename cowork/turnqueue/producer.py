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
import uuid

from cowork.handlers.turn_errors import remote_turn_error, response_failed_sse
from cowork.services.providers import minds_chat_base_url
from cowork.turnqueue.auth_keys import mint_turn_key
from cowork.turnqueue.models import TurnJob, TurnReply
from cowork.turnqueue.redis_client import get_redis
from cowork.common.settings.app_settings import TurnQueueSettings, default_minds_api_host

logger = logging.getLogger(__name__)


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
    base_url = settings.minds_base_url or minds_chat_base_url(default_minds_api_host())
    block = {"provider": "minds-cloud", "api_key": api_key, "base_url": base_url}
    # Coding-model calls (verifier, scratchpad nested calls) must use a model
    # the turn key can pay for, not anton's built-in default. Same
    # unscoped-cache caveat as the mint above.
    from cowork.common.settings.user_settings import get_user_settings
    coding_model = get_user_settings().resolved_coding_model
    if coding_model:
        block["coding_model"] = coding_model
    return block


async def produce_remote_turn(*, conversation_id: str, org_id: str | None,
                              user_id: str | None, input_text: str,
                              model: str | None, buffer,
                              history: list | None = None,
                              harness_id: str | None = None,
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

    job = TurnJob(
        op="anton_turn",
        conversation_id=conversation_id,
        correlation_id=corr,
        reply_stream=reply_stream,
        organization_id=org_id,
        user_id=user_id,
        params={"input": input_text, "workspace_path": "/workspace",
                "model": model, "history": history or [], "llm": llm_block},
    )
    await r.xadd(settings.jobs_stream, {"payload": job.model_dump_json()})
    # conversation_id + harness mirror _inject_created on the in-process path:
    # the client learns the canonical id (it may have sent a non-UUID one).
    created = {"type": "response.created", "conversation_id": conversation_id}
    if harness_id:
        created["harness"] = harness_id
    await buffer.append("sse", {"sse": _sse("response.created", created)})

    last_id = "0-0"
    while True:
        resp = await r.xread({reply_stream: last_id}, count=10, block=5000)
        if not resp:
            continue
        for _stream, entries in resp:
            for entry_id, fields in entries:
                last_id = entry_id
                reply = TurnReply.model_validate_json(fields["payload"])
                if reply.correlation_id != corr:
                    continue
                kind = reply.kind
                data = reply.data or {}
                if kind == "turn_failed":
                    # Classify once; the SSE frame and the caller's persisted
                    # events log must carry the same (code, message).
                    code, message = remote_turn_error(data.get("error"))
                    data = {**data, "code": code, "message": message}
                if on_event is not None and kind in ("turn_delta", "turn_completed", "turn_failed"):
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
