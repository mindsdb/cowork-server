"""Remote turn producer: enqueue a whole-turn job, tail the reply stream,
and emit SSE frames for the turn.

Emits `response.created`, one `response.output_text.delta` per `turn_delta`
reply, and closes the buffer only on a terminal reply (`turn_completed` or
`turn_failed`) with a final `response.completed` frame. No S3, no
heartbeat. Wired into the responses handler in a later task.
"""
from __future__ import annotations

import json
import uuid

from cowork.turnqueue.redis_client import get_redis
from cowork.common.settings.app_settings import TurnQueueSettings


def _new_correlation_id() -> str:
    return str(uuid.uuid4())


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


async def produce_remote_turn(*, conversation_id: str, org_id: str | None,
                              user_id: str | None, input_text: str,
                              model: str | None, buffer,
                              history: list | None = None) -> None:
    settings = TurnQueueSettings()
    r = get_redis()
    corr = _new_correlation_id()
    reply_stream = f"scratchpad:reply:{conversation_id}"

    job = {
        "op": "anton_turn",
        "conversation_id": conversation_id,
        "correlation_id": corr,
        "reply_stream": reply_stream,
        "organization_id": org_id,
        "user_id": user_id,
        "params": {"input": input_text, "workspace_path": "/workspace",
                   "model": model, "history": history or []},
    }
    await r.xadd(settings.jobs_stream, {"payload": json.dumps(job)})
    await buffer.append("sse", {"sse": _sse("response.created",
                                            {"type": "response.created"})})

    last_id = "0-0"
    while True:
        resp = await r.xread({reply_stream: last_id}, count=10, block=5000)
        if not resp:
            continue
        for _stream, entries in resp:
            for entry_id, fields in entries:
                last_id = entry_id
                payload = json.loads(fields["payload"])
                if payload.get("correlation_id") != corr:
                    continue
                kind = payload["kind"]
                data = payload.get("data") or {}
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
                    await buffer.append("sse", {"sse": _sse(
                        "response.completed",
                        {"type": "response.completed", "error": data.get("error")})})
                    await buffer.close("error")
                    return
