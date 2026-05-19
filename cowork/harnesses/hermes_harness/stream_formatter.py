"""SSE event formatter for the Hermes harness.

Hermes is a request/response agent (no streaming). This formatter wraps the
single response dict yielded by HermesHarness.stream_response into the same
Responses-API SSE format used by the Anton formatter so the handler is agnostic.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Callable, Optional

from cowork.schemas.responses import Role


async def format_hermes_stream(
    event_stream: AsyncIterator,
    model: str,
    event_sink: Optional[Callable[[str, dict], None]] = None,
) -> AsyncIterator[str]:
    resp_id = f"resp-{uuid.uuid4().hex[:12]}"
    msg_id = f"msg-{uuid.uuid4().hex[:12]}"
    seq = 0

    def _event(event_type: str, data: dict) -> str:
        if "at_ms" not in data:
            data["at_ms"] = int(time.time() * 1000)
        if event_sink is not None:
            try:
                event_sink(event_type, data)
            except Exception:
                pass
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    seq += 1
    yield _event("response.created", {
        "type": "response.created",
        "sequence_number": seq,
        "response": {"id": resp_id, "model": model, "status": "created"},
    })

    accumulated: list[str] = []
    async for item in event_stream:
        item_type = item.get("type", "")

        if item_type == "delta":
            delta = item["delta"]
            if delta:
                accumulated.append(delta)
                seq += 1
                yield _event("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "sequence_number": seq,
                    "item_id": msg_id,
                    "delta": delta,
                })

        elif item_type == "thought.tool_call.start":
            seq += 1
            yield _event("response.in_progress", {
                "type": "response.in_progress",
                "sequence_number": seq,
                "thought_role": Role.thought_tool_call_start.value,
                "content": item["name"],
                "args": item.get("args"),
                "tool_use_id": item["tool_call_id"],
            })

        elif item_type == "thought.tool_call.end":
            seq += 1
            yield _event("response.in_progress", {
                "type": "response.in_progress",
                "sequence_number": seq,
                "thought_role": Role.thought_tool_call_end.value,
                "content": item.get("result", "")[:65536],
                "tool_use_id": item["tool_call_id"],
            })

        elif item_type == "thought.tool_call.progress":
            seq += 1
            yield _event("response.in_progress", {
                "type": "response.in_progress",
                "sequence_number": seq,
                "thought_role": Role.thought_tool_call_progress.value,
                "content": item.get("preview") or item.get("name", ""),
                "event": item.get("event"),
                "tool_name": item.get("name"),
            })

        elif item_type == "thought.progress":
            seq += 1
            yield _event("response.in_progress", {
                "type": "response.in_progress",
                "sequence_number": seq,
                "thought_role": Role.thought_progress.value,
                "content": item["content"],
                "subtype": item.get("subtype"),
            })

        else:
            # Final result dict — fallback if stream_callback was never fired
            fallback = item.get("final_response", "")
            if fallback and not accumulated:
                accumulated.append(fallback)
                seq += 1
                yield _event("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "sequence_number": seq,
                    "item_id": msg_id,
                    "delta": fallback,
                })

    final_text = "".join(accumulated)

    seq += 1
    yield _event("response.completed", {
        "type": "response.completed",
        "sequence_number": seq,
        "response": {
            "id": resp_id,
            "model": model,
            "status": "completed",
            "output": [{
                "id": msg_id,
                "status": "completed",
                "content": [{"text": final_text}],
            }],
        },
    })
