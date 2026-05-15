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

    final_text = ""
    async for result in event_stream:
        final_text = result.get("final_response", "")

    if final_text:
        seq += 1
        yield _event("response.output_text.delta", {
            "type": "response.output_text.delta",
            "sequence_number": seq,
            "item_id": msg_id,
            "delta": final_text,
        })

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
