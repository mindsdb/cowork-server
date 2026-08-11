"""Pydantic models for the turn-queue Redis envelope.

Cross-repo duplicate: these mirror scratchpad-controller's
`ScratchpadJobPayload` / `ScratchpadReplyPayload` (src/scratchpad_controller/
payload.py) field-for-field. cowork cannot import that package directly, so
this is a deliberate hand-kept-in-sync duplicate until a shared schema
package exists.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TurnJob(BaseModel):
    """Mirror of scratchpad-controller ScratchpadJobPayload (job the controller consumes).

    ``params`` carries an ``llm`` block minted per turn by
    ``cowork.turnqueue.producer._mint_llm_block``:
    ``{"provider": "minds-cloud", "api_key": <short-TTL mdb_ turn key>,
    "base_url": <MindsHub chat base URL>}``. MVP is MindsHub-inference-only, so
    this is the only provider/credential shape carried here. The key is scoped
    to this turn's correlation id and expires within minutes (see
    ``TurnQueueSettings.turn_key_ttl_seconds``); it travels cowork -> Redis job
    -> controller -> exec stdin -> anton and must never be placed in the pod
    env or argv (pods are reused across turns).
    """

    op: str
    conversation_id: str
    correlation_id: str
    reply_stream: str
    organization_id: str | None = None
    user_id: str | None = None
    deadline_ms: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class TurnReply(BaseModel):
    """Mirror of scratchpad-controller ScratchpadReplyPayload (reply cowork consumes)."""

    correlation_id: str
    kind: Literal["progress", "cell", "error", "turn_delta", "turn_memory",
                  "turn_completed", "turn_failed"]
    data: dict[str, Any] = Field(default_factory=dict)
