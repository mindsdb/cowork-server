"""Pydantic models for the turn-queue Redis envelope.

Cross-repo duplicate: these mirror scratchpad-controller's
`ScratchpadJobPayload` / `ScratchpadReplyPayload` (src/scratchpad_controller/
payload.py) field-for-field. cowork cannot import that package directly, so
this is a deliberate hand-kept-in-sync duplicate until a shared schema
package exists.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


#: Ceiling for ``TurnJob.deadline_ms``. 24h is far past any real turn and far
#: below any epoch value, so an epoch fails validation instead of silently
#: disabling the controller's timeout. Mirrors payload.MAX_DEADLINE_MS.
MAX_DEADLINE_MS = 24 * 60 * 60 * 1000
DATASOURCE_PROTOCOL_VERSION = 1
MAX_DATASOURCE_CONNECTIONS = 100


def _validate_datasource_block(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("datasource must be an object")
    if set(value) != {"protocol_version", "connections"}:
        raise ValueError("datasource contains unsupported fields")
    block_version = value.get("protocol_version")
    # `True == 1` and `1.0 == 1`, so an equality check alone accepts a block
    # that is not v1 and this then treats it as one. Same rule as the ids below.
    if (
        isinstance(block_version, bool)
        or not isinstance(block_version, int)
        or block_version != DATASOURCE_PROTOCOL_VERSION
    ):
        raise ValueError("unsupported datasource protocol version")
    connections = value.get("connections")
    if not isinstance(connections, list) or not connections or len(connections) > MAX_DATASOURCE_CONNECTIONS:
        raise ValueError("datasource connections are invalid")
    seen: set[int] = set()
    for connection in connections:
        if not isinstance(connection, dict) or set(connection) != {"connection_id", "credential_version"}:
            raise ValueError("datasource connection reference is invalid")
        connection_id = connection.get("connection_id")
        version = connection.get("credential_version")
        if (
            isinstance(connection_id, bool)
            or not isinstance(connection_id, int)
            or connection_id < 1
            or isinstance(version, bool)
            or not isinstance(version, int)
            or version < 1
            or connection_id in seen
        ):
            raise ValueError("datasource connection reference is invalid")
        seen.add(connection_id)


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
    #: Project the turn runs in. The pod joins the params' org-relative
    #: workspace path under its own mount root to reach
    #: ``projects/<name>/conversations/<conversation_id>/``.
    project_id: str | None = None
    #: How long the turn may run, in milliseconds. A duration, NOT an epoch
    #: timestamp: the controller reads it as a relative budget, so an epoch value
    #: would mean a ~57 year deadline and no timeout at all.
    deadline_ms: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _workspace_mode_is_declared(self) -> TurnJob:
        if self.op == "anton_turn_v2" and self.params.get("workspace_mode") not in ("persistent", "ephemeral"):
            raise ValueError("anton_turn_v2 requires a persistent or ephemeral workspace_mode")
        if "datasource" in self.params:
            _validate_datasource_block(self.params["datasource"])
        return self

    @field_validator("deadline_ms")
    @classmethod
    def _deadline_is_a_duration(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if v <= 0:
            raise ValueError("deadline_ms must be positive")
        if v > MAX_DEADLINE_MS:
            raise ValueError(
                f"deadline_ms={v} exceeds {MAX_DEADLINE_MS}ms; it is a duration, not an epoch timestamp"
            )
        return v


class TurnReply(BaseModel):
    """Mirror of scratchpad-controller ScratchpadReplyPayload (reply cowork consumes)."""

    correlation_id: str
    # Must accept every kind the controller can publish: the reply loop validates
    # each entry unguarded, so a missing kind fails the turn rather than being
    # ignored. Kinds this build does nothing with are dropped further down.
    kind: Literal["progress", "cell", "error", "turn_delta", "turn_step",
                  "turn_memory", "turn_skill", "turn_history", "turn_compaction",
                  "turn_completed", "turn_failed"]
    data: dict[str, Any] = Field(default_factory=dict)
