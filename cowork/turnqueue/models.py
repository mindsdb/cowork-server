"""Pydantic models for the turn-queue Redis envelope.

Cross-repo duplicate: these mirror scratchpad-controller's
`ScratchpadJobPayload` / `ScratchpadReplyPayload` (src/scratchpad_controller/
payload.py) field-for-field. cowork cannot import that package directly, so
this is a deliberate hand-kept-in-sync duplicate until a shared schema
package exists.
"""
from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StrictStr, field_validator, model_validator


#: Ceiling for ``TurnJob.deadline_ms``. 24h is far past any real turn and far
#: below any epoch value, so an epoch fails validation instead of silently
#: disabling the controller's timeout. Mirrors payload.MAX_DEADLINE_MS.
MAX_DEADLINE_MS = 24 * 60 * 60 * 1000
DATASOURCE_PROTOCOL_VERSION = 1
MAX_DATASOURCE_CONNECTIONS = 100
#: The most custom API connections one turn carries. The pod fetches every one
#: while its session starts, so the cap bounds that start (anton holds the same).
MAX_CUSTOM_API_CONNECTIONS = 25
_REQUIRED_REF_KEYS = frozenset({"connection_id", "credential_version"})
#: Names for the prompt only. anton's `anton/cloud_turn/contract.py` allows the
#: same two, cleans each value for the prompt and drops what it cannot use, so
#: this check enforces shape alone; the producer sends only values that pass it.
_OPTIONAL_REF_KEYS = frozenset({"label", "engine"})
#: Auth's cap on a connection's name.
_MAX_CONNECTION_LABEL_CHARS = 128
_CONNECTION_ENGINE_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,31}")

#: The ``turn_completed`` key that names the artifact folders the turn's tools
#: touched. anton holds the same name as ``ARTIFACTS_TOUCHED_KEY`` in
#: ``anton/cloud_turn/contract.py``, and scratchpad-controller as
#: ``ARTIFACTS_TOUCHED_KEY`` in ``src/scratchpad_controller/anton_turn.py``.
ARTIFACTS_TOUCHED_KEY = "artifacts_touched"
#: The most names that list carries. anton's ``_stream_turn`` leaves out a
#: longer list and scratchpad-controller relays none, each against its own
#: ``MAX_ARTIFACTS_TOUCHED`` in the files above; this server reads one as
#: absent (``TurnCompleted``).
MAX_ARTIFACTS_TOUCHED = 1000
#: The longest name that list may hold, this server's own bound: no folder
#: name is longer on the file systems artifacts live on. A list holding a
#: longer name reads as absent (``TurnCompleted``), and the shutdown line
#: that names a turn's artifacts leaves one out
#: (``cowork.db.units.drain_turn_end_records``).
MAX_ARTIFACT_NAME_CHARS = 255


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
        if (
            not isinstance(connection, dict)
            or not _REQUIRED_REF_KEYS <= set(connection) <= _REQUIRED_REF_KEYS | _OPTIONAL_REF_KEYS
        ):
            raise ValueError("datasource connection reference is invalid")
        if "label" in connection and not is_connection_label(connection["label"]):
            raise ValueError("datasource connection reference is invalid")
        if "engine" in connection and not is_connection_engine(connection["engine"]):
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


def _is_positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 1


def _validate_custom_api_block(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"turn_key", "connections"}:
        raise ValueError("custom_api must carry exactly turn_key and connections")
    if not isinstance(value["turn_key"], str) or not value["turn_key"]:
        raise ValueError("custom_api turn key is missing")
    connections = value["connections"]
    if not isinstance(connections, list) or not connections or len(connections) > MAX_CUSTOM_API_CONNECTIONS:
        raise ValueError("custom_api connections are invalid")
    seen: set[int] = set()
    for connection in connections:
        if not isinstance(connection, dict) or set(connection) != {
            "connection_id", "connector_id", "name", "credential_version",
        }:
            raise ValueError("custom_api connection reference is invalid")
        connection_id = connection["connection_id"]
        if (
            not _is_positive_int(connection_id)
            or not _is_positive_int(connection["credential_version"])
            or not isinstance(connection["connector_id"], str)
            or not connection["connector_id"]
            or not isinstance(connection["name"], str)
            or not connection["name"]
            or connection_id in seen
        ):
            raise ValueError("custom_api connection reference is invalid")
        seen.add(connection_id)


def is_connection_label(value: object) -> bool:
    """Whether ``value`` has the shape of a connection name the producer sends: a non-empty string within auth's cap."""
    return isinstance(value, str) and 0 < len(value) <= _MAX_CONNECTION_LABEL_CHARS


def is_connection_engine(value: object) -> bool:
    """Whether ``value`` has the shape of auth's connector id; ``fullmatch``, so a trailing newline fails."""
    return isinstance(value, str) and _CONNECTION_ENGINE_PATTERN.fullmatch(value) is not None


class TurnJob(BaseModel):
    """Mirror of scratchpad-controller ScratchpadJobPayload (job the controller consumes).

    ``params`` carries an ``llm`` block minted per turn by
    ``cowork.turnqueue.producer.mint_llm_block``:
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
        if "custom_api" in self.params:
            _validate_custom_api_block(self.params["custom_api"])
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


class TurnReplyHeader(BaseModel):
    """The field of a reply the relay reads before anything else: the turn it
    answers. Another turn's reply is skipped on this alone, whatever its kind."""

    correlation_id: str


class TurnReply(BaseModel):
    """Mirror of scratchpad-controller ScratchpadReplyPayload (reply cowork consumes)."""

    correlation_id: str
    # Must accept every kind the controller can publish: the reply loop validates
    # each of its own turn's entries unguarded, so a missing kind fails the turn
    # rather than being ignored. Kinds this build does nothing with are dropped
    # further down.
    kind: Literal["progress", "cell", "error", "turn_delta", "turn_step",
                  "turn_memory", "turn_skill", "turn_connector", "turn_history", "turn_compaction",
                  "turn_completed", "turn_failed"]
    data: dict[str, Any] = Field(default_factory=dict)
    # The reply's number within its turn, counting up by one. The controller
    # resends a reply whose XADD reply it lost, and the relay drops a reply
    # whose number repeats the last one it applied (producer.py). A
    # controller that numbers nothing sends none, and every reply is applied.
    n: int | None = None


class TurnJobHeader(BaseModel):
    """The field of a queued job the producer reads back after a job XADD
    whose reply it lost: the turn it queues."""

    correlation_id: str


class TurnCompleted(BaseModel):
    """The ``data`` of a ``turn_completed`` reply, as far as this server reads
    it. Written by anton's ``_stream_turn`` (``anton/cloud_turn/__main__.py``,
    documented in ``anton/cloud_turn/contract.py``) and relayed by
    scratchpad-controller's ``src/scratchpad_controller/anton_turn.py``.
    Untrusted: the pod writes it."""

    # The folder names of the artifacts the turn's tools touched; None for a
    # pod that sends no list.
    artifacts_touched: list[Annotated[StrictStr, Field(max_length=MAX_ARTIFACT_NAME_CHARS)]] | None = Field(
        default=None, alias=ARTIFACTS_TOUCHED_KEY, max_length=MAX_ARTIFACTS_TOUCHED,
    )
