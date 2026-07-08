"""Normalized request/event model shared by every CLI-based coworker.

Every CliConfig/BaseCliHarness subclass translates its CLI's own wire
format (stream-json, NDJSON, whatever) into these shapes on the way out,
and receives a ConversationRequest on the way in. The rest of cowork
(SSE formatting, the conversation engine) only ever sees these — it
never needs to know which CLI produced them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

EventType = Literal[
    "text_chunk", "thinking_chunk", "tool_call", "tool_result",
    "progress", "error", "completed", "cancelled",
]


@dataclass
class NormalizedEvent:
    type: EventType
    text: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_args: Any = None
    tool_result: str | None = None
    detail: str | None = None
    final_text: str | None = None


@dataclass
class ConversationRequest:
    conversation_id: str
    prompt: str
    cwd: str
    profile: dict
    """Execution-profile dict — {"model": ..., "skipPermissions": ...},
    shape defined by the coworker's own configuration_schema(). Held
    inline per-request for now (not yet a persisted ExecutionProfile
    table — see minds-multisource-hub memory for the phasing note)."""
    resume: bool
    """Whether this conversation already has a prior turn with this
    coworker — tells the harness whether to resume a session or start
    fresh."""
