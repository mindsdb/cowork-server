import time
from typing import Any
import uuid
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field, model_serializer

from cowork.schemas.connectors import DisabledConnection


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    # Thought events (tool activity visible to client).
    # These are specific to the Anton harness at the moment.
    thought_scratchpad_start = "thought.scratchpad.start"
    thought_scratchpad_progress = "thought.scratchpad.progress"
    thought_scratchpad_result = "thought.scratchpad.result"
    thought_scratchpad_end = "thought.scratchpad.end"
    thought_memorize_start = "thought.memorize.start"
    thought_memorize_end = "thought.memorize.end"
    thought_recall_start = "thought.recall.start"
    thought_recall_end = "thought.recall.end"
    thought_progress = "thought.progress"
    thought_context_compacted = "thought.context_compacted"
    # General tool call events (relevant to other harnesses).
    thought_tool_call_start = "thought.tool_call.start"
    thought_tool_call_progress = "thought.tool_call.progress"
    thought_tool_call_end = "thought.tool_call.end"


class ContentType(str, Enum):
    text = "input_text"
    file = "input_file"


class Content(BaseModel):
    type: ContentType
    text: str | None = None
    file_id: str | None = None


class Message(BaseModel):
    role: Role
    # `list[dict]` carries raw tool_use / tool_result blocks verbatim into
    # LLM history (their id/name/input would be lost if coerced into Content).
    content: dict | BaseModel | str | list[Content] | list[dict] | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    @model_serializer
    def _serialize(self) -> dict[str, Any]:
        """Omit None tool fields from serialization."""
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls is not None:
            data["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            data["name"] = self.name
        return data


class ResponseOutputContent(BaseModel):
    # TODO: There are some other types that have not been included here. Are they needed?
    type: str = "output_text"
    text: str


class ResponseStatus(str, Enum):
    # TODO: Are there other statuses that we are interested in?
    created = "created"
    in_progress = "in_progress"
    completed = "completed"


class ResponseOutput(BaseModel):
    type: str = "message"
    id: str  # The ID here should be the same as the Message stored in the database.
    status: ResponseStatus
    role: str = Role.assistant.value
    content: list[ResponseOutputContent]


class Response(BaseModel):
    # TODO: Should we look at storing the response ID in the database?
    # This is to allow previous_response_id chains.
    id: str = Field(default_factory=lambda: f"resp-{uuid.uuid4()}")
    object: str = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    status: ResponseStatus
    error: str | None = None
    model: str | None = None  # The model can be set to None here because of the reasons mentioned in the request schema.
    output: list[ResponseOutput] = Field(default_factory=list)
    # TODO: There are some other that have not been included here. Are they needed?


class StreamingResponseEvent(str, Enum):
    created = "response.created"
    in_progress = "response.in_progress"
    output_text_delta = "response.output_text.delta"
    completed = "response.completed"


class ResponseDelta(BaseModel):
    item_id: str
    type: str = StreamingResponseEvent.output_text_delta.value
    delta: str


class StreamingResponse(BaseModel):
    type: StreamingResponseEvent
    sequence_number: int
    response: Response | ResponseDelta


class ResponsesRequest(BaseModel):
    input: str | list[Message] | None = Field(
        default=None, description="Input for the responses request, either a string or a list of messages"
    )
    # TODO: The OpenAI API also supports a conversation object?
    conversation: str | None = Field(
        default=None,
        description="Conversation ID for the responses request, if not provided, a new conversation will be created",
    )
    project: str | None = Field(
        default=None,
        description="Project name for a new conversation",
    )
    project_id: UUID | None = Field(
        default=None,
        description="Project ID for a new conversation",
    )
    # In OpenAI's Responses API, the model is required.
    # However, we currently do not allow this to be specified at the time of making the request,
    # via the Cowork UI. Instead, the model is retrieved from the user provided settings.
    model: str | None = Field(
        default=None,
        description="Model name for the chat completion request"
    )
    stream: bool | None = Field(
        default=False,
        description="Whether the chat completion request is streaming or not",
    )
    # TODO(migration): Per MIGRATION.md, attachment_ids should be removed.
    # The client should instead send input_file content blocks in the input
    # field (the handler already supports this path). This field is a compat
    # bridge for the current client which uploads via /v1/attachments/.
    attachment_ids: list[str] | None = Field(
        default=None,
        description="IDs of uploaded attachments (images, files) to include with this message",
    )
    disabled_connections: list[DisabledConnection] | None = Field(
        default=None,
        description="Connections to exclude from this turn (client sends on every request)",
    )
    # Which UI surface the turn came from. "browser" = the Browser Agent dock:
    # the handler injects a live <browser-context> block (copilot guidance +
    # open-tab state) into the LLM input only — never persisted, rebuilt per
    # turn. Absent/None = ordinary chat turn, no injection.
    surface: str | None = Field(
        default=None,
        description="UI surface this turn originated from (e.g. 'browser' for the Browser Agent dock)",
    )
    # Generic observability pass-through. Whatever a caller puts here is
    # forwarded verbatim to the harness → LLM router → Langfuse trace, so new
    # eval / telemetry use-cases can attach data without changing this server or
    # the harness. `trace_tags` become Langfuse trace tags (indexed, filterable
    # — e.g. an eval run id); `trace_metadata` becomes free-form trace metadata.
    # Both are optional and ignored by harnesses that don't emit traces.
    trace_tags: list[str] | None = Field(
        default=None,
        description="Tags to attach to this turn's LLM trace (e.g. an eval run id).",
    )
    trace_metadata: dict[str, str] | None = Field(
        default=None,
        description="Arbitrary key/value metadata to attach to this turn's LLM trace.",
    )
