import time
from typing import Any
import uuid
from enum import Enum

from pydantic import BaseModel, Field, model_serializer


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    # Thought events (tool activity visible to client)
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


class ContentType(str, Enum):
    text = "input_text"
    file = "input_file"


class Content(BaseModel):
    type: ContentType
    text: str | None = None
    file_id: str | None = None


class Message(BaseModel):
    role: Role
    content: dict | BaseModel | str | list[Content] | None = None
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
    model: str
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
    model: str = Field(description="Model name for the chat completion request")
    stream: bool | None = Field(
        default=False,
        description="Whether the chat completion request is streaming or not",
    )
