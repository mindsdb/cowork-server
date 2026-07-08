from dataclasses import dataclass
from typing import AsyncIterator, Literal, Protocol
from typing_extensions import TypedDict

from cowork.models.conversation import Conversation
from cowork.models.skill import Skill


class TextInputBlock(TypedDict):
    type: Literal["text"]
    text: str


class FileInputBlock(TypedDict):
    type: Literal["file"]
    path: str
    filename: str


@dataclass(frozen=True)
class ChannelContext:
    """Origin of a turn that arrived via a chat channel (Telegram, Slack, ...).

    None on the harness call means the turn came from the desktop UI. Harnesses
    use it to swap desktop-oriented prompt guidance for chat/support-mode
    guidance; harnesses without channel-aware prompts accept and ignore it.
    """

    channel_type: str
    is_group: bool = False
    display_name: str | None = None
    instructions: str | None = None


class HarnessProvider(Protocol):
    id: str
    label: str
    formatter: AsyncIterator[str]

    async def stream_response(
        self,
        *,
        conversation: Conversation,
        input: list[TextInputBlock | FileInputBlock],
        # model: str,
        disabled_connections: list[dict] | None = None,
        # Optional observability pass-through (see ResponsesRequest). Forwarded
        # to the trace the harness emits; harnesses without tracing accept and
        # ignore them. Generic on purpose so callers can add eval/telemetry
        # data without changing the harness contract.
        trace_tags: list[str] | None = None,
        trace_metadata: dict[str, str] | None = None,
        channel_context: ChannelContext | None = None,
    ) -> AsyncIterator[str]:
        ...


_registry: dict[str, type[HarnessProvider]] = {}


def register(cls: type[HarnessProvider]) -> type[HarnessProvider]:
    _registry[cls.id] = cls
    return cls


def get_harness(name: str) -> HarnessProvider:
    cls = _registry.get(name)
    if cls is None:
        available = ", ".join(_registry) or "none"
        raise ValueError(f"Unknown harness {name!r}. Available: {available}")
    return cls()
