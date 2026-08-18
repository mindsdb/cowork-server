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
    # Whether this harness is offered in org (multi-tenant) deployments.
    # Defaults True for implementers that don't set it (see _harness_options).
    supports_org_mode: bool

    async def stream_response(
        self,
        *,
        conversation: Conversation,
        input: list[TextInputBlock | FileInputBlock],
        # Per-conversation model pick (the composer's dropdown), overriding
        # this harness's planning/router/coding roles for just this call.
        # None keeps the account-wide default for every role, as before.
        model: str | None = None,
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
    # available_harness_ids() only HIDES a single-tenant harness from the
    # picker. The chosen harness is an org-scoped user setting, so a stored row
    # naming one still resolved here and ran it: Hermes keeps skills, memory and
    # sessions under an unscoped cowork_home() path, which on an org deployment
    # is shared storage every organization can read. Enforce the flag where the
    # instance is actually built, not only where the list is rendered.
    from cowork.common.settings.app_settings import get_app_settings

    if get_app_settings().tenancy_mode == "org" and not getattr(cls, "supports_org_mode", True):
        raise ValueError(f"Harness {name!r} does not support multi-tenant deployments.")
    return cls()


def available_harness_ids() -> list[str]:
    """Registered harness ids offered to users. In org mode, harnesses that
    don't support multi-tenancy are hidden (Anton-only for now); the getattr
    default keeps every other harness available."""
    from cowork.common.settings.app_settings import get_app_settings

    org_mode = get_app_settings().tenancy_mode == "org"
    return [
        hid for hid, cls in _registry.items()
        if not org_mode or getattr(cls, "supports_org_mode", True)
    ]
