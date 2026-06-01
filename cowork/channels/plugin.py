from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anton.core.dispatch import ChannelAdapter

    from cowork.channels.lifecycle import ChannelLifecycle


AdapterFactory = Callable[[Mapping[str, str]], Awaitable["ChannelAdapter | None"]]


@dataclass(frozen=True)
class CredentialField:
    """One credential a channel needs (bot token, signing secret, …).

    ``secret`` drives masking: secret values are stored in the host secret
    store and never returned over the API (surfaced as ``is_set`` / value-null).
    Non-secret fields (e.g. a webhook URL) may be echoed back.
    """

    name: str
    label: str
    secret: bool = True
    required: bool = True
    description: str | None = None


@dataclass(frozen=True)
class CredentialSchema:
    """The set of credentials a channel accepts."""

    fields: tuple[CredentialField, ...] = ()

    def get(self, name: str) -> CredentialField | None:
        return next((f for f in self.fields if f.name == name), None)

    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]

    def secret_field_names(self) -> list[str]:
        return [f.name for f in self.fields if f.secret]

    def required_field_names(self) -> list[str]:
        return [f.name for f in self.fields if f.required]


@dataclass(frozen=True)
class WebhookRoute:
    """Declares one HTTP ingress route the host will mount for the channel.
    """

    path: str
    methods: tuple[str, ...] = ("POST",)
    needs_raw_body: bool = True
    name: str | None = None


@dataclass(frozen=True)
class OAuthSpec:
    """Declares the OAuth routes + scopes for channels that install via OAuth.
    """

    start_path: str = "/oauth/start"
    callback_path: str = "/oauth/callback"
    scopes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChannelPlugin:
    """Everything the host needs to expose one channel.
    """

    channel_type: str
    display_name: str
    factory: AdapterFactory
    credentials: CredentialSchema
    webhooks: tuple[WebhookRoute, ...] = ()
    oauth: OAuthSpec | None = None
    connector_spec: dict[str, Any] | None = field(default=None)
    lifecycle: ChannelLifecycle | None = field(default=None)
