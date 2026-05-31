"""Channel plugin contract — the declarative shape a channel ships.

A :class:`ChannelPlugin` bundles everything the host needs to expose a channel
without per-channel route/config/credential boilerplate. The host (later
slices) reads these descriptors to generate config endpoints, webhook routes,
OAuth routes, credential masking, and adapter wiring. The plugin itself only
provides the channel-specific pieces: a credential schema, a factory that
builds the adapter from resolved credentials, and route descriptors.

Channels are an Anton-specific app extension — the ``factory`` returns an
``anton.core.dispatch.ChannelAdapter``. That symbol is imported only under
``TYPE_CHECKING`` so this module does not require the dispatch engine to be
present at import time (the pinned anton revision may predate it); annotations
are strings under ``from __future__ import annotations`` and never evaluated at
runtime.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anton.core.dispatch import ChannelAdapter


# Resolved credentials in, a ready (not yet ``setup``-ed) adapter out — or
# ``None`` when the channel is not configured. The host loads secrets from its
# own store and injects them, so plugins never reach into the vault directly.
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

    The host generates the route under ``/channels/{channel_type}{path}`` (later
    slice). Handshake, signature verification, and inbound parsing live on the
    adapter/bridge, not here — this descriptor only declares the HTTP surface.
    ``needs_raw_body`` tells the host to pass the unparsed request bytes through
    (signature verification needs the exact payload).
    """

    path: str
    methods: tuple[str, ...] = ("POST",)
    needs_raw_body: bool = True
    # Discriminator for channels that expose more than one ingress route
    # (e.g. Slack "events" vs "interactions"); the bridge uses it to dispatch.
    name: str | None = None


@dataclass(frozen=True)
class OAuthSpec:
    """Declares the OAuth routes + scopes for channels that install via OAuth.

    Data-only for now: the host uses it to mount ``start``/``callback`` routes
    (later slice). The authorize-URL build and token exchange are implemented by
    the OAuth-using plugin's adapter when those channels land (Slack/Discord),
    so this stays a lean descriptor rather than carrying callables.
    """

    start_path: str = "/oauth/start"
    callback_path: str = "/oauth/callback"
    scopes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChannelPlugin:
    """Everything the host needs to expose one channel.

    First-party plugins live under :mod:`cowork.channels.plugins`; each module
    exposes a module-level ``plugin: ChannelPlugin`` that discovery registers.
    The same shape also fits external entry-point plugins (``pkg:plugin``) if
    that seam is opened later.
    """

    channel_type: str
    display_name: str
    factory: AdapterFactory
    credentials: CredentialSchema
    webhooks: tuple[WebhookRoute, ...] = ()
    oauth: OAuthSpec | None = None
    connector_spec: dict[str, Any] | None = field(default=None)
