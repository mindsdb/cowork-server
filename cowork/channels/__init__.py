"""Channels — an Anton-specific app extension surfaced through cowork-server.

Every channel message routes to the Anton runtime. This is intentional and must
NOT become harness-generic: the cowork UI's agent hotswitch (Anton/Hermes) does
not apply to channels. Do not thread ``selected_agent`` / ``active_harness`` /
``provider`` / ``harness`` through this package — channels are Anton-only by
design, and those names invite Hermes into a path that should never see it.

This package owns the host-side plugin contract and registry. The dispatch
engine itself (router, entities, policy) is shared from ``anton.core.dispatch``.
"""
from __future__ import annotations

from cowork.channels.plugin import (
    AdapterFactory,
    ChannelPlugin,
    CredentialField,
    CredentialSchema,
    OAuthSpec,
    WebhookRoute,
)
from cowork.channels.registry import (
    PluginRegistry,
    default_registry,
    get_registry,
    load_first_party_plugins,
)

# The single agent identity channels route to. Used by the runtime (later
# slice); defined here so the Anton-only invariant has a concrete anchor.
ANTON_CHANNEL_AGENT_ID = "anton"

__all__ = [
    "ANTON_CHANNEL_AGENT_ID",
    "AdapterFactory",
    "ChannelPlugin",
    "CredentialField",
    "CredentialSchema",
    "OAuthSpec",
    "WebhookRoute",
    "PluginRegistry",
    "default_registry",
    "get_registry",
    "load_first_party_plugins",
]
