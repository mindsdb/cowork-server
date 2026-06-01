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
from cowork.channels.webhooks import (
    SignatureError,
    WebhookAck,
    WebhookBridge,
    WebhookHandshake,
    build_channel_webhook_router,
    drain_background_tasks,
)

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
    "SignatureError",
    "WebhookAck",
    "WebhookBridge",
    "WebhookHandshake",
    "build_channel_webhook_router",
    "drain_background_tasks",
]
