from __future__ import annotations

from urllib.parse import urlparse

from sqlmodel import Session

from cowork.channels.lifecycle import LifecycleContext, LifecycleError, LifecycleResult
from cowork.channels.plugin import ChannelPlugin
from cowork.channels.registry import PluginRegistry, get_registry
from cowork.common.settings.app_settings import get_app_settings
from cowork.services.channels import ChannelConfigService, UnknownChannelError

_LOCAL_ENVS = {"local", "dev", "development", "test"}
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def validate_public_base_url(raw: str, *, allow_insecure_local: bool) -> str:

    base = (raw or "").strip()
    if not base:
        raise LifecycleError(409, "public base URL is not configured; cannot register a webhook")
    parsed = urlparse(base)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise LifecycleError(
            400, "public base URL is malformed; expected an absolute http(s) URL with a host"
        )
    if parsed.scheme != "https":
        host = (parsed.hostname or "").lower()
        if not (host in _LOCAL_HOSTS and allow_insecure_local):
            raise LifecycleError(
                400,
                "public base URL must use https (http is allowed only for "
                "localhost in a local/dev environment)",
            )

    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")


class LifecycleNotImplementedError(Exception):
    """The channel exists but has no setup/teardown lifecycle (→ 501)."""


class ChannelLifecycleService:
    def __init__(self, session: Session, adapters, registry: PluginRegistry | None = None) -> None:
        self.session = session
        self.adapters = adapters
        self.registry = registry if registry is not None else get_registry()
        self.config = ChannelConfigService(session, registry=self.registry)

    async def setup(self, channel_type: str) -> LifecycleResult:
        plugin = self._require_lifecycle(channel_type)
        webhook_url = self._resolve_webhook_url(plugin)
        return await plugin.lifecycle.setup(self._context(plugin, webhook_url))

    async def teardown(self, channel_type: str) -> LifecycleResult:
        plugin = self._require_lifecycle(channel_type)
        return await plugin.lifecycle.teardown(self._context(plugin, None))

    def _require_lifecycle(self, channel_type: str) -> ChannelPlugin:
        plugin = self.registry.get(channel_type)
        if plugin is None:
            raise UnknownChannelError(channel_type)
        if plugin.lifecycle is None:
            raise LifecycleNotImplementedError(channel_type)
        return plugin

    def _context(self, plugin: ChannelPlugin, webhook_url: str | None) -> LifecycleContext:
        channel_type = plugin.channel_type

        def persist(values: dict[str, str]) -> None:
            self.config.set_config(channel_type, values)

        async def refresh() -> bool:
            return await self.adapters.refresh(channel_type, session=self.session)

        async def remove() -> None:
            await self.adapters.remove(channel_type)

        return LifecycleContext(
            channel_type=channel_type,
            webhook_url=webhook_url,
            credentials=self.config.load_credentials(channel_type),
            persist_credentials=persist,
            refresh_adapter=refresh,
            remove_adapter=remove,
        )

    def _resolve_webhook_url(self, plugin: ChannelPlugin) -> str:
        settings = get_app_settings()
        base = validate_public_base_url(
            settings.public_base_url,
            allow_insecure_local=settings.env in _LOCAL_ENVS,
        )
        path = plugin.webhooks[0].path if plugin.webhooks else ""
        return f"{base}/api/v1/channels/{plugin.channel_type}{path}"
