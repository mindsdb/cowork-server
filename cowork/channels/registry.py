"""Channel plugin registry + first-party discovery.

The registry holds :class:`ChannelPlugin` descriptors keyed by ``channel_type``.
First-party plugins live under :mod:`cowork.channels.plugins`; each module
exposes a module-level ``plugin`` attribute, and :func:`load_first_party_plugins`
imports every module in that package and registers the ones it finds. Discovery
is side-effect-free at the contract level — a module is only registered if it
declares a ``plugin``, so importing the package alone registers nothing.

External entry-point discovery (third-party ``pip``-installed channels) is
intentionally NOT wired here: loading arbitrary external code into the server
is a trust boundary deferred to a later, opt-in step.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil

from cowork.channels.plugin import ChannelPlugin

_log = logging.getLogger(__name__)


class PluginRegistry:
    """In-process map of ``channel_type`` → :class:`ChannelPlugin`."""

    def __init__(self) -> None:
        self._plugins: dict[str, ChannelPlugin] = {}

    def register(self, plugin: ChannelPlugin) -> None:
        """Register a plugin. Re-registering a channel type replaces the prior
        one (supports reload) but is logged, since for first-party plugins a
        duplicate ``channel_type`` is usually a mistake."""
        if plugin.channel_type in self._plugins:
            _log.warning("channel plugin %r already registered; replacing", plugin.channel_type)
        self._plugins[plugin.channel_type] = plugin

    def get(self, channel_type: str) -> ChannelPlugin | None:
        return self._plugins.get(channel_type)

    def all(self) -> list[ChannelPlugin]:
        return list(self._plugins.values())

    def channel_types(self) -> list[str]:
        return sorted(self._plugins)


# Process-wide default registry. The app wires discovery into it at startup
# (later slice); tests can construct their own PluginRegistry in isolation.
default_registry = PluginRegistry()


def get_registry() -> PluginRegistry:
    return default_registry


def load_first_party_plugins(registry: PluginRegistry | None = None) -> list[str]:
    """Import every module under :mod:`cowork.channels.plugins` and register
    any module-level ``plugin``.

    Returns the channel types loaded. A module that fails to import, or whose
    ``plugin`` is not a :class:`ChannelPlugin`, is logged and skipped — one
    broken plugin must not take down discovery.
    """
    target = registry if registry is not None else default_registry

    from cowork.channels import plugins as plugins_pkg

    loaded: list[str] = []
    for module_info in pkgutil.iter_modules(plugins_pkg.__path__):
        module_name = f"{plugins_pkg.__name__}.{module_info.name}"
        try:
            module = importlib.import_module(module_name)
        except Exception:
            _log.exception("failed to import channel plugin module %s", module_name)
            continue

        plugin = getattr(module, "plugin", None)
        if not isinstance(plugin, ChannelPlugin):
            _log.warning("plugin module %s has no module-level ChannelPlugin; skipping", module_name)
            continue

        target.register(plugin)
        loaded.append(plugin.channel_type)

    return loaded
