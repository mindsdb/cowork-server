from __future__ import annotations

from typing import Any

from cowork.coding.contracts import ExtensionEntry, ExtensionInventory
from cowork.coding.engines.codex_events import enum_value


def add_extension_response(inventory: ExtensionInventory, kind: str, response: Any) -> None:
    if kind == "skills":
        _add_skills(inventory, response)
    elif kind == "mcp":
        _add_mcp_servers(inventory, response)
    elif kind == "hooks":
        _add_hooks(inventory, response)
    elif kind == "apps":
        _add_apps(inventory, response)
    else:
        _add_plugins(inventory, response)


def _add_skills(inventory: ExtensionInventory, response: Any) -> None:
    for group in response.data:
        for skill in group.skills:
            inventory.skills.append(ExtensionEntry(
                id=skill.name,
                label=skill.name,
                description=skill.description or skill.short_description or "",
                status="enabled" if skill.enabled else "disabled",
                detail=enum_value(skill.scope),
                path=str(skill.path),
            ))


def _add_mcp_servers(inventory: ExtensionInventory, response: Any) -> None:
    for server in response.data:
        info = server.server_info
        label = (info.title or info.name) if info else server.name
        description = (info.description or "") if info else ""
        inventory.mcp_servers.append(ExtensionEntry(
            id=server.name,
            label=label,
            description=description,
            status=enum_value(server.auth_status),
            detail=f"{len(server.tools)} tools · {len(server.resources)} resources",
        ))


def _add_hooks(inventory: ExtensionInventory, response: Any) -> None:
    for group in response.data:
        for hook in group.hooks:
            inventory.hooks.append(ExtensionEntry(
                id=hook.key,
                label=hook.key,
                description=enum_value(hook.event_name),
                status="enabled" if hook.enabled else "disabled",
                detail=f"{enum_value(hook.source)} · {enum_value(hook.trust_status)}",
                path=str(hook.source_path),
            ))


def _add_apps(inventory: ExtensionInventory, response: Any) -> None:
    for app in response.apps:
        if app.callable:
            status = "callable"
        elif app.enabled:
            status = "enabled"
        else:
            status = "disabled"
        inventory.apps.append(ExtensionEntry(
            id=app.id,
            label=app.runtime_name or app.id,
            status=status,
        ))


def _add_plugins(inventory: ExtensionInventory, response: Any) -> None:
    for marketplace in response.marketplaces:
        for plugin in marketplace.plugins:
            if not plugin.installed:
                continue
            interface = plugin.interface
            label = (interface.display_name or plugin.name) if interface else plugin.name
            description = (interface.short_description or "") if interface else ""
            inventory.plugins.append(ExtensionEntry(
                id=plugin.id,
                label=label,
                description=description,
                status="enabled" if plugin.enabled else "disabled",
                detail=marketplace.name,
                path=str(marketplace.path) if marketplace.path else None,
            ))
