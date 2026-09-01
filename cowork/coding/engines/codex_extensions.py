from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from anton.core.tools.skill_format import normalize_name
from cowork.coding.contracts import ExtensionEntry, ExtensionInventory
from cowork.coding.engines.codex_events import enum_value


def add_extension_response(
    inventory: ExtensionInventory,
    kind: str,
    response: Any,
    *,
    skill_roots: Sequence[str | Path] = (),
) -> None:
    if kind == "skills":
        _add_skills(inventory, response, skill_roots)
    elif kind == "mcp":
        _add_mcp_servers(inventory, response)
    elif kind == "hooks":
        _add_hooks(inventory, response)
    elif kind == "apps":
        _add_apps(inventory, response)
    else:
        _add_plugins(inventory, response)


def _add_skills(inventory: ExtensionInventory, response: Any, skill_roots: Sequence[str | Path]) -> None:
    roots = [Path(root).expanduser().resolve() for root in skill_roots]
    surviving: dict[str, ExtensionEntry] = {}
    for group in response.data:
        for skill in group.skills:
            entry = ExtensionEntry(
                id=skill.name,
                label=skill.name,
                description=skill.description or skill.short_description or "",
                status="enabled" if skill.enabled else "disabled",
                detail=enum_value(skill.scope),
                path=str(skill.path) if skill.path else None,
            )
            key = normalize_name(skill.name) or skill.name
            kept = surviving.get(key)
            if kept is None:
                surviving[key] = entry
            elif _under_any_root(entry, roots) and not _under_any_root(kept, roots):
                entry.supersedes = [kept, *kept.supersedes]
                kept.supersedes = []
                surviving[key] = entry
            else:
                kept.supersedes.append(entry)
    for entry in surviving.values():
        if entry.supersedes:
            other_scopes = ", ".join(dict.fromkeys(other.detail for other in entry.supersedes))
            entry.detail = f"{entry.detail} · also installed in {other_scopes}"
        inventory.skills.append(entry)


def _under_any_root(entry: ExtensionEntry, roots: Sequence[Path]) -> bool:
    if entry.path is None:
        return False
    path = Path(entry.path).expanduser().resolve()
    return any(path.is_relative_to(root) for root in roots)


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
