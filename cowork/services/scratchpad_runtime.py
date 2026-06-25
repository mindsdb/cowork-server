"""Scratchpad runtime pool — adapter for anton's `launch_artifact_backend`.

Ported from `cowork/server/anton_api/scratchpad_runtime.py`. We only
need the `WorkspaceScopedPool` adapter and the underlying `_pads`
registry — full /v1/scratchpad/* endpoints are not exposed by
cowork-server. The pool gives `launch_artifact_backend` access to a
slug-keyed Python venv at `<workspace>/.anton/scratchpad-venvs/<slug>/`
so artifact backends share dependencies provisioned by the agent
during artifact creation.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Optional


MAX_PADS = int(os.environ.get("ANTON_SERVER_MAX_PADS", "5"))


_pads: dict[str, object] = {}  # name -> LocalScratchpadRuntime
last_activity: float = time.time()


def _touch_activity() -> None:
    global last_activity
    last_activity = time.time()


def _resolve_workspace(workspace_path: Optional[str]) -> Path:
    return (
        Path(workspace_path).expanduser().resolve()
        if workspace_path
        else Path.cwd().resolve()
    )


def _resolve_coding(
    *,
    coding_provider: str,
    coding_model: str,
    coding_api_key: str,
    coding_base_url: str,
) -> tuple[str, str, str, str]:
    """Fill in any blank coding fields from AntonSettings."""
    from anton.config.settings import AntonSettings

    from cowork.services.providers import provider_base_url

    s = AntonSettings()
    provider = coding_provider or s.coding_provider or ""
    model = coding_model or s.coding_model or ""
    norm = provider.replace("_", "-")

    # Resolve the API key from the correct slot per provider. openai, gemini,
    # and openai-compatible all share the single openai_api_key slot; anthropic
    # and minds-cloud have their own dedicated slots.
    if coding_api_key:
        api_key = coding_api_key
    elif norm == "anthropic":
        api_key = s.anthropic_api_key or ""
    elif norm == "minds-cloud":
        api_key = s.minds_api_key or ""
    else:  # openai, gemini, openai-compatible
        api_key = s.openai_api_key or ""

    # Derive the base URL deterministically per provider (see
    # providers.provider_base_url): openai/gemini never inherit the shared
    # openai_base_url slot, so a stale value left by another provider can't
    # misroute this key. An explicit coding_base_url always wins. Empty string
    # means "let anton's OpenAIProvider use its SDK default host".
    base_url = coding_base_url or provider_base_url(
        norm, openai_base_url=s.openai_base_url or "", minds_url=s.minds_url
    ) or ""

    # anton's scratchpad (scratchpad_boot.py) only understands "openai" /
    # "openai-compatible" → OpenAIProvider; every other string falls through to
    # AnthropicProvider. minds-cloud and gemini are OpenAI-compatible gateways,
    # so present them as "openai-compatible" (with their correct base above) —
    # otherwise the scratchpad would silently hit Anthropic with the wrong key.
    if norm in ("minds-cloud", "gemini"):
        provider = "openai-compatible"

    return provider, model, api_key, base_url


def _make_runtime(
    name: str,
    *,
    workspace_path: Optional[str],
    coding_provider: str,
    coding_model: str,
    coding_api_key: str,
    coding_base_url: str,
):
    from anton.core.backends.local import LocalScratchpadRuntime

    return LocalScratchpadRuntime(
        name,
        coding_provider=coding_provider,
        coding_model=coding_model,
        coding_api_key=coding_api_key,
        coding_base_url=coding_base_url,
        workspace_path=_resolve_workspace(workspace_path),
    )


def get(name: str):
    _touch_activity()
    return _pads.get(name)


def get_or_create(
    name: str,
    *,
    workspace_path: Optional[str] = None,
    coding_provider: str = "",
    coding_model: str = "",
    coding_api_key: str = "",
    coding_base_url: str = "",
):
    _touch_activity()
    if name in _pads:
        return _pads[name]
    if len(_pads) >= MAX_PADS:
        raise RuntimeError(
            f"Maximum concurrent scratchpads ({MAX_PADS}) reached. "
            f"Close an existing pad first."
        )
    provider, model, api_key, base_url = _resolve_coding(
        coding_provider=coding_provider,
        coding_model=coding_model,
        coding_api_key=coding_api_key,
        coding_base_url=coding_base_url,
    )
    pad = _make_runtime(
        name,
        workspace_path=workspace_path,
        coding_provider=provider,
        coding_model=model,
        coding_api_key=api_key,
        coding_base_url=base_url,
    )
    _pads[name] = pad
    return pad


def remove(name: str) -> None:
    _pads.pop(name, None)


def list_pads() -> list[str]:
    return list(_pads.keys())


async def close_all() -> None:
    for name in list(_pads):
        try:
            pad = _pads[name]
            await pad.close()
        except Exception:
            pass
    _pads.clear()


class WorkspaceScopedPool:
    """ScratchpadPoolLike adapter satisfying anton's launch_artifact_backend.

    Pins the workspace_path at construction so the helper sees a no-arg
    `venv_python(name)` / `get_or_create(name)` API. `venv_python`
    provisions the venv on demand via `LocalScratchpadRuntime._ensure_venv`
    — fast no-op when the deterministic disk path
    (`<workspace>/.anton/scratchpad-venvs/<name>/`) already exists,
    full `uv venv` / `python -m venv` creation when it doesn't. The
    runtime is registered in the module-level pool so subsequent
    `get_or_create` calls (e.g. for `install_packages`) reuse the same
    instance.

    Note: anton's `_ensure_venv` is a private sync method that returns
    None and populates `_venv_python` on success. We touch both
    private members because anton doesn't expose a public equivalent;
    using `start()` would be heavier (spawns the scratchpad subprocess)
    and unnecessary — `launch_artifact_backend` only needs the path
    to the venv's python binary.
    """

    def __init__(self, workspace_path: str):
        self._workspace_path = workspace_path

    async def venv_python(self, name: str) -> Optional[str]:
        pad = get_or_create(name, workspace_path=self._workspace_path)
        # Off-thread because `_ensure_venv` may shell out to `uv venv` /
        # `python -m venv` on first call (cold-start cost is seconds).
        await asyncio.to_thread(pad._ensure_venv)
        return pad._venv_python

    async def get_or_create(self, name: str):
        return get_or_create(name, workspace_path=self._workspace_path)
